"""S86q 锚定 v0 的验收测试（pytest，CPU-only，只用合成小夹具）。

覆盖（`stages/86q_anchor_v0.py` 的五条硬要求）：
  (a) 确定性：同一合成 ckpt 跑两次 `--anchor` 到两个目录 ⇒ 两份 manifest **逐位相同**、快照 sha 相同；
  (b) 篡改检测：把某个输入卡权重改一位后 `--verify` ⇒ **非零退出**并指名 weights_sha256；
  (c) 冻结开关：`--load-check --freeze` ⇒ zh_in/en_in 全部 `requires_grad is False` 且 sha 不变；
  (d) 契约 fail-closed：分词器 sha 与 manifest 不符 / 文件缺失 ⇒ `--verify` 必须失败；
  (e) `--cards all`：只改 `bridge.*`（思维卡）⇒ **mv 不变**、`thinking_card.sha256` 变；
      只改 `zh_out.*`（输出卡）⇒ **mv 变**；缺输出卡前缀 ⇒ 非零退出并指名；
      默认（不带 `--cards`）仍是 input 口径、字段集不变。

夹具全部合成（`tmp_path` 里的小分词器 + 7 行 pairs + 构造出的 ZHENModel ckpt，含带思维卡的变体），
不依赖 `logs/` 的 38MB 真 ckpt、不依赖 8000 步产物、不碰 GPU：
    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_anchor_v0.py -q
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "stages" / "86q_anchor_v0.py"
SEED = 1234
STEPS = 200
CKPT_NAME = f"synth_ckpt_s{SEED}_st{STEPS}.pt"
BRIDGE_CKPT_NAME = f"synth_ckpt_bridge_s{SEED}_st{STEPS}.pt"


def _cpu_env() -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""            # 本机无 GPU 也必须过（也不许碰 GPU）
    return env


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          cwd=str(cwd), env=_cpu_env(), capture_output=True, text=True)


def _make_fixture(dirpath: Path, name: str = CKPT_NAME, bridge: str = "none") -> Path:
    """合成 ckpt：构造一个真 ZHENModel，按 86g 的 dict(state=..., seed=..., ...) 格式落盘（不训练）。

    `bridge="parser"` 造带思维卡（`bridge.*`）的变体 —— `--cards all` 的思维卡夹具。
    """
    ck = dirpath / name
    src = (
        "import sys, torch\n"
        f"sys.path.insert(0, {str(ROOT / 'experiments' / 'meme_anchor')!r})\n"
        "from model_zh_en import build\n"
        f"m = build(seed={SEED}, bridge={bridge!r})\n"
        f"torch.save(dict(state=m.state_dict(), seed={SEED}, lam=0.0, maps=('AaBb', 'AbBa'),"
        f" loss_kind='none', steps={STEPS}, shared_out=False), {str(ck)!r})\n"
        "print('fixture-ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", src], cwd=str(dirpath),
                       env=_cpu_env(), capture_output=True, text=True)
    assert r.returncode == 0, f"夹具 ckpt 构造失败：\n{r.stdout}\n{r.stderr}"
    assert "fixture-ok" in r.stdout
    return ck


@pytest.fixture(scope="session")
def fx(tmp_path_factory) -> dict:
    """会话级夹具：分词器 + pairs + 合成 ckpt（都在 tmp 里，与 logs/ 无关）。"""
    d = tmp_path_factory.mktemp("anchorv0_fx")
    tok = d / "char_tokenizer.json"
    tok.write_text('{"fake": "tokenizer-for-anchor-v0-test", "hanzi_top": 7808}\n', encoding="utf-8")
    pairs = d / "pairs.jsonl"
    pairs.write_text("\n".join(f'["句{i}号测试", "sentence number {i}"]' for i in range(7)) + "\n",
                     encoding="utf-8")
    ck = _make_fixture(d)
    ck_bridge = _make_fixture(d, name=BRIDGE_CKPT_NAME, bridge="parser")
    return {"dir": d, "tokenizer": tok, "pairs": pairs, "ckpt": ck, "ckpt_bridge": ck_bridge}


def anchor_args(fx: dict, out: Path) -> list[str]:
    return ["--anchor",
            "--ckpts", str(fx["ckpt"]),
            "--out", str(out),
            "--tokenizer", str(fx["tokenizer"]),
            "--pairs", str(fx["pairs"]),
            "--note", "合成夹具（tests/test_anchor_v0.py）"]


def do_anchor(fx: dict, out: Path) -> Path:
    r = run_cli(*anchor_args(fx, out), cwd=fx["dir"])
    assert r.returncode == 0, f"--anchor 失败：\n{r.stdout}\n{r.stderr}"
    mp = out / "manifest.json"
    assert mp.is_file()
    return mp


# ---------------------------------------------------------------- (a) 确定性
def test_anchor_determinism(fx: dict, tmp_path: Path) -> None:
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    r1 = run_cli(*anchor_args(fx, out1), cwd=fx["dir"])
    assert r1.returncode == 0, f"第一次 --anchor 失败：\n{r1.stdout}\n{r1.stderr}"
    r2 = run_cli(*anchor_args(fx, out2), cwd=fx["dir"])
    assert r2.returncode == 0, f"第二次 --anchor 失败：\n{r2.stdout}\n{r2.stderr}"

    b1 = (out1 / "manifest.json").read_bytes()
    b2 = (out2 / "manifest.json").read_bytes()
    assert b1 == b2, "同一输入两次 --anchor 的 manifest 必须逐位相同"

    man = json.loads(b1.decode("utf-8"))
    name = man["cards"][0]["snapshot"]
    s1 = (out1 / name).read_bytes()
    s2 = (out2 / name).read_bytes()
    assert hashlib.sha256(s1).hexdigest() == hashlib.sha256(s2).hexdigest(), "快照 sha256 必须相同"
    assert hashlib.sha256(s1).hexdigest() == man["cards"][0]["snapshot_sha256"]
    # 只读：manifest 与快照都必须是 0444
    assert stat.S_IMODE((out1 / "manifest.json").stat().st_mode) == 0o444
    assert stat.S_IMODE((out1 / name).stat().st_mode) == 0o444
    # manifest 里绝不落绝对路径（tmp 目录、仓库根都不许出现）
    text = b1.decode("utf-8")
    assert str(tmp_path) not in text and str(ROOT) not in text, "manifest 含绝对路径"


# ---------------------------------------------------------------- (b) 篡改检测
def test_tamper_detected(fx: dict, tmp_path: Path) -> None:
    # 复制一份到夹具目录（所有 CLI 都以夹具目录为 cwd ⇒ 相对路径可解析），不碰会话夹具本体
    ck = fx["dir"] / f"tamper_copy_s{SEED}_st{STEPS}.pt"
    shutil.copy(fx["ckpt"], ck)
    fx_local = dict(fx, ckpt=ck)
    out = tmp_path / "out"
    do_anchor(fx_local, out)

    blob = torch.load(ck, map_location="cpu", weights_only=False)
    w = blob["state"]["zh_in.emb.weight"]
    w[0, 0] += 1.0                                 # ★ 改一个输入卡权重的一位
    torch.save(blob, ck)

    r = run_cli("--verify", str(out / "manifest.json"), cwd=fx["dir"])
    assert r.returncode != 0, f"篡改后 --verify 必须非零退出：\n{r.stdout}\n{r.stderr}"
    combined = r.stdout + r.stderr
    assert "weights_sha256" in combined, f"错误信息必须指名是哪一项：\n{combined}"


# ---------------------------------------------------------------- (c) 冻结开关
def test_load_check_freeze(fx: dict, tmp_path: Path) -> None:
    out = tmp_path / "out"
    mp = do_anchor(fx, out)

    r = run_cli("--load-check", str(mp), "--seed", str(SEED), "--freeze", cwd=fx["dir"])
    assert r.returncode == 0, f"--load-check --freeze 失败：\n{r.stdout}\n{r.stderr}"
    assert "freeze=True" in r.stdout
    assert "requires_grad=False" in r.stdout
    assert "sha=OK" in r.stdout
    assert "cards=1" in r.stdout
    assert f"c{1}" in r.stdout                      # 契约版本出现在摘要里

    r2 = run_cli("--load-check", str(mp), "--seed", str(SEED), cwd=fx["dir"])
    assert r2.returncode == 0, f"--load-check（不冻结）失败：\n{r2.stdout}\n{r2.stderr}"
    assert "freeze=False" in r2.stdout
    assert "requires_grad=True" in r2.stdout
    assert "sha=OK" in r2.stdout


# ---------------------------------------------------------------- (d) 契约 fail-closed
def test_contract_fail_closed(fx: dict, tmp_path: Path) -> None:
    out = tmp_path / "out"
    mp = do_anchor(fx, out)
    man = json.loads(mp.read_text(encoding="utf-8"))
    assert str(fx["tokenizer"]) not in json.dumps(man, ensure_ascii=False)  # 路径确实被相对化

    # (d-1) 分词器 sha 与 manifest 不符
    m1 = copy.deepcopy(man)
    m1["contract"]["tokenizer_sha256"] = "0" * 64
    p1 = out / "manifest_badsha.json"
    p1.write_text(json.dumps(m1, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                  encoding="utf-8")
    r1 = run_cli("--verify", str(p1), cwd=fx["dir"])
    assert r1.returncode != 0, f"sha 不符时 --verify 必须失败：\n{r1.stdout}\n{r1.stderr}"
    assert "contract.tokenizer_sha256" in (r1.stdout + r1.stderr)

    # (d-2) 分词器文件缺失
    m2 = copy.deepcopy(man)
    m2["contract"]["tokenizer"] = "no_such_tokenizer.json"
    p2 = out / "manifest_missing.json"
    p2.write_text(json.dumps(m2, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                  encoding="utf-8")
    r2 = run_cli("--verify", str(p2), cwd=fx["dir"])
    assert r2.returncode != 0, f"文件缺失时 --verify 必须失败：\n{r2.stdout}\n{r2.stderr}"
    assert "contract.tokenizer:" in (r2.stdout + r2.stderr)


# ---------------------------------------------------------------- (e) --cards all
def anchor_scope_args(fx: dict, out: Path, ckpt: Path, scope: str | None) -> list[str]:
    args = ["--anchor"]
    if scope is not None:
        args += ["--cards", scope]
    return args + ["--ckpts", str(ckpt), "--out", str(out),
                   "--tokenizer", str(fx["tokenizer"]), "--pairs", str(fx["pairs"]),
                   "--note", "合成夹具（tests/test_anchor_v0.py）"]


def anchor_scope_manifest(fx: dict, out: Path, ckpt: Path, scope: str | None) -> dict:
    r = run_cli(*anchor_scope_args(fx, out, ckpt, scope), cwd=fx["dir"])
    assert r.returncode == 0, f"--anchor --cards {scope} 失败：\n{r.stdout}\n{r.stderr}"
    assert (out / "manifest.json").is_file()
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def mutate_card(ckpt: Path, prefix: str) -> str:
    """把某张卡（前缀）里第一个浮点张量改一位，原地覆盖夹具 ckpt；返回被改的键名。"""
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    key = next((k for k, v in blob["state"].items()
                if k.startswith(prefix) and torch.is_tensor(v) and v.dtype.is_floating_point), None)
    assert key is not None, f"夹具 ckpt 里没有 {prefix}* 浮点张量"
    blob["state"][key][0, 0] += 1.0
    torch.save(blob, ckpt)
    return key


def test_cards_all_scope(fx: dict, tmp_path: Path) -> None:
    ck = fx["dir"] / f"allmode_copy_s{SEED}_st{STEPS}.pt"
    shutil.copy(fx["ckpt_bridge"], ck)

    # (e-0) 默认口径不变：不带 --cards ⇒ 仍是 input 口径，字段集与旧版一致
    inp = anchor_scope_manifest(fx, tmp_path / "input", ck, scope=None)
    assert "cards_scope" not in inp, "默认 --anchor 不许落 cards_scope（字段集必须与旧版一致）"
    assert "zh_out_sha256" not in inp["cards"][0] and "thinking_card" not in inp["cards"][0]

    # (e-1) --cards all：输出卡两半入 manifest，思维卡单独记
    base = anchor_scope_manifest(fx, tmp_path / "base", ck, scope="all")
    assert base["cards_scope"] == "all"
    c0 = base["cards"][0]
    for k in ("zh_out_sha256", "en_out_sha256", "thinking_card"):
        assert k in c0, f"--cards all 的 manifest 缺字段 {k}"
    assert c0["thinking_card"]["sha256"] and c0["thinking_card"]["n_params"] > 0
    assert "不属于 MV" in c0["thinking_card"]["note"]
    # all 模式下 --verify / --load-check 照旧工作（校验输入卡 + 输出卡两半）
    mp = tmp_path / "base" / "manifest.json"
    rv = run_cli("--verify", str(mp), cwd=fx["dir"])
    assert rv.returncode == 0, f"--verify（all）失败：\n{rv.stdout}\n{rv.stderr}"
    assert "scope=all" in rv.stdout
    rl = run_cli("--load-check", str(mp), "--seed", str(SEED), "--freeze", cwd=fx["dir"])
    assert rl.returncode == 0, f"--load-check（all）失败：\n{rl.stdout}\n{rl.stderr}"
    assert "zh_out/en_out requires_grad=False" in rl.stdout
    assert "thinking_card=sha256=" in rl.stdout

    # (e-2) 只改 bridge.*（思维卡）⇒ mv 不变、thinking_card.sha256 变
    mutate_card(ck, "bridge.")
    br = anchor_scope_manifest(fx, tmp_path / "bridge", ck, scope="all")
    assert br["mv"]["interface_sha256"] == base["mv"]["interface_sha256"], "改思维卡不许动 mv.interface_sha256"
    assert br["mv"]["sha256"] == base["mv"]["sha256"], "改思维卡不许动 mv.sha256"
    assert br["cards"][0]["weights_sha256"] == c0["weights_sha256"], "思维卡不进接口权重 sha"
    assert br["cards"][0]["thinking_card"]["sha256"] != c0["thinking_card"]["sha256"], \
        "改了 bridge.* ⇒ thinking_card.sha256 必须变"
    assert br["cards"][0]["zh_out_sha256"] == c0["zh_out_sha256"]

    # (e-3) 只改 zh_out.*（输出卡）⇒ mv 必变
    mutate_card(ck, "zh_out.")
    zo = anchor_scope_manifest(fx, tmp_path / "zhout", ck, scope="all")
    assert zo["mv"]["interface_sha256"] != base["mv"]["interface_sha256"], "改输出卡必须改 mv.interface_sha256"
    assert zo["mv"]["sha256"] != base["mv"]["sha256"], "改输出卡必须改 mv.sha256"
    assert zo["cards"][0]["zh_out_sha256"] != c0["zh_out_sha256"]
    assert zo["cards"][0]["thinking_card"] == br["cards"][0]["thinking_card"], \
        "输出卡与思维卡互不影响（bridge 自 (e-2) 起未再改）"

    # (e-4) fail-closed：ckpt 缺 zh_out.* 前缀时 --cards all 必须非零退出并指名
    no_out = fx["dir"] / f"no_zhout_s{SEED}_st{STEPS}.pt"
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    blob["state"] = {k: v for k, v in blob["state"].items() if not k.startswith("zh_out.")}
    torch.save(blob, no_out)
    r = run_cli(*anchor_scope_args(fx, tmp_path / "no_out", no_out, "all"), cwd=fx["dir"])
    assert r.returncode != 0, f"缺 zh_out.* 时 --cards all 必须非零退出：\n{r.stdout}\n{r.stderr}"
    assert "zh_out." in (r.stdout + r.stderr), "错误信息必须指名缺的是哪个前缀"
