"""模因槽数 `n` 可配置的验收测试（pytest，CPU-only）—— PREREG_SLOTN §2 的 **N9 / N10**。

覆盖：
  (a) **逐位不变**：`MZ.build(1234)`（默认）与 `MZ.build(1234, n_slots=16)` 同 seed 构造，
      `state_dict` 的 max|Δ| = 0、键名顺序一致、构造后全局 RNG 状态一致，`zh_in.slots.q.shape == (16,128)`；
  (b) **n=4 可用**：前向 / `meme_zh` 形状 `(b,4,128)` / logits 形状 / `_junction`+`full_metrics` 跑通
      （取 `pack["test_ids"][:64]`、固定 `torch.manual_seed`），`metrics["input"]["pooled"]["delta"]` 是有限数；
      另断言 `_junction` 的槽数契约在 4 槽张量上被 16 的默认挡住（fail-closed）；
  (c) **fail-closed**：16 槽 ckpt 配 `--slots 4 --init-inputs` ⇒ CLI **非零退出**、指名"形状不符"、
      且**没有开始训练**（stdout 里没有 `[M2] run`）；
  (d) **产物名**：n=16 的 tag 与既有产物**逐字符相同**、只有 n≠16 才追加 `_n{n}`（不训练就能测）。

依赖（本机已具备）：语料缓存 `logs/86_cache_zhen/pairs.jsonl` + 分词器 `logs/86_tokenizer/char_tokenizer.json`
（重建见 `DEVLOG.md` 2026-10-12 条）；全程 CPU、不碰 GPU、不训练。

    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_slots_config.py -q
"""
from __future__ import annotations

import importlib.util
import inspect
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MEME = ROOT / "experiments" / "meme_anchor"
SCRIPT = ROOT / "stages" / "86g_zh_en.py"
SEED = 1234


def _cpu_env() -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""            # 本机无 GPU 也必须过（也不许碰 GPU）
    return env


def _load_driver():
    spec = importlib.util.spec_from_file_location("slots86g", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EG = _load_driver()                              # 复用 86g 的 get_pack / eval_model / _junction / build_tag
MZ, MS, DRV = EG.MZ, EG.MS, EG.DRV


@pytest.fixture(scope="session")
def pack() -> dict:
    """真实语料（与 86g 同一份切分）—— (b) 的 `_junction`/`full_metrics` 只取前 64 条控成本。"""
    return EG.get_pack()


# ---------------------------------------------------------------- (a) n=16 逐位不变
def test_default_slots_bit_identical() -> None:
    a, b = MZ.build(SEED), MZ.build(SEED, n_slots=16)
    sda, sdb = a.state_dict(), b.state_dict()
    assert list(sda) == list(sdb), "键名/顺序必须与默认构造一致"
    mx = max(float((x.float() - y.float()).abs().max()) for x, y in zip(sda.values(), sdb.values()))
    assert mx == 0.0, f"默认构造与显式 n_slots=16 必须逐位相同：max|Δ|={mx}"
    assert tuple(a.zh_in.slots.q.shape) == (16, 128)
    assert tuple(b.en_in.slots.q.shape) == (16, 128)
    assert a.n_slots == b.n_slots == 16

    torch.manual_seed(0)
    MZ.build(SEED)
    r1 = torch.get_rng_state().clone()
    torch.manual_seed(0)
    MZ.build(SEED, n_slots=16)
    r2 = torch.get_rng_state().clone()
    assert torch.equal(r1, r2), "构造后全局 RNG 状态必须一致（R34：槽数变了会多/少抽随机数）"


# ---------------------------------------------------------------- (b) n=4 前向 + 度量
def test_n4_forward_and_metrics(pack: dict) -> None:
    sub_ids = pack["test_ids"][:64]
    sub = {"test": pack["test"][:64], "test_ids": sub_ids}
    torch.manual_seed(20261012)
    m = MZ.build(7777, bridge="parser", n_slots=4)          # 顺带覆盖思维卡路（L3-t2 的 C 臂）
    assert m.n_slots == 4 and tuple(m.zh_in.slots.q.shape) == (4, 128)

    zl = [p[0] for p in sub_ids]
    ids = EG.pad_ids(zl, "cpu")
    mz = m.meme_zh(ids)
    assert tuple(mz.shape) == (len(zl), 4, 128), f"meme_zh 形状 {tuple(mz.shape)}"

    tgt_in, tgt_out = EG.make_pairs(zl, "cpu")
    lg = m.logits("zh", mz, tgt_in)
    assert tuple(lg.shape) == tuple(tgt_in.shape) + (EG.VOCAB,)
    assert lg.shape[0] == tgt_out.shape[0]

    # 逐槽口径在 n=4 下泛化（shuffle / slot_matrix 不写死 16）
    assert tuple(MS.shuffle_slots(mz, 5).shape) == tuple(mz.shape)
    assert tuple(DRV.slot_matrix(mz, mz).shape) == (len(zl), len(zl))
    assert tuple(MS.floor_random(3, 7, m.n_slots, 128).shape) == (3, 4, 128)

    # fail-closed：4 槽张量不能拿 n_slots=16 的默认去算（否则地板/逐槽口径错形）
    with pytest.raises(AssertionError, match="槽数与 n_slots 不符"):
        EG._junction(mz, mz, sub, 1, n_slots=16)

    m.eval()
    torch.manual_seed(31)
    ev = EG.eval_model(m, sub, "cpu", chunk=16)
    assert tuple(ev["m_zh"].shape) == (64, 4, 128)
    assert tuple(ev["b_zh"].shape) == (64, 4, 128)
    metrics = EG.full_metrics(ev, sub, 20261012, n_slots=m.n_slots)
    assert set(metrics) == {"input", "bridge"}
    d = metrics["input"]["pooled"]["delta"]
    assert math.isfinite(d), f"input pooled.delta 不是有限数：{d}"
    assert math.isfinite(metrics["bridge"]["pooled"]["delta"])
    assert metrics["input"]["slot"]["n"] == 64


# ---------------------------------------------------------------- (c) fail-closed（CLI 端到端）
def _make_ckpt(path: Path, n_slots: int = 16) -> Path:
    """按 86g 的 `dict(state=..., seed=..., ...)` 格式落一个**未训练**的 ckpt（不训练、只构造）。"""
    m = MZ.build(SEED, n_slots=n_slots)
    torch.save({"state": m.state_dict(), "seed": SEED, "lam": 0.0, "maps": ("AaBb", "AbBa"),
                "loss_kind": "none", "steps": 1, "shared_out": False}, path)
    return path


def test_fail_closed_16slot_ckpt_under_slots4(tmp_path: Path) -> None:
    ck = _make_ckpt(tmp_path / "ck_16slots.pt")
    blob = torch.load(ck, map_location="cpu", weights_only=False)

    # 正例：同槽数放行（不抛）
    EG._check_init_slots(blob, 16)

    # 反例①：16 槽 ckpt 配 4 槽 ⇒ 必须指名"形状不符"
    with pytest.raises(SystemExit) as e1:
        EG._check_init_slots(blob, 4)
    assert "形状不符" in str(e1.value) and "--slots 4" in str(e1.value)

    # 反例②：有输入卡却缺 `slots.q`（strict=False 只记 missing ⇒ 会退化成随机槽）也必须 fail-closed
    no_head = {"state": {k: v for k, v in blob["state"].items() if k != "zh_in.slots.q"}}
    with pytest.raises(SystemExit) as e2:
        EG._check_init_slots(no_head, 16)
    assert "zh_in.slots.q" in str(e2.value)

    # 端到端：真实 CLI，非零退出 + 指名形状 + **没开始训练**
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--run", "0.0", str(SEED), "--steps", "1",
         "--slots", "4", "--init-inputs", str(ck), "--device", "cpu"],
        cwd=str(tmp_path), env=_cpu_env(), capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode != 0, f"16 槽 ckpt 配 --slots 4 必须非零退出：\n{out}"
    assert "形状不符" in out, f"错误信息必须指名形状不符：\n{out}"
    assert "--slots 4" in out, f"错误信息必须报出当前槽数：\n{out}"
    assert "[M2] run" not in out, "必须在开训之前就退出（不许先跑起来）"


# ---------------------------------------------------------------- (d) 产物命名 / CLI
def test_artifact_tag_and_cli() -> None:
    base = {"loss_kind": "none", "mapset": ("AaBb", "AbBa"), "shared_out": False,
            "abba_weight": 0.5, "bridge": "none", "bridge_mid": 64, "bridge_ff": 256,
            "align_on": "bridge", "obj": "tf", "roll_w": 0.5, "steps": 8000, "accum": 1,
            "freeze_inputs": False, "init_outputs": False, "freeze_bridge": False,
            "init_lab": ""}
    # 金标 1：logs/86g_ckpt_none_AaBb_AbBa_a0.5_o-tf_st8000_l0.0_s1234.pt 的 tag（既有产物）
    assert EG.build_tag(**base, n_slots=16) == "none_AaBb_AbBa_a0.5_o-tf_st8000"
    # 金标 2：PREREG_SLOTN §1 记录的 n=4 产物名
    assert EG.build_tag(**base, n_slots=4) == "none_AaBb_AbBa_a0.5_n4_o-tf_st8000"
    assert "_n16" not in EG.build_tag(**base, n_slots=16), "n=16 不许改产物名"

    # 默认值三处都是 16，且 train_run 的槽位是**关键字**（加在末尾，不与 accum 等错位）
    assert inspect.signature(MZ.build).parameters["n_slots"].default == 16
    sig = inspect.signature(EG.train_run)
    assert sig.parameters["n_slots"].default == 16
    assert list(sig.parameters).index("n_slots") == len(sig.parameters) - 1

    r = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                       env=_cpu_env(), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "--slots" in r.stdout, "CLI 必须有 --slots"
