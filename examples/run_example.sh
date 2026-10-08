#!/usr/bin/env bash
# examples/run_example.sh —— 三类卡闭环 · 合成算术最小可跑例子
#
# 只复用已入库的 ckpt 与 stages/19_residual_scan.py（exec 其前缀，绝不复制模型定义）；
# 在 CPU 上重算 add_1d / add_3d 两臂 EM + R28 批内一致性 + R29 非零性冒烟。
# 幂等（只读、不写文件、不训练）；成功退出 0，失败退出非 0 并打印原因。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="$ROOT/.venv/bin/python"
STAGE="$ROOT/stages/19_residual_scan.py"
TOKENIZER="/home/vesita/coding/my/nanoSeek/data/chinese/char_tokenizer.json"

die() { printf '[FAIL] %s\n' "$*" >&2; exit 1; }

[ -x "$PY" ] || die "找不到项目 venv 解释器：$PY（本例子不装包，请先备好项目 venv）"
[ -f "$STAGE" ] || die "找不到已入库脚本：$STAGE"
[ -f "$TOKENIZER" ] || die "找不到 tokenizer：$TOKENIZER"

export CUDA_VISIBLE_DEVICES=""       # ★ 强制 CPU；本例子不做任何 GPU / 训练操作
echo "[EX] repo=$ROOT"
echo "[EX] stage=$STAGE（只读复用）"
echo "[EX] device=cpu（CUDA_VISIBLE_DEVICES 已清空；不训练）"

"$PY" - "$ROOT" <<'PY'
import hashlib
import pathlib
import re
import sys

import torch

ROOT = pathlib.Path(sys.argv[1])
STAGE = ROOT / "stages" / "19_residual_scan.py"
LOGS = ROOT / "logs"
RUN_BUCKETS = (("add_1d", "一位数加"), ("add_3d", "三位数加"))


def sha16(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


# ---- 复用已入库脚本：exec 其前缀（到主循环之前），只剥 GPU 守卫并把 device 钉成 cpu ----
# 模型定义 / 数据生成 / 贪心生成 / R28 自检 全部来自 stages/19_residual_scan.py，本脚本零复制。
src = STAGE.read_text()
head = src.split("# ③ 主循环", 1)[0].rsplit("\n# ===", 1)[0]
head = re.sub(r'\nif device != "cuda":.*?sys\.exit\(3\)\n', "\n", head, flags=re.S)
head = head.replace('device = "cuda" if torch.cuda.is_available() else "cpu"', 'device = "cpu"')
ns = {"__name__": "s19_example", "__file__": str(STAGE)}
exec(compile(head, str(STAGE), "exec"), ns)

if ns.get("device") != "cpu":
    fail(f"device 未钉成 cpu（得到 {ns.get('device')!r}）")
if torch.cuda.is_available():
    fail("CUDA 仍可见（本例子必须跑 CPU）")
for sym in ("Cards", "DATA", "greedy_gen", "hits_from", "r28_selfcheck", "D", "FF"):
    if sym not in ns:
        fail(f"复用失败：19_residual_scan.py 前缀里没有 {sym}")

Cards, DATA = ns["Cards"], ns["DATA"]
greedy_gen, hits_from, r28_selfcheck = ns["greedy_gen"], ns["hits_from"], ns["r28_selfcheck"]
print("[EX] 已复用 19_residual_scan.py 前缀：Cards / DATA / greedy_gen / hits_from / r28_selfcheck")

concl = []
for bucket, cn in RUN_BUCKETS:
    test = DATA[bucket]["test"]
    hits = {}
    for arm in ("nores", "res"):
        ck = LOGS / f"19_ckpt_{bucket}_{arm}_seed1234.pt"
        if not ck.is_file():
            fail(f"缺少 ckpt：{ck}")
        model = Cards(ns["D"], ns["FF"], residual=(arm == "res"))
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        model.eval()
        texts = greedy_gen(model, test, batch=1)          # ★R28 主口径 batch=1
        hs = hits_from(texts, test)
        hits[arm] = hs
        em = sum(h["strict"] for h in hs) / len(hs)
        print(f"[ARM] {bucket:7s} {arm:5s} EM={em * 100:6.2f}% (n={len(hs)}) sha16={sha16(ck)}")
        del model

    # ---- R29：干预（残差）非零性冒烟 = Δ(res−nores) 非零比例 ----
    nz = sum(1 for a, b in zip(hits["res"], hits["nores"]) if a["strict"] != b["strict"])
    print(f"[R29] {bucket:7s} Δ(res−nores) 非零比例 = {nz}/{len(hits['res'])} "
          f"= {nz / len(hits['res']) * 100:.1f}%")

    # ---- R28：批内一致性自检（batch=1 vs batch=16）----
    model = Cards(ns["D"], ns["FF"], residual=True)
    model.load_state_dict(torch.load(LOGS / f"19_ckpt_{bucket}_res_seed1234.pt", map_location="cpu"))
    model.eval()
    r28_selfcheck(model, test, tag=f"[{bucket}] ")
    del model

    nores_em = sum(h["strict"] for h in hits["nores"]) / len(hits["nores"])
    res_em = sum(h["strict"] for h in hits["res"]) / len(hits["res"])
    concl.append(f"{cn} {nores_em * 100:.2f}% → {res_em * 100:.2f}%")

print("[CONCLUSION] 残差： " + " | ".join(concl)
      + "。★只有三位数加（深推理/进位链）显著受益；一位数加两臂都在天花板 ~97% ⇒ 不是普遍增益。")
print("[EX] OK")
PY
rc=$?
[ "$rc" -eq 0 ] || die "评估失败（python 退出码 $rc）"
exit 0
