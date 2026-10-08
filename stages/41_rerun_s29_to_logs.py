#!/usr/bin/env python3
"""S41 · 把 S29 的产物从 /tmp 落回仓库 logs/ —— 零训练 · 全程 CPU。

背景（R32）：S29（stages/29_cross_task_meme.py）的判决"模因 = 公共线性子空间 + 每任务偏置"
只在 /tmp 留过产物，/tmp 已丢 ⇒ 从仓库无法复现 ⇒ F7 拒绝定强度。
S29 已入库、不改它：它支持环境变量 S29_OUT 重定向 JSON 输出
（stages/29_cross_task_meme.py:327 `OUT = os.environ.get("S29_OUT", "/tmp/...")`），
所以本脚本只做「薄封装」——用 S29_OUT 把 JSON 指到 logs/41_s29_rerun.json，
并把 S29 的 stdout/stderr 全量重定向到 logs/41_s29_rerun.log。

只允许写：本文件、logs/41_*。不碰 09…40、speculation/methodology/src/tests。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = "/home/vesita/coding/my/flowme"
SRC29 = f"{ROOT}/stages/29_cross_task_meme.py"
LOG = f"{ROOT}/logs/41_s29_rerun.log"
JSON = f"{ROOT}/logs/41_s29_rerun.json"

# ---- 门禁：确认 S29 确实把产物写 /tmp、且认 S29_OUT 这个环境变量（否则不许乱改它）----
_src = open(SRC29, encoding="utf-8").read()
assert "S29_OUT" in _src and "/tmp/29_cross_task_meme_summary.json" in _src, \
    "S29 不再支持 S29_OUT 重定向 —— 停手，不要改已入库脚本"

os.makedirs(f"{ROOT}/logs", exist_ok=True)
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = ""          # ★强制 CPU：先于 torch 导入
env["S29_OUT"] = JSON
env.setdefault("S26_THREADS", "8")
env.setdefault("S26_K", "128")
env.setdefault("S26_NPERM", "50")
env.setdefault("S26_BATCH", "64")

t0 = time.time()
with open(LOG, "w", encoding="utf-8") as lf:
    lf.write(f"[S41] 薄封装重跑 S29 | src={SRC29} | S29_OUT={JSON} | "
             f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']!r} | python={sys.executable}\n")
    lf.write(f"[S41] cmd: {sys.executable} {SRC29}\n")
    lf.flush()
    r = subprocess.run([sys.executable, SRC29], env=env, cwd=ROOT,
                       stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    lf.write(f"\n[S41] 返回码={r.returncode} | 墙钟={wall:.2f}s | "
             f"JSON 存在={os.path.exists(JSON)}\n")
    if os.path.exists(JSON):
        s = json.load(open(JSON, encoding="utf-8"))
        lf.write(f"[S41] JSON device={s.get('device')} cuda={s.get('cuda')} "
                 f"K={s.get('K')} wall_s(S29)={s.get('wall_s'):.2f}\n")

print(f"[S41] rc={r.returncode} wall={wall:.2f}s log={LOG} json={JSON}")
sys.exit(r.returncode)
