#!/usr/bin/env python3
"""S86h · L3-m3 **parser 卡试验的判据表**（口径见 `experiments/meme_anchor/PREREG_PARSER.md` §3）。

- 臂 I  主    ：`bridge=parser` + 冻结 + 载入四卡，init = α=0.5 的 v0（2 seed）
- 臂 II 对照  ：`bridge=none` + **同样的冻结与载入**（2 seed）  ← **判据的对照就是它**（只差 parser 一处）
- 臂 III 强测 ：`bridge=parser` + 冻结，init = **α=0（纯重构）** 的 v0（1 seed）⇒ 输入侧 Δ≈0 时能否从零造出对齐

    CUDA_VISIBLE_DEVICES="" .venv/bin/python stages/86h_parser_verdict.py
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

LOGS = Path("/home/vesita/coding/my/flowme/logs")
MARGIN, SE_K = 0.05, 2.0


def load() -> list[dict]:
    return [json.loads(Path(f).read_text(encoding="utf-8"))
            for f in sorted(glob.glob(str(LOGS / "86g_result_*.json")))]


def arm(r: dict) -> str:
    """把 run 分到 臂I / 臂II / 臂III / 其它（靠 init_inputs 里的 a0.5 / 无 α 标记 区分）。"""
    if not r.get("frozen"):
        return "其它"
    src = str(r.get("init_inputs", ""))
    a05 = "_a0.5_" in src or "_a0.25_" in src
    if r.get("bridge") == "parser":
        return "臂I parser@a0.5" if a05 else "臂III parser@a0"
    return "臂II 无parser@a0.5" if a05 else "其它"


def dig(r: dict, path) -> float:
    x = r
    for k in path:
        x = x[k]
    return float(x)


def main() -> None:
    groups: dict[str, list[dict]] = {}
    for r in load():
        groups.setdefault(arm(r), []).append(r)
    rows = {k: v for k, v in groups.items() if k != "其它"}
    if "臂I parser@a0.5" not in rows or "臂II 无parser@a0.5" not in rows:
        print(f"[PARSER] 读数还不完整：{ {k: len(v) for k, v in rows.items()} }"); return

    def mu(rs, path):
        v = [dig(r, path) for r in rs]
        n = len(v)
        return (sum(v) / n, (math.sqrt(sum((x - sum(v) / n) ** 2 for x in v) / (n - 1)) / math.sqrt(n)) if n > 1 else float("nan"), v)

    P_BR = ("metrics", "bridge")
    P_IN = ("metrics", "input")
    print("| run | bridge | init | seed | 输入侧Δ | **衔接Δ** | 衔接Δ长度匹配 | 中→中 | 英→英 | 中→英 | 英→中 | 生成EM 中→中/英→英 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ("臂II 无parser@a0.5", "臂I parser@a0.5", "臂III parser@a0"):
        for r in rows.get(k, []):
            f, m = r["final"], r["metrics"]
            src = "a0.5" if "_a0.5_" in str(r.get("init_inputs", "")) else "a0"
            print(f"| {k} | {r['bridge']} | {src} | {r['seed']} | {dig(r, P_IN + ('pooled', 'delta')):+.4f} | "
                  f"**{dig(r, P_BR + ('pooled', 'delta')):+.4f}** | {dig(r, P_BR + ('m8_length_matched', 'delta')):+.4f} | "
                  f"{f['char_zz']*100:.1f}% | {f['char_ee']*100:.1f}% | {f['char_ze']*100:.1f}% | {f['char_ez']*100:.1f}% | "
                  f"{r['gen']['gen_em_zz']*100:.1f}%/{r['gen']['gen_em_ee']*100:.1f}% |")

    A, B = rows["臂I parser@a0.5"], rows["臂II 无parser@a0.5"]
    if len(B) < 2:
        print("[PARSER] 臂 II 还不足 2 seed"); return
    checks = [
        ("P1a 衔接 Δ池化 ≥ 臂II+2SE 且 2 seed 同号", (*P_BR, "pooled", "delta"), "q1"),
        ("P1b 衔接 Δ长度匹配 ≥ 臂II+2SE 且 2 seed 同号", (*P_BR, "m8_length_matched", "delta"), "q1"),
        ("P2a 中→中字准 ≥ 臂II−5pp", ("final", "char_zz"), "m"),
        ("P2b 英→英字准 ≥ 臂II−5pp", ("final", "char_ee"), "m"),
        ("P3a 中→英字准 ≥ 臂II−5pp", ("final", "char_ze"), "m"),
        ("P3b 英→中字准 ≥ 臂II−5pp", ("final", "char_ez"), "m"),
        ("P4a 生成EM 中→中 ≥ 臂II−5pp", ("gen", "gen_em_zz"), "m"),
        ("P4b 生成EM 英→英 ≥ 臂II−5pp", ("gen", "gen_em_ee"), "m"),
    ]
    print("\n| 判据 | 臂 II（对照） | 臂 I（parser） | 过？ | 臂I 逐 seed |")
    print("|---|---|---|---|---|")
    allok = True
    for name, path, kind in checks:
        bm, _, _ = mu(B, path)
        cm, cs, cv = mu(A, path)
        ok = (cm >= bm + SE_K * cs and all(v > bm for v in cv)) if kind == "q1" else (cm >= bm - MARGIN)
        allok &= ok
        print(f"| {name} | {bm:+.4f} | {cm:+.4f} | {'✅' if ok else '❌'} | {[round(v, 4) for v in cv]} |")
    print(f"\n[PARSER-VERDICT] {'✅ P1∧P2∧P3∧P4 全过 ⇒ 共享 parser 卡能让两语言在衔接处涌现同一内容码，且不牺牲重构/翻译' if allok else '❌ 未全过 ⇒ 按上面卡住的那条报读数'}")
    if "臂III parser@a0" in rows:
        r3 = rows["臂III parser@a0"][0]
        print(f"[臂III 强测] 输入侧Δ={dig(r3, P_IN + ('pooled','delta')):+.4f}（应≈0）→ 衔接Δ={dig(r3, P_BR + ('pooled','delta')):+.4f}"
              f"；中→英 {r3['final']['char_ze']*100:.1f}% 英→中 {r3['final']['char_ez']*100:.1f}%"
              f" ⇒ {'parser 从零造出了对齐' if dig(r3, P_BR + ('pooled','delta')) > 0.01 else '未能从零造出对齐'}")


if __name__ == "__main__":
    main()
