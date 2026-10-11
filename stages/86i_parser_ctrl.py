#!/usr/bin/env python3
"""S86i · L3-m3b 控制臂判据表（`experiments/meme_anchor/PREREG_PARSER_CTRL.md` §2）。

  臂 I  shared parser 可训（被检对象）      臂 II 无 bridge（基线）
  臂 IV shared parser **冻结随机**（同尺寸）  臂 V  **两条不共享** adapter（同单卡尺寸，都可训）
判据：C1 臂IV ≤ 臂II+2SE（方法自检）；C2 臂V ≤ 臂I−2SE ⇒ **共享是机制**；C3 臂V 相对臂II 有无增益。
    CUDA_VISIBLE_DEVICES="" .venv/bin/python stages/86i_parser_ctrl.py
"""
from __future__ import annotations
import glob, json, math
from pathlib import Path
LOGS = Path("/home/vesita/coding/my/flowme/logs")

def bucket(r: dict) -> str:
    if not r.get("frozen"): return "其它"
    br, fb = r.get("bridge"), bool(r.get("freeze_bridge"))
    if br == "dual": return "臂V 不共享adapter"
    if br == "parser": return "臂IV 冻结随机parser" if fb else "臂I shared parser(可训)"
    if br == "none": return "臂II 无bridge"
    return "其它"

def g(r, *path):
    x = r
    for k in path: x = x[k]
    return float(x)

def mu(rs, *path):
    v = [g(r, *path) for r in rs]; n = len(v)
    return (sum(v)/n, (math.sqrt(sum((x-sum(v)/n)**2 for x in v)/(n-1))/math.sqrt(n)) if n > 1 else float("nan"), v)

def main() -> None:
    G: dict[str, list] = {}
    for f in sorted(glob.glob(str(LOGS / "86g_result_*.json"))):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        if "_a0.5_" not in str(r.get("init_inputs", "")): continue   # 只看冻 α=0.5 输入的那批
        G.setdefault(bucket(r), []).append(r)
    for k in sorted(G): print(f"[ARM] {k}: {len(G[k])} run")
    need = ("臂I shared parser(可训)", "臂II 无bridge", "臂IV 冻结随机parser", "臂V 不共享adapter")
    if any(k not in G for k in need):
        print("[CTRL] 读数还不完整"); return
    I, II, IV, V = (G[k] for k in need)
    print("\n| 臂 | seed | 输入侧Δ | **衔接Δ** | 衔接Δ长度匹配 | 中→英 | 英→中 | 生成EM 英→英 |")
    print("|---|---|---|---|---|---|---|---|")
    for k in need:
        for r in G[k]:
            print(f"| {k} | {r['seed']} | {g(r,'metrics','input','pooled','delta'):+.4f} | "
                  f"**{g(r,'metrics','bridge','pooled','delta'):+.4f}** | "
                  f"{g(r,'metrics','bridge','m8_length_matched','delta'):+.4f} | "
                  f"{r['final']['char_ze']*100:.1f}% | {r['final']['char_ez']*100:.1f}% | "
                  f"{r['gen']['gen_em_ee']*100:.1f}% |")
    bii, _, _ = mu(II, "metrics", "bridge", "pooled", "delta")
    dii, sii, _ = mu(II, "metrics", "bridge", "pooled", "delta")
    biv, _, iv = mu(IV, "metrics", "bridge", "pooled", "delta")
    bi, si, ivv = mu(I, "metrics", "bridge", "pooled", "delta")
    bv, sv, vv = mu(V, "metrics", "bridge", "pooled", "delta")
    c1 = biv <= dii + 2 * sii
    c2 = bv <= bi - 2 * si
    c3 = bv >= dii + 2 * sv
    print(f"\n- C1 方法自检：臂IV 衔接Δ {biv:+.4f} ≤ 臂II {dii:+.4f}+2SE({sii:.4f}) ⇒ {'✅' if c1 else '❌（实现或测量有问题，本试验作废）'}")
    print(f"- C2 机制 vs 容量（核心）：臂V {bv:+.4f} vs 臂I {bi:+.4f}−2SE({si:.4f})={bi-2*si:+.4f} ⇒ "
          f"{'✅ 共享是机制' if c2 else '❌ 增益来自额外容量，不是共享'}")
    print(f"- C3 臂V 相对臂II 有无增益：{bv:+.4f} vs {dii:+.4f}+2SE ⇒ {'有' if c3 else '无'}")
    print("\n[CTRL-VERDICT] " + ("✅ C1∧C2 ⇒ 「parser 卡的**共享瓶颈**是有效机制」" if (c1 and c2)
          else ("❌ C1 不过 ⇒ 本试验作废" if not c1 else "⚠ C1 过、C2 不过 ⇒ 增益归功于**额外容量**，「parser 机制」在本试验里不成立")))

if __name__ == "__main__":
    main()
