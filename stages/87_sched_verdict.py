#!/usr/bin/env python3
"""S87 判词：读两臂逐 token CE，出四档判词。规格 = `experiments/meme_anchor/PREREG_SCHED.md` §3/§4。

判据（写死，不许在这里改口径）：
  逐 token 配对 ΔCE，SE 按**样本聚类稳健**（同句内 token 相关，不许当独立样本）。
    ΔCE < −2SE 且 2 seed 同号            ⇒ 一 成立（该换调度）
    |ΔCE| ≤ 1SE（两 seed 同时）          ⇒ 三 未测出差异（本轴关闭）
    1SE < |ΔCE| < 2SE 或 2 seed 异号     ⇒ 三′ 不可判（只许加 seed/步数）
    ΔCE > +2SE 且 2 seed 同负号          ⇒ 四 有害
    读数缺失                              ⇒ 五 作废

    python stages/87_sched_verdict.py --verify      # ★ R48：用假结果逐档驱动（跑前必过）
    python stages/87_sched_verdict.py --steps 3000  # 真判
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/vesita/coding/my/flowme")
LOGS = ROOT / "logs"
SEEDS = (1234, 5678)
SCHEDULES = ("const", "cosine")


def chunk_lens(steps: int, bs: int = 64) -> list[int]:
    """★ 逐样本在批内被 padding 到多长。

    `eval_paired` 走 `MD.make_pairs`：`seq = [BOS] + ids + [EOS]`（`model.py:208`）
    ⇒ 批内补齐长度 = `max(len(s) + 2)`，**不是** `max(len(s))`。
    ★ 独立复核抓到此处的 +2 漏算（严重度：致命）：长度断言当场拦下，否则样本边界会每句错位 2 位、
    `pad_mask` 全错而**静默**算废 SE。
    """
    meta = json.loads((ROOT / "logs" / "87_cache" / "sent_meta.json").read_text(encoding="utf-8"))
    L = [len(s) + 2 for s in meta["test"]]        # +BOS +EOS
    out: list[int] = []
    for i in range(0, len(L), bs):
        m = max(L[i:i + bs])
        out += [m] * len(L[i:i + bs])
    return out


def cluster_delta(d: torch.Tensor, lens: list[int], mask: torch.Tensor) -> tuple[float, float, int]:
    """按样本聚类的配对 ΔCE 及其 SE。

    d / mask 均为**含 PAD 位的全长**展平数组；先在**样本内**对真 token 取均值（token 相关），
    再在样本之间算 SE（独立单元 = 句子）。
    """
    if d.numel() != sum(lens) or mask.numel() != d.numel():
        raise ValueError(f"长度不符：d={d.numel()} mask={mask.numel()} Σlen={sum(lens)}")
    per_sample: list[float] = []
    off = 0
    for n in lens:
        sel = mask[off:off + n]
        if int(sel.sum()) == 0:
            off += n
            continue
        per_sample.append(float(d[off:off + n][sel].mean()))
        off += n
    s = torch.tensor(per_sample, dtype=torch.float64)
    n_s = s.numel()
    if n_s < 2:
        raise ValueError(f"有效样本不足：{n_s}")
    return float(s.mean()), float(s.std(unbiased=True) / (n_s ** 0.5)), n_s


def load_arm(sched: str, seed: int, steps: int) -> dict | None:
    p = LOGS / f"87_tok_{sched}_s{seed}_st{steps}.pt"
    if not p.exists():
        return None
    return torch.load(p, weights_only=False)


def em_floor() -> float:
    """EM 的朴素地板：全预测"最常见类"能拿到的比例（这里按 test 句长众数占位，保守取 0）。

    ★ 诚实标注：本线的 EM 是"整句逐位全对"，最常见类不是句子 ⇒ 地板只能给一个**保守下界**；
    因此非劣门阈值实际由 `2×SE` 主导（`max(em_floor, 2SE)`）。这里返回 0 并在报告里写明。
    """
    return 0.0


def per_seed(lens: list[int], steps: int) -> dict:
    out: dict = {}
    for seed in SEEDS:
        a = load_arm("const", seed, steps)
        b = load_arm("cosine", seed, steps)
        if a is None or b is None:
            out[seed] = {"missing": True, "have": {"const": a is not None, "cosine": b is not None}}
            continue
        mask = a["pad_mask"]
        d = (b["ce_tok_t"].double() - a["ce_tok_t"].double())
        mean, se, n_s = cluster_delta(d, lens, mask)
        em_a = sum(a["exact_t"]) / max(len(a["exact_t"]), 1)
        em_b = sum(b["exact_t"]) / max(len(b["exact_t"]), 1)
        out[seed] = {"missing": False, "dce": mean, "se": se, "dmin": 2 * se,
                     "n_samples": n_s, "em_const": em_a, "em_cosine": em_b,
                     "em_floor": em_floor(), "em_drop": em_a - em_b}  # type: ignore[arg-type]
    return out


def verdict(per: dict) -> dict:
    """★ 纯函数：只吃数字，出判词档位。R48 要求它能被假结果驱动。

    ★ 独立复核抓到的缺口（已修）：初版**完全没实现「二 代价过大」** ——
    预登记 §4 写了"主判据过但非劣门破 ⇒ 记目标互斥"，代码里却直接判"一 成立" ⇒
    EM 崩掉也会被报成"该换调度"。非劣门定义（PREREG §3）：EM 掉幅 > `max(地板, 2×SE)` 即破。
    """
    missing = [s for s, v in per.items() if v.get("missing")]
    if missing:
        return {"档位": "五 作废", "理由": f"读数缺失 seed={missing}", "seed 明细": per}

    dmin_max = max(v["dmin"] for v in per.values())
    # 非劣门：EM 掉幅阈值 = max(地板, 2×SE)；地板用 EM 的朴素地板（最常见类别占比的补）
    gates = []
    for s, v in per.items():
        em_drop = v["em_const"] - v["em_cosine"]          # 正数 = 掉了
        thr = max(v.get("em_floor", 0.0), 2 * v["se"])
        gates.append({"seed": s, "em_drop": em_drop, "thr": thr, "pass": em_drop <= thr})

    signs = []
    for s, v in per.items():
        d, se = v["dce"], v["se"]
        if d < -2 * se:
            signs.append(-1)
        elif d > 2 * se:
            signs.append(+1)
        else:
            signs.append(0)
    unified = len(set(signs)) == 1
    all_neg = unified and signs[0] == -1
    all_pos = unified and signs[0] == +1
    # ★ 三 未测出差异 要求**两 seed 都严格在 1SE 内**；恰好 |Δ|==1SE 归"带内不可判"
    within1 = all(abs(v["dce"]) < v["se"] for v in per.values())

    if all_neg and not all(g["pass"] for g in gates):
        return {"档位": "二 代价过大", "理由": "主判据过但 EM 非劣门破 ⇒ 目标互斥（须新预注册再调）",
                "非劣门": gates, "seed 明细": per}
    if within1:
        return {"档位": "三 未测出差异", "理由": f"两 seed 都在 |Δ|<1SE 内；Δ_min={dmin_max:.2e}",
                "非劣门": gates, "seed 明细": per}
    if all_neg:
        return {"档位": "一 成立", "理由": f"ΔCE < −2SE 且 2 seed 同号；Δ_min={dmin_max:.2e}",
                "非劣门": gates, "seed 明细": per}
    if all_pos:
        return {"档位": "四 有害", "理由": "ΔCE > +2SE 且 2 seed 同号 ⇒ 该配置下有害",
                "非劣门": gates, "seed 明细": per}
    return {"档位": "三′ 不可判", "理由": "带内（1SE..2SE）或 2 seed 异号 ⇒ 只许加 seed/步数，不许外推",
            "非劣门": gates, "seed 明细": per}


# ------------------------------------------------------------------ R48 自测
def _fake(seed: int, dce: float, se: float, missing: bool = False, **over) -> dict:
    if missing:
        return {"missing": True}
    d = {"missing": False, "dce": dce, "se": se, "dmin": 2 * se, "n_samples": 2000,
         "em_const": 0.62, "em_cosine": 0.63, "em_floor": 0.0}
    d.update(over)
    # ★ em_drop 由 const/cosine 派生，不许被 over 单独覆盖（假数据必须自洽）
    d["em_drop"] = d["em_const"] - d["em_cosine"]
    return d


CASES = [
    ("成立", {1234: _fake(1234, -0.030, 0.010), 5678: _fake(5678, -0.025, 0.010)}, "一 成立"),
    ("未测出", {1234: _fake(1234, -0.005, 0.010), 5678: _fake(5678, -0.008, 0.010)}, "三 未测出差异"),
    ("有害", {1234: _fake(1234, +0.030, 0.010), 5678: _fake(5678, +0.021, 0.010)}, "四 有害"),
    ("异号", {1234: _fake(1234, -0.030, 0.010), 5678: _fake(5678, +0.030, 0.010)}, "三′ 不可判"),
    # 「带内」= 有一条落在 1SE 与 2SE **之间** ⇒ 不可判（不是"未测出"）
    ("带内", {1234: _fake(1234, -0.015, 0.010), 5678: _fake(5678, -0.014, 0.010)}, "三′ 不可判"),
    ("缺读数", {1234: _fake(1234, -0.030, 0.010), 5678: _fake(5678, 0, 0, missing=True)}, "五 作废"),
    ("零效应", {1234: _fake(1234, 0.0, 0.010), 5678: _fake(5678, 0.0, 0.010)}, "三 未测出差异"),
    ("一个过线一个带内", {1234: _fake(1234, -0.030, 0.010), 5678: _fake(5678, -0.012, 0.010)},
     "三′ 不可判"),
    # ★ 独立复核指出缺的档位与边界（补上）
    ("二 代价过大", {1234: _fake(1234, -0.030, 0.010, em_const=0.65, em_cosine=0.60),
                    5678: _fake(5678, -0.025, 0.010, em_const=0.64, em_cosine=0.60)}, "二 代价过大"),
    ("恰好 1SE", {1234: _fake(1234, -0.010, 0.010), 5678: _fake(5678, -0.010, 0.010)},
     "三′ 不可判"),
    ("恰好 2SE", {1234: _fake(1234, -0.020, 0.010), 5678: _fake(5678, -0.020, 0.010)},
     "三′ 不可判"),
    ("一正一零", {1234: _fake(1234, -0.030, 0.010), 5678: _fake(5678, -0.002, 0.010)},
     "三′ 不可判"),
]


def verify() -> None:
    bad = 0
    for name, per, want in CASES:
        got = verdict(per)["档位"]
        ok = "✅" if got == want else "❌"
        if got != want:
            bad += 1
        print(f"{ok} 假结果「{name}」⇒ {got}（期望 {want}）")
    if bad:
        print(f"[VERIFY-FAIL] {bad}/{len(CASES)} 档位不符 —— 判词接线有错，不许开跑")
        sys.exit(1)
    print(f"[VERIFY-PASS] {len(CASES)} 档全部命中")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    if args.verify:
        verify()
        return
    sizes = chunk_lens(args.steps)
    per = per_seed(sizes, args.steps)
    v = verdict(per)
    print(f"[VERDICT-S87] 档位 = {v['档位']}")
    print(f"[VERDICT-S87] 理由 = {v['理由']}")
    for s, d in v["seed 明细"].items():
        if d.get("missing"):
            print(f"  seed {s}: 读数缺失 {d.get('have')}")
        else:
            print(f"  seed {s}: ΔCE={d['dce']:+.4e} SE={d['se']:.4e} "
                  f"Δ_min={d['dmin']:.4e} EM {d['em_const']:.4f}→{d['em_cosine']:.4f}")
    (LOGS / f"87_verdict_st{args.steps}.json").write_text(
        json.dumps(v, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
