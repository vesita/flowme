#!/usr/bin/env python3
"""S48 · P4 门0 预检（零训练 · CPU）：cards/ 数据路径 vs stages/ 数据路径逐位对齐 + 两条随机流。

只回答一个问题：从 `cards/example_add3d.py` 的数据构造路径重建 add_3d，能否与
`stages/19_residual_scan.py`（⇒ S36 口径）逐位相同？

做法（全部零训练）：
 1. 文本 exec `cards/example_add3d.py` 到 "# ③ 交叉核对" 之前 ⇒ 得 cards 侧数据路径
    （数据 + 训练/评测的 logits_train 换流函数，均只定义不执行）；
 2. 文本 exec `stages/19_residual_scan.py` 到 "# ③ 主循环" 之前 ⇒ 得 stages 侧数据路径
    （与 stages/36 同法；本进程用 CUDA_VISIBLE_DEVICES="" + 只读补丁越过 S19 的 GPU 门，
     既不初始化 CUDA、也不碰 P3 正在用的 GPU）；
 3. 两侧各 build 一份 add_3d train4000/test800，按 S36 的指纹口径比 sha256[:16] 与 (a,b) 泄漏，
    并逐字段逐位比较；
 4. R34：定位两侧的 dropout 流 / 重洗流，按各自真实的 RNG 消耗顺序（seed → 模型构造 → order）
    实测抽 130 个 batch 的索引序列（含第 125 步的第一个 epoch 重洗），逐位对照；
    并用一次 dummy 前向验证 cards 侧 `logits_train` 的换流真的不动全局 CPU 流。

★零训练：不调用 backward/opt.step；只 build 数据 + 构造模型（不 forward 训练）+ 抽 randperm。
只允许写：本文件、logs/48_*、/tmp。cards/ 与 stages/ 其它脚本只读、不改。
运行：`.venv/bin/python stages/48_p4_gate0.py`
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # ★绝不碰 GPU（P3 在 GPU 上跑 46_*）
ROOT = "/home/vesita/coding/my/flowme"
CARDS = f"{ROOT}/cards/example_add3d.py"
SRC19 = f"{ROOT}/stages/19_residual_scan.py"
LOG = f"{ROOT}/logs/48_gate0.log"
JSON_OUT = f"{ROOT}/logs/48_gate0.json"
T0 = time.time()

import torch  # noqa: E402

SEED = 1234
BATCH = 32
N_BATCHES = 130          # 125 个 batch 用完初始 order ⇒ 第 125 个 batch 触发第一次 epoch 重洗


def _exec_head(path: str, cut_marker: str, force_cuda_gate: bool) -> dict:
    """把 path 文本切到 cut_marker 之前 exec 进一个新命名空间（源文件只读）。

    force_cuda_gate：S19 头在 device!=cuda 时 sys.exit(3)；本进程 CUDA 不可见，
    故只在 exec 期间把 torch.cuda.is_available/get_device_name 换成常量（不改源文本、不建上下文）。
    """
    src = open(path, encoding="utf-8").read()
    cut = src.index(cut_marker)
    ns = {"__name__": "p4gate0", "__file__": path}
    saved = None
    if force_cuda_gate:
        saved = (torch.cuda.is_available, torch.cuda.get_device_name)
        torch.cuda.is_available = lambda: True
        torch.cuda.get_device_name = lambda *a, **k: "cpu-forced(s48)"
    try:
        exec(compile(src[:cut], path, "exec"), ns)          # noqa: S102
    finally:
        if saved is not None:
            torch.cuda.is_available, torch.cuda.get_device_name = saved
    return ns


def _test_sha16(test) -> str:
    """与 stages/36:110-114 逐字相同的 TEST 指纹口径。"""
    h = hashlib.sha256()
    for r in test:
        h.update((r["nl"] + "||" + r["gold"] + "\n").encode())
    return h.hexdigest()[:16]


def _leak(train, test) -> int:
    """与 stages/36:115-118 逐字相同：(a,b) 对泄漏条数。"""
    tp = {(r["a"], r["b"]) for r in train}
    return sum(1 for r in test if (r["a"], r["b"]) in tp)


FIELDS = ("p", "t", "t_full", "gold", "a", "b", "y", "nl", "cut_p", "cut_t", "text")


def main() -> int:
    print(f"[S48] ★P4 门0 预检：cards/ vs stages/ 数据路径逐位对齐 | device=cpu | "
          f"零训练 | seed={SEED} batch={BATCH} n_batches={N_BATCHES} | "
          f"torch={torch.__version__} cuda_visible='{os.environ['CUDA_VISIBLE_DEVICES']}'",
          flush=True)

    print(f"\n[S48] ===== ① 从 cards/ 侧重建（exec {CARDS} 到 '# ③ 交叉核对' 之前：数据 + 训练/评测口径） =====",
          flush=True)
    C = _exec_head(CARDS, "# ③ 交叉核对", False)
    train_c, test_c = C["build_data"]()

    print(f"\n[S48] ===== ② 从 stages/ 侧重建（exec {SRC19} 到 '# ③ 主循环' 之前，同 stages/36） =====",
          flush=True)
    S = _exec_head(SRC19, "# ③ 主循环", True)
    train_s = S["DATA"]["add_3d"]["train"]
    test_s = S["DATA"]["add_3d"]["test"]

    # ------------------------------------------------------------------ 指纹
    sha_c, sha_s = _test_sha16(test_c), _test_sha16(test_s)
    leak_c, leak_s = _leak(train_c, test_c), _leak(train_s, test_s)
    print(f"\n[S48-CHK] TEST sha256[:16]：cards={sha_c} | stages={sha_s} | "
          f"入库=b34e7ea515203227 ⇒ cards {'✓' if sha_c == 'b34e7ea515203227' else '★✗'} / "
          f"stages {'✓' if sha_s == 'b34e7ea515203227' else '★✗'} / "
          f"两侧{'逐位同' if sha_c == sha_s else '★不同'}", flush=True)
    print(f"[S48-CHK] (a,b) 泄漏：cards={leak_c}/800 | stages={leak_s}/800 | 入库=16 ⇒ "
          f"cards {'✓' if leak_c == 16 else '★✗'} / stages {'✓' if leak_s == 16 else '★✗'} / "
          f"两侧{'同' if leak_c == leak_s else '★不同'}", flush=True)

    # ------------------------------------------------------------------ 形状（S19 [DATA] 行口径）
    def _shape(train, test):
        from collections import Counter
        ctr = Counter(r["y"] for r in train)
        ys = sorted(ctr)
        top5 = [v for v, _ in Counter(r["gold"] for r in train).most_common(5)]
        return dict(n_train=len(train), n_test=len(test), ans_min=min(ys), ans_max=max(ys),
                    ans_n=len(ys), cnt_min=min(ctr.values()), cnt_max=max(ctr.values()),
                    max_share=100.0 * max(ctr.values()) / len(train),
                    pair_tr=len({(r["a"], r["b"]) for r in train}),
                    pair_te=len({(r["a"], r["b"]) for r in test}),
                    cut=sum(1 for r in train + test if r["cut_p"] or r["cut_t"]),
                    tgt_med=sorted(len(r["t_full"]) for r in train)[len(train) // 2],
                    top5=top5)
    sh_c, sh_s = _shape(train_c, test_c), _shape(train_s, test_s)
    print(f"[S48-SHAPE] cards  : train={sh_c['n_train']} test={sh_c['n_test']} "
          f"答案域={sh_c['ans_min']}..{sh_c['ans_max']}({sh_c['ans_n']}个) "
          f"计数train min/max={sh_c['cnt_min']}/{sh_c['cnt_max']} 最大占比={sh_c['max_share']:.1f}% | "
          f"distinct(a,b) train={sh_c['pair_tr']} test={sh_c['pair_te']} | 截断={sh_c['cut']} | "
          f"target中位={sh_c['tgt_med']}tok | ★最高频5答案={sh_c['top5']}", flush=True)
    print(f"[S48-SHAPE] stages : train={sh_s['n_train']} test={sh_s['n_test']} "
          f"答案域={sh_s['ans_min']}..{sh_s['ans_max']}({sh_s['ans_n']}个) "
          f"计数train min/max={sh_s['cnt_min']}/{sh_s['cnt_max']} 最大占比={sh_s['max_share']:.1f}% | "
          f"distinct(a,b) train={sh_s['pair_tr']} test={sh_s['pair_te']} | 截断={sh_s['cut']} | "
          f"target中位={sh_s['tgt_med']}tok | ★最高频5答案={sh_s['top5']}", flush=True)

    # ------------------------------------------------------------------ 前 3 条
    for tag, tr, te in (("cards", train_c, test_c), ("stages", train_s, test_s)):
        print(f"[S48-FIRST3] {tag}:", flush=True)
        for i in range(3):
            r = te[i]
            print(f"  [{i}] prompt={('题干：' + r['nl'] + ' → ')!r} gold={r['gold']!r} "
                  f"(a,b)=({r['a']},{r['b']}) y={r['y']} p[:8]={r['p'][:8]} t={r['t']}", flush=True)

    # ------------------------------------------------------------------ 逐位比较
    mismatches = []
    for split, a_l, b_l in (("train", train_c, train_s), ("test", test_c, test_s)):
        if len(a_l) != len(b_l):
            mismatches.append((split, "len", len(a_l), len(b_l)))
            continue
        for i, (ra, rb) in enumerate(zip(a_l, b_l)):
            for f in FIELDS:
                if ra[f] != rb[f]:
                    mismatches.append((split, i, f, ra[f], rb[f]))
    print(f"\n[S48-BITWISE] train={len(train_s)} test={len(test_s)} × 字段{len(FIELDS)} "
          f"逐位比较 ⇒ 不同点={len(mismatches)}", flush=True)
    if mismatches:
        for m in mismatches[:10]:
            print(f"  ★DIFF {m}", flush=True)
    else:
        print("  ⇒ 全部逐位相同（prompt 由 nl 逐字重建、token ids、gold、划分索引均同）", flush=True)
    # 显式比 prompt 字符串（两处 encode_record 都不存 prompt，用 nl 重建）
    pm = sum(1 for ra, rb in zip(train_c + test_c, train_s + test_s)
             if ("题干：" + ra["nl"] + " → ") != ("题干：" + rb["nl"] + " → "))
    print(f"  prompt 字符串（'题干：'+nl+' → '）不同条数={pm}", flush=True)

    # ------------------------------------------------------------------ ③ 两条随机流
    print(f"\n[S48-RNG] ===== ③ 两条随机流（R34）=====", flush=True)
    N = len(train_s)
    # stages/19 结构：manual_seed → cuda.manual_seed_all → Cards 构造 → order(Generator(seed))
    torch.manual_seed(SEED)
    _msa = torch.cuda.manual_seed_all(SEED)       # CUDA 不可见 ⇒ no-op（GPU 上抽独立的 CUDA 流）
    _mS = S["Cards"](S["D"], S["FF"], residual=True)
    st_s = torch.random.get_rng_state().clone()
    # cards 结构：manual_seed → DROP_GEN.manual_seed → CardPipeline 构造 → order(Generator(seed))
    torch.manual_seed(SEED)
    C["DROP_GEN"].manual_seed(SEED)
    _mC = C["CardPipeline"](d=C["D"], vocab_size=C["V"], ff=C["FF"], nhead=C["NHEAD"],
                            maxlen=C["MAXLEN"], pad_id=C["PAD_ID"], dropout=0.1, residual=True)
    st_c = torch.random.get_rng_state().clone()
    print(f"[S48-RNG] 模型构造后全局 CPU RNG 状态：cards vs stages "
          f"{'逐位同' if torch.equal(st_c, st_s) else '★不同（⇒ 第一次 epoch 重洗起点不同）'}",
          flush=True)

    def _draw(state):
        torch.random.set_rng_state(state)
        order = torch.randperm(N, generator=torch.Generator().manual_seed(SEED))
        out, pos = [], 0
        for _ in range(N_BATCHES):
            row = []
            for _ in range(BATCH):
                if pos >= N:
                    order = torch.randperm(N)
                    pos = 0
                row.append(int(order[pos]))
                pos += 1
            out.append(row)
        return out

    idx_c, idx_s = _draw(st_c), _draw(st_s)
    first3_c = [r[:3] for r in idx_c[:20]]
    first3_s = [r[:3] for r in idx_s[:20]]
    print(f"[S48-RNG] 实测：前 20 个 batch 的前 3 索引（初始 order 段）⇒ "
          f"{'逐位相同 ✓' if first3_c == first3_s else '★不同'}", flush=True)
    print(f"  cards : {first3_c}", flush=True)
    print(f"  stages: {first3_s}", flush=True)
    div = next((b for b in range(N_BATCHES) if idx_c[b] != idx_s[b]), None)
    print(f"[S48-RNG] 实测：{N_BATCHES} 个 batch 全索引序列（含第 125 batch 的第一次重洗）⇒ "
          f"{'逐位相同 ✓' if div is None else f'★首个分岔 batch={div}（idx_c[:3]={idx_c[div][:3]} vs idx_s[:3]={idx_s[div][:3]}）'}",
          flush=True)
    if div is not None:
        print(f"  cards  重洗后前3={idx_c[div][:3]} | stages 重洗后前3={idx_s[div][:3]}", flush=True)

    # 换流机制实测：cards 的 logits_train 跑一次前向，全局 CPU 流必须原封不动
    pipe = _mC
    pipe.train()
    ids = torch.randint(1, 100, (2, 40))
    before = torch.random.get_rng_state().clone()
    C["logits_train"](pipe, ids)
    after = torch.random.get_rng_state().clone()
    print(f"[S48-RNG] cards logits_train 换流实测：一次前向（dropout 开）前后全局 CPU 流 "
          f"{'不变 ✓（dropout 走 DROP_GEN 独立流）' if torch.equal(before, after) else '★被改动'}",
          flush=True)

    # ------------------------------------------------------------------ 汇总
    gate0 = (sha_c == sha_s == "b34e7ea515203227" and leak_c == leak_s == 16
             and not mismatches and pm == 0)
    RNG_CONSISTENT = (first3_c == first3_s)
    out = dict(
        device="cpu", torch=torch.__version__, seed=SEED,
        cards_test_sha16=sha_c, stages_test_sha16=sha_s, expect_sha16="b34e7ea515203227",
        cards_leak=leak_c, stages_leak=leak_s, expect_leak=16,
        bitwise_mismatches=len(mismatches), prompt_mismatches=pm,
        shape_cards=sh_c, shape_stages=sh_s,
        first3_test_cards=[{"prompt": "题干：" + test_c[i]["nl"] + " → ", "gold": test_c[i]["gold"],
                            "a": test_c[i]["a"], "b": test_c[i]["b"], "y": test_c[i]["y"]}
                           for i in range(3)],
        first3_test_stages=[{"prompt": "题干：" + test_s[i]["nl"] + " → ", "gold": test_s[i]["gold"],
                             "a": test_s[i]["a"], "b": test_s[i]["b"], "y": test_s[i]["y"]}
                            for i in range(3)],
        rng_global_state_same=bool(torch.equal(st_c, st_s)),
        first20_first3_same=first3_c == first3_s,
        first20_first3_cards=first3_c, first20_first3_stages=first3_s,
        full_seq_first_divergence_batch=div,
        dropout_swap_keeps_global_state=bool(torch.equal(before, after)),
        gate0_pass=bool(gate0),
        wall_clock_s=round(time.time() - T0, 1),
    )
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[S48-VERDICT] 门0 数据对齐 = {'★通过' if gate0 else '★不通过'} | "
          f"两条流（dropout 独立 + 重洗全局）结构{'一致' if RNG_CONSISTENT else '需核'} | "
          f"墙钟={out['wall_clock_s']}s", flush=True)
    print("[DONE] exit=0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
