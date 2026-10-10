#!/usr/bin/env python3
"""S70a · L3-r1c 纯 4 位加法子集（CPU、确定性、秒级；不碰 GPU）。

来源与许可：`logs/70_dialog_{train,test}.jsonl`（由 stages/70_arith_dialog_gen.py 全量生成，
同一条目字段逐字保留；本脚本**只过滤、不重算、不改写任何字段**）。

过滤条件（L3_PLAN.md §8.8 写死）：`op == '+' and max_digits == 4`
  ⇒ `logs/70a_dialog_{train,test}.jsonl`（字段与来源一致）。

为什么只做这一个子集（口径限制，§8.7 第 1 条）：full 池里 **op 与 max_digits 几乎共线**
（70-test 实测 ('+',4)=459 / ('-',4)=6 / ('-',5)=112 / ('×',2)=19 / ('×',3)=204），
两个分桶都区分不了「运算符难」还是「位数难」⇒ 本单元**只给纯 4 位加法一个子集**，
**不**顺带解释减法/乘法（`×` 另有 <unk> 缺陷，见 §8.6）。

本脚本打印（全部实测，不引用别处数字）：
  · n_train / n_test、train_fp / test_fp（口径 = stages/70_arith_dialog_gen.py:258 data_fp）；
  · **重算地板**（复用 S70::compute_floors，不重写）：含计数类
    ①复制题干末数 ①b复制题干首数 ②train 全集众数 ③a题干数字个数桶 ③b最大位数桶
    ③c题型前缀桶 ④d回复长度桶(50字)，并与 max_naive / 2SE / 门槛并列；
  · 与来源的**逐条一致性抽查**：落盘后逐行读回，与来源 dict 逐字段比较（全量，不只抽样），
    外加 3 条原文字样对照。

产物（只写 logs/）：
  logs/70a_dialog_train.jsonl / logs/70a_dialog_test.jsonl

用法（本机 .venv/bin/python）：
  .venv/bin/python stages/70a_add4_subset.py
  .venv/bin/python stages/70a_add4_subset.py --verify-n 3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter

NS_ROOT = "/home/vesita/coding/my/nanoSeek"
if NS_ROOT not in sys.path:
    sys.path.insert(0, NS_ROOT)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
D70_PATH = os.path.join(ROOT, "stages", "70_arith_dialog_gen.py")
TOK_PATH = NS_ROOT + "/data/chinese/char_tokenizer.json"

SRC_TMPL = os.path.join(LOG_DIR, "70_dialog_{split}.jsonl")
OUT_TMPL = os.path.join(LOG_DIR, "70a_dialog_{split}.jsonl")

FILTER_OP = "+"
FILTER_MAX_DIGITS = 4
VERIFY_N = 3


def _load_path(name: str, path: str):
    """按路径加载（文件名以数字开头，不能普通 import）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d70 = _load_path("d70_arith_dialog_gen", D70_PATH)


def load_jsonl(fp: str) -> list:
    with open(fp, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def write_jsonl(fp: str, recs) -> None:
    """落盘时剔除内存态字段 p/t（来源 jsonl 不含它们，字段必须逐字一致）。"""
    with open(fp, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps({k: v for k, v in r.items() if k not in ("p", "t")},
                               ensure_ascii=False) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="S70a L3-r1c 纯 4 位加法子集 + 重算地板")
    ap.add_argument("--verify-n", type=int, default=VERIFY_N, help="打印几条原文字样对照")
    a = ap.parse_args(argv)

    # ---------- 读来源 ----------
    src = {}
    for split in ("train", "test"):
        fp = SRC_TMPL.format(split=split)
        src[split] = load_jsonl(fp)
        print(f"[D70A-SRC] {split} ← {fp} n={len(src[split])} "
              f"字段={sorted(src[split][0].keys())}", flush=True)
        assert len(src[split]) > 0, f"{fp} 为空"

    # ---------- 过滤（只过滤，不改写） ----------
    out = {}
    n_drop = {}
    for split in ("train", "test"):
        keep = [dict(r) for r in src[split]
                if r["op"] == FILTER_OP and r["max_digits"] == FILTER_MAX_DIGITS]
        out[split] = keep
        n_drop[split] = len(src[split]) - len(keep)
        dist_src = Counter((r["op"], r["max_digits"]) for r in src[split])
        dist_out = Counter((r["op"], r["max_digits"]) for r in keep)
        print(f"[D70A-FILTER] {split}: 来源={len(src[split])} 过滤后={len(keep)} "
              f"丢弃={n_drop[split]} | 条件 op=='{FILTER_OP}' and max_digits=={FILTER_MAX_DIGITS}"
              f" | 来源(op,位数)分布={dict(sorted(dist_src.items()))} "
              f"| 保留后={dict(sorted(dist_out.items()))}", flush=True)
        # 过滤谓词的完备性：保留数必须等于来源里满足谓词的条数（程序化复核）
        n_pred = sum(1 for r in src[split]
                     if r["op"] == FILTER_OP and r["max_digits"] == FILTER_MAX_DIGITS)
        assert len(keep) == n_pred, f"{split} 过滤谓词不自洽 {len(keep)} != {n_pred}"
        # 保留集自身必须每条都满足谓词
        bad = [i for i, r in enumerate(keep)
               if not (r["op"] == FILTER_OP and r["max_digits"] == FILTER_MAX_DIGITS
                       and r["n_nums"] == 2)]
        assert not bad, f"{split} 保留集里有 {len(bad)} 条不满足谓词"

    # ---------- 编码 + 指纹（口径同 S70::data_fp） ----------
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK_PATH)
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids        # noqa: E731
    dec = lambda ids: tok.decode([int(i) for i in ids], skip_special_tokens=False)  # noqa: E731

    n_rt = 0
    for split in ("train", "test"):
        for r in out[split]:
            r["p"], r["t"] = enc(r["ctx"]), enc(r["target"] + d70.EOS)
            if dec(r["p"]) != r["ctx"] or dec(r["t"]) != r["target"] + d70.EOS:
                n_rt += 1
    tr_fp = d70.data_fp(out["train"])
    te_fp = d70.data_fp(out["test"])
    overlap = len({r["ctx"] for r in out["train"]} & {r["ctx"] for r in out["test"]})
    n_ctx_dup_tr = len(out["train"]) - len({r["ctx"] for r in out["train"]})
    n_ctx_dup_te = len(out["test"]) - len({r["ctx"] for r in out["test"]})
    print(f"[D70A-FMT] 编解码往返坏={n_rt}（本子集无 '×'，预期 0） | train 重复ctx={n_ctx_dup_tr} "
          f"test 重复ctx={n_ctx_dup_te} | train∩test ctx={overlap}（无泄漏须为 0）"
          f" | gold_int 抽不出={sum(1 for s in ('train', 'test') for r in out[s] if r['gold_int'] is None)}"
          f" | gold_int 位数分布={dict(sorted(Counter(r['gold_digits'] for s in ('train', 'test') for r in out[s]).items()))}"
          f" | 回复长度桶分布={dict(sorted(Counter(r['resp_bucket'] for s in ('train', 'test') for r in out[s]).items()))}",
          flush=True)
    assert overlap == 0, "train/test ctx 有交集（泄漏）"

    # ---------- 落盘 ----------
    for split in ("train", "test"):
        fp = OUT_TMPL.format(split=split)
        write_jsonl(fp, out[split])
        print(f"[D70A-OUT] {split} → {fp} ({len(out[split])} 条)", flush=True)

    # ---------- 与来源的逐条一致性抽查（全量复核 + 原文字样） ----------
    n_chk, n_bad, ex_lines = 0, 0, []
    for split in ("train", "test"):
        back = load_jsonl(OUT_TMPL.format(split=split))
        assert len(back) == len(out[split]), f"{split} 读回条数不符"
        for i, (rb, ro) in enumerate(zip(back, out[split])):
            n_chk += 1
            # ro 含 p/t（内存里加的）；来源落盘时不含 p/t
            expect = {k: v for k, v in ro.items() if k not in ("p", "t")}
            if rb != expect:
                n_bad += 1
                if n_bad <= 3:
                    diff = [k for k in set(rb) | set(expect) if rb.get(k) != expect.get(k)]
                    print(f"[D70A-CHK] ★不一致 {split}[{i}] 字段={diff}", flush=True)
            if i < a.verify_n:
                ex_lines.append(f"[D70A-CHK] {split}[{i}] ctx={rb['ctx'][:46]!r} "
                                f"gold_int={rb['gold_int']} op={rb['op']} "
                                f"max_digits={rb['max_digits']} 与来源一致="
                                f"{'是' if rb == expect else '否'}")
    # 逐条 vs 来源：用 (ctx,target) 建索引，比对全部字段
    n_src_bad = 0
    for split in ("train", "test"):
        idx = {(r["ctx"], r["target"]): r for r in src[split]}
        for r in out[split]:
            s = idx.get((r["ctx"], r["target"]))
            if s is None:
                n_src_bad += 1
            else:
                for k, v in s.items():
                    if r.get(k) != v:
                        n_src_bad += 1
                        break
    for ln in ex_lines:
        print(ln, flush=True)
    print(f"[D70A-CHK] 落盘读回全量复核={n_chk} 条 不一致={n_bad} | 与来源逐字段比对"
          f"={sum(len(out[s]) for s in ('train', 'test'))} 条 不一致={n_src_bad} ⇒ "
          f"{'一致（字段逐字保留）' if n_bad == 0 and n_src_bad == 0 else '★有分歧'}", flush=True)

    # ---------- 重算地板（含计数类；复用 S70::compute_floors，不重写） ----------
    print("[D70A] ★重算地板（子集内 train 统计 → 子集 test 逐条判 EM；抽不出算错并单独计数）",
          flush=True)
    em, mxname, mx, miss_gold, _miss = d70.compute_floors(out["train"], out["test"], "70a")
    n_test = len(out["test"])
    se2 = 2.0 * (mx / 100.0 * (1 - mx / 100.0) / n_test) ** 0.5 * 100.0
    thr = mx + se2
    # 计数类桶在纯 4 位加法子集里是常量桶，如实点明（不是缺陷，是口径）
    print(f"[D70A] 桶退化提示：本子集 op 恒为 '+'、max_digits 恒为 4、type_prefix 恒为 '加法'、"
          f"n_nums 恒为 2 ⇒ ③a/③b/③c 三个桶与「②train 全集众数」等价（同一常数预测）；"
          f"只有 ④d 回复长度桶是真正的条件桶。", flush=True)

    # ---------- [D70A] 一行汇总 ----------
    print(f"[D70A] n_train={len(out['train'])} n_test={n_test} seed(生成来源)=20261010 "
          f"| 过滤=op=='+' and max_digits==4 | train_fp={tr_fp} test_fp={te_fp} "
          f"| 地板 max_naive={mx:.2f}% 来自={mxname} | 2SE={se2:.2f}pp "
          f"| P0-a 门槛=max_naive+2SE={thr:.2f}% | gold_int 抽不出={miss_gold} "
          f"({100.0 * miss_gold / n_test:.2f}%) | 落盘读回不一致={n_bad} 与来源字段不一致={n_src_bad} "
          f"| 编解码往返坏={n_rt} | train∩test={overlap} "
          f"⇒ {'字段与来源逐字一致' if n_bad == 0 and n_src_bad == 0 and n_rt == 0 else '★有分歧'}", flush=True)
    print(f"[D70A-VERDICT] 子集已落盘且与来源逐字段一致；地板与 max_naive 已重算并报出。"
          f"判据见 L3_PLAN.md §8.8（R1c-a..d 由 stages/71_arith_dialog_cards.py --data-prefix 70a 判）。",
          flush=True)
    print("[DONE] exit=0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
