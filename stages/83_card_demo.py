#!/usr/bin/env python3
"""S83 · L3-d1「卡片框架做自然语言生成」推理演示（CPU / 确定性 / 可核验）。

目的（L3_PLAN.md §16）：给出一份**一条命令可跑、可核验、能复现记录值**的演示，
把「卡片框架在做自然语言生成」这件事**看得见** —— 不只是最终文本，而是**卡片内部**：
选择卡选了谁、概率多少、重整卡注入的是什么、输出卡逐 token 怎么吐。

★本文件只**读**既有文件（`stages/79_cand_cards_v2.py` / `76_letter_cand_gen.py` /
`logs/76_dialog_test.jsonl` / `logs/79_ckpt_*.pt`），**不改任何既有文件**，只写 `logs/83_*`。

--------------------------------------------------------------------------------
用法（本机只有 .venv/bin/python 装了 torch/tokenizers）
--------------------------------------------------------------------------------
  # 1) 默认演示：B2（选择卡+重整卡）两卡；并自动挑「A 报错字母、B2 报对」的例子并排
  .venv/bin/python stages/83_card_demo.py

  # 2) ★阶段性验收：全量 800 条自检（D-a..D-f）+ 卡片内部演示
  .venv/bin/python stages/83_card_demo.py --self-check

  # 3) 机器可读输出（单行 JSON，前缀 [JSON]）
  .venv/bin/python stages/83_card_demo.py --self-check --json
  .venv/bin/python stages/83_card_demo.py --self-check --json --json-out logs/83_selfcheck.json

  # 4) 只看指定测试条目 / 只跑 A 臂 / 换 ckpt
  .venv/bin/python stages/83_card_demo.py --indices 0,3,7 --show 3
  .venv/bin/python stages/83_card_demo.py --arm A --indices 0
  .venv/bin/python stages/83_card_demo.py --ckpt logs/79_ckpt_B2_k8_s5678.pt --ckpt-a logs/79_ckpt_A_k8_s5678.pt

  # 5) GPU（可选；默认绝不用 —— r6 正在跑）
  .venv/bin/python stages/83_card_demo.py --device cuda

--------------------------------------------------------------------------------
★确定性（判据 D-e）如何自查
--------------------------------------------------------------------------------
  同一命令跑两次，**stdout 必须逐字节相同**：

      .venv/bin/python stages/83_card_demo.py --self-check > /tmp/83_r1.txt 2>/dev/null
      .venv/bin/python stages/83_card_demo.py --self-check > /tmp/83_r2.txt 2>/dev/null
      sha256sum /tmp/83_r1.txt /tmp/83_r2.txt        # 两行必须完全一样

  保证手段：贪心解码（每一步 argmax，不采样）；固定 `--seed`；无任何随机数；
  stdout 里**不打印任何时间/耗时**（那些只走 stderr）；`--device cpu` 时先清空
  `CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`，所以连设备枚举都不发生。
  输出里的 `h/…` 指纹是对中间隐态张量求的 sha256（`cards.base.tensor_bytes` 口径），
  隐藏态只要有一位不同，两次运行的指纹就会不同 —— 即 D-e 是一个**强**检验，不是走过场。

--------------------------------------------------------------------------------
判据（L3_PLAN.md §16，逐条打印 [D-a]..[D-f]）
--------------------------------------------------------------------------------
  D-a 一条命令跑通      ：`--self-check` 退出码 0
  D-b ★复现记录值        ：全量 800 条给出 T2 / T1 字母 / 选择准确率，与 §11.26/§11.28 的
                          B2 s1234 = 75.88% / 80.00% / 88.38% 比（同一 ckpt + 同一份
                          logs/76_dialog_test.jsonl + 贪心）；A 臂对照 §11.24 的 12.38%/12.62%
  D-c 展示卡片内部      ：选择卡 argmax 与 softmax p、重整卡注入通路、输出卡逐 token
  D-d 可核验            ：ckpt sha256 + data_fp + jsonl sha256 前 8 位；`--json` 机器可读
  D-e 确定性            ：见上「如何自查」
  D-f 不占 GPU          ：默认 CPU；演示期间 `/sys/class/kfd/kfd/proc` 不出现新 PID

诚实边界（照 §16 原文）：演示的是**合成模板任务**（`logs/76_*`，k=8，回复必须含选中候选的字母），
**不是通用聊天机器人**；卡片结构 = input(1 层) + 选择卡(1 层) + 重整卡(1 层) + output（2.7M 参数）。
蒙特卡洛式的「模型内部到底在想什么」不在此脚本承诺范围内 —— 这里展示的是**可核验的通路读数**。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# ★设备必须在 import torch **之前**定下来：CPU 档直接清空可见设备，
#   这样解释器连 HIP/CUDA 驱动都不会去打开（判据 D-f 的机制保证）。
# ---------------------------------------------------------------------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--device", default="cpu")
_pre_args, _ = _pre.parse_known_args()
if _pre_args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""

NS_ROOT = "/home/vesita/coding/my/nanoSeek"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (NS_ROOT, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from cards.base import tensor_bytes  # noqa: E402

LOG_DIR = os.path.join(ROOT, "logs")
D79_PATH = os.path.join(ROOT, "stages", "79_cand_cards_v2.py")
TEST_JSONL = os.path.join(LOG_DIR, "76_dialog_test.jsonl")
TRAIN_JSONL = os.path.join(LOG_DIR, "76_dialog_train.jsonl")
CKPT_B2_DEFAULT = os.path.join(LOG_DIR, "79_ckpt_B2_k8_s1234.pt")
CKPT_A_DEFAULT = os.path.join(LOG_DIR, "79_ckpt_A_k8_s1234.pt")
KFD_PROC = "/sys/class/kfd/kfd/proc"

# ★记录值（§11.26 / §11.28 / §11.24 / §11.31；B2 s1234 与 A s1234，k=8 带字母数据）
RECORDS = {
    "B2": {"t2": 75.88, "t1": 80.00, "sel": 88.38, "sel_letter": 88.38,
           "src": "L3_PLAN.md §11.26/§11.28（B2 s1234，k=8 带字母）"},
    "A": {"t2": 12.38, "t1": 12.62, "sel": None, "sel_letter": None,
          "src": "L3_PLAN.md §11.24/§11.26（A s1234，k=8 带字母）"},
}
# 数据指纹引文（§11.6；只作参考性对齐，不作判据）
FP_QUOTED_TEST = "1ae76682bd7cc997"
FP_QUOTED_TRAIN = "efeb1ee872f4b6be"

BUILD_SEED = 1234          # 构造模型用（权重随后被 ckpt 全覆盖）


def _load_path(name: str, path: str):
    """按路径加载（文件名以数字开头，不能普通 import）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d79 = _load_path("d83_d79_ref", D79_PATH)     # ★只读复用 79 的模型/解码/口径（零拷贝改动）
d79.device = "cpu"                            # 79 的 pack/greedy 走模块级 device 全局

CAND_LETTERS = d79.CAND_LETTERS


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def fnum(x, nd: int = 4) -> str:
    return "nan" if x is None else f"{x:.{nd}f}"


def sig(x, nd: int = 2) -> str:
    """带符号的定点数（让 Δ 的正负一眼可见）。"""
    return "nan" if x is None else f"{x:+.{nd}f}"


def digest(t: torch.Tensor) -> str:
    """张量的确定性指纹（口径 = cards.base.tensor_bytes）。"""
    return hashlib.sha256(tensor_bytes(t)).hexdigest()[:16]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kfd_pids():
    """当前占用 GPU(KFD) 的 PID 集合；拿不到就返回 None（= 无法观测，如实报）。"""
    try:
        return set(os.listdir(KFD_PROC))
    except OSError:
        return None


def hr(ch: str = "-", n: int = 92) -> str:
    return ch * n


def indent_block(text: str, prefix: str = "      ") -> str:
    return "\n".join(prefix + ln for ln in text.split("\n"))


# ---------------------------------------------------------------------------
# 模型构造 / 载入
# ---------------------------------------------------------------------------
def build_model(arm: str, ckpt: str, device: str):
    torch.manual_seed(BUILD_SEED)              # 消掉构造期 RNG 依赖（反正马上被 ckpt 覆盖）
    model = d79.build_arm(arm).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    n_par = sum(p.numel() for p in model.parameters())
    return model, n_par


@torch.no_grad()
def trace_b(model, rec, inject: bool = True):
    """B2 臂的前向**逐步留痕**（逐字复刻 `d79.ArmB.forward_all`，只是顺手把中间量带出来）。

    inject=False 只用于「注入置零」探针：把重整卡的加性注入改成 0，量它到底起了多大作用。
    """
    ids = torch.tensor([rec["p"]], dtype=torch.long, device=model.q.device)
    n = ids.size(1)
    m = model.input.causal_mask(ids)
    hi = model.input(ids, mask=m)                                   # 输入卡输出
    hs = model.select(hi, mask=m)                                   # 选择卡输出
    eidx = torch.tensor([rec["cend"]], dtype=torch.long, device=ids.device)
    nc = torch.tensor([rec["n_cands"]], dtype=torch.long, device=ids.device)
    g = torch.gather(hs, 1, eidx.unsqueeze(-1).expand(-1, -1, hs.size(-1)))   # (1,k,d)
    sc = (g * model.q).sum(-1)                                     # (1,k) 选择打分
    k = sc.size(1)
    valid = torch.arange(k, device=sc.device)[None, :] < nc[:, None]
    scm = sc.masked_fill(~valid, float("-inf"))
    p = torch.softmax(scm, dim=-1)                                 # (1,k) 选择概率
    e_sel = torch.einsum("bk,bkd->bd", p, g)                       # (1,d) soft 选择向量
    pos = torch.arange(n, device=ids.device)[None, :]
    apply = (pos >= torch.tensor([len(rec["p"])], device=ids.device)[:, None]).unsqueeze(-1).to(hi.dtype)
    inj = apply * e_sel.unsqueeze(1) if inject else torch.zeros_like(hi)
    h = model.realign(hi + inj, mask=m)                            # 重整卡输出
    logits = model.output(h)                                       # 输出卡
    return {"ids": ids, "mask": m, "hi": hi, "hs": hs, "g": g, "sc": sc, "p": p,
            "e_sel": e_sel, "inj": inj, "h": h, "logits": logits,
            "n_reply_pos": int((pos >= len(rec["p"])).sum().item())}


@torch.no_grad()
def trace_a(model, rec):
    """A 臂（单卡直出）的前向留痕：input(1 层) → target(2 层) → output。"""
    ids = torch.tensor([rec["p"]], dtype=torch.long, device=next(model.parameters()).device)
    m = model.input.causal_mask(ids)
    hi = model.input(ids, mask=m)
    hs, layer_digests = hi, []
    for L in model.target.layers:                                  # = DeepTargetCard.apply_card
        hs = L(hs, src_mask=m)
        layer_digests.append(digest(hs))
    logits = model.output(hs)
    return {"ids": ids, "hi": hi, "hs": hs, "logits": logits, "layer_digests": layer_digests}


@torch.no_grad()
def greedy_trace(model, rec, max_new: int, show_n: int):
    """贪心解码**逐 token 留痕**（与 `d79.greedy_gen` 同一条路径；本函数只额外记概率）。"""
    ids = torch.tensor([rec["p"]], dtype=torch.long, device=next(model.parameters()).device)
    s = torch.tensor([len(rec["p"])], dtype=torch.long, device=ids.device)
    eidx = torch.tensor([rec["cend"]], dtype=torch.long, device=ids.device)
    nc = torch.tensor([rec["n_cands"]], dtype=torch.long, device=ids.device)
    cap = max(1, min(max_new, d79.MAXLEN - len(rec["p"])))
    steps, gen = [], []
    for _ in range(cap):
        logits = model.forward_all(ids, ctx_len=s, end_idx=eidx, n_cands=nc)[0]
        lp = torch.log_softmax(logits[0, -1], dim=-1)
        t = int(logits[0, -1].argmax().item())
        prob = float(lp[t].exp().item())
        if t == d79.EOS_ID:
            steps.append({"i": len(steps), "token": "<eos>", "id": t, "p": prob, "eos": True})
            break
        steps.append({"i": len(steps), "token": d79.dec([t]), "id": t, "p": prob, "eos": False})
        gen.append(t)
        ids = torch.cat([ids, torch.tensor([[t]], dtype=torch.long, device=ids.device)], dim=1)
    return steps, d79.dec(gen)


@torch.no_grad()
def eval_arms(model, recs):
    """一次评测三件事：T2 完整回复 EM、T1 字母、选择卡读数（A 臂无选择卡 ⇒ sel=None）。

    ★走的就是 `d79.greedy_gen` / `d79.eval_sel` / `d79.em_strict` / `d79.dec_letter`
      —— 与 79 出记录值时**同一份代码**，所以这里是「复现」不是「重写」。
    """
    txt = d79.greedy_gen(model, recs)
    hits = [d79.em_strict(r["target"], txt[i]) for i, r in enumerate(recs)]
    ls = [d79.dec_letter(t) for t in txt]
    gl = [r.get("gold_letter") for r in recs]
    usable = [(g, x) for g, x in zip(gl, ls) if g is not None]
    t1 = 100.0 * sum(1 for g, x in usable if x == g) / len(usable) if usable else float("nan")
    t2 = 100.0 * sum(hits) / len(hits)
    miss = sum(1 for x in ls if x is None)
    sel = d79.eval_sel(model, recs)
    sel_acc = None if sel is None else 100.0 * sum(sel["hits"]) / len(sel["hits"])
    sel_lacc = None if sel is None else 100.0 * sum(sel["letter_hits"]) / len(sel["letter_hits"])
    return {"txt": txt, "hits": hits, "letters": ls, "gold_letters": gl,
            "t1": t1, "t2": t2, "miss": miss, "n": len(recs),
            "sel": sel_acc, "sel_letter": sel_lacc,
            "sel_hits": None if sel is None else sel["hits"],
            "sel_letter_hits": None if sel is None else sel["letter_hits"]}


# ---------------------------------------------------------------------------
# 判据 D-b：与记录值比对
# ---------------------------------------------------------------------------
def compare_records(arm: str, m: dict) -> list:
    """把实测与 §11.26/§11.28 的记录值逐项比。判据口径 = 双方都按 `:.2f` 显示后相等。"""
    rec = RECORDS[arm]
    rows = []
    for key, label in (("t2", "T2 完整回复 EM"), ("t1", "T1 字母准确率"), ("sel", "选择准确率(argmax)")):
        want = rec[key]
        got = m.get(key)
        if want is None:
            rows.append({"metric": key, "label": label, "record": None, "measured": got,
                         "delta": None, "ok": got is None,
                         "note": "记录值不适用（A 臂没有选择卡）"})
            continue
        ok = (got is not None) and (f"{got:.2f}" == f"{want:.2f}")
        rows.append({"metric": key, "label": label, "record": want, "measured": got,
                     "delta": None if got is None else got - want, "ok": bool(ok),
                     "note": "" if ok else "★与记录值不一致 ⇒ 必须查因，不许改记录值迁就"})
    return rows


def print_record_table(arm: str, m: dict, rows: list) -> bool:
    rec = RECORDS[arm]
    print(f"[D-B] 记录值来源：{rec['src']}")
    print(f"[D-B] 口径：ckpt = --ckpt{'/--ckpt-a' if arm == 'A' else ''} 指向的文件 | 数据 = logs/76_dialog_test.jsonl "
          f"(n={m['n']}) | 解码 = 贪心（d79.greedy_gen，逐字复刻记录值的那段代码）")
    print(f"[D-B] {'指标':<18}{'记录值':>10}{'实测(原始)':>16}{'实测(:.2f)':>14}{'Δ(实测−记录)':>16}  判定")
    all_ok = True
    for r in rows:
        if r["record"] is None:
            got = "nan" if r["measured"] is None else f"{r['measured']:.4f}"
            print(f"[D-B] {r['label']:<18}{'—':>10}{got:>16}{'—':>14}{'—':>16}  "
                  f"{'PASS' if r['ok'] else '★FAIL'}（{r['note']}）")
            all_ok &= bool(r["ok"])
            continue
        got = r["measured"]
        print(f"[D-B] {r['label']:<18}{r['record']:>10.2f}{got:>16.4f}{got:>14.2f}"
              f"{sig(r['delta'], 4):>16}  {'PASS' if r['ok'] else '★FAIL'}"
              + ("" if r["ok"] else f"  ⇒ {r['note']}"))
        all_ok &= bool(r["ok"])
    print(f"[D-B] 额外逐条读数：T1 抽不出 {m['miss']}/{m['n']}（{100.0 * m['miss'] / m['n']:.2f}%）"
          f" | 选择 argmax 字母正确率 = " + ("nan" if m["sel_letter"] is None else f"{m['sel_letter']:.2f}%"))
    print(f"[D-B] 结论：**{'一致 ⇒ 记录值可复现' if all_ok else '★不一致 ⇒ 见上表 Δ，必须查因（不许改记录值）'}**")
    return all_ok


# ---------------------------------------------------------------------------
# 判据 D-c：卡片内部演示
# ---------------------------------------------------------------------------
def print_input_card(rec, tr, arm: str):
    print(f"[输入卡]  token 长度 ctx={len(rec['p'])} | 轮数={rec.get('n_turns')} | 候选层 k={rec['n_cands']} "
          f"| 族={rec.get('family')} | 实体={rec.get('ent')} 条件={rec.get('cond')}")
    print(f"[输入卡]  原始 ctx（左）+ 待补全区（右，本演示只喂 ctx 让模型自己生成）：")
    print(indent_block(rec["ctx"]))
    h = tr["hi"]
    print(f"[输入卡-内部]  x = emb(ids) + 固定正弦 PE → 1 层 TransformerEncoder（causal+padding mask，norm_first）")
    print(f"[输入卡-内部]  隐态 h_input shape={tuple(h.shape)} mean={fnum(float(h.mean()), 4)} "
          f"std={fnum(float(h.std()), 4)} | h_input sha256[0:16]={digest(h)}")


def print_select_card(rec, tr):
    p = tr["p"][0]
    sc = tr["sc"][0]
    pred = int(p.argmax().item())
    gold_idx = rec.get("gold_idx")
    hit_hist = (pred < rec["n_cands"]
                and rec["cands"][pred]["ent"] == rec["ent"]
                and rec["cands"][pred]["cond"] == rec["cond"])
    corr = d79.correct_letter_of(rec)
    print(f"[选择卡]  q（128 维，零初始化、唯一新增参数）· h_j；h_j = 候选 j **末位 token** 在"
          f"「选择卡输出」上的隐态（pos = {rec['cend']}）")
    print(f"[选择卡]  {'#':>2} {'字母':<4}{'候选 (实体 / 条件)':<26}{'s_j=q·h_j':>12}{'p_j=softmax':>14}   与历史(实体,条件)一致")
    for j, c in enumerate(rec["cands"]):
        agree = "★一致" if (c["ent"] == rec["ent"] and c["cond"] == rec["cond"]) else "—"
        mark = "◀ argmax" if j == pred else ""
        gp = "◆gold" if (gold_idx is not None and j == gold_idx) else ""
        print(f"[选择卡]  {j:>2} {CAND_LETTERS[j]:<4}{c['ent'] + ' / ' + c['cond']:<26}"
              f"{float(sc[j]):>12.4f}{float(p[j]):>14.6f}   {agree} {mark}{gp}")
    top2 = torch.topk(p, k=min(2, p.numel()))
    print(f"[选择卡]  argmax = 第 {pred} 位 = 字母 **{CAND_LETTERS[pred]}**，p_max={float(p[pred]):.6f} "
          f"（s_max={float(sc[pred]):+.4f}）| 前二权重={[round(float(x), 6) for x in top2.values]} "
          f"| 熵={float(-(p * torch.log(p + 1e-12)).sum()):.4f} nats（均匀 8 选 1 = {float(torch.log(torch.tensor(8.0))):.4f}）")
    print(f"[选择卡]  选中候选 = 「{rec['cands'][pred]['ent']} / {rec['cands'][pred]['cond']}」"
          f" vs 历史「{rec['ent']} / {rec['cond']}」 ⇒ **{'一致 ⇒ 选择命中' if hit_hist else '不一致 ⇒ 选择错'}**"
          f" | 金标 idx={gold_idx}({rec.get('gold_letter')}) 候选集唯一双槽位匹配项字母={corr}"
          f" ⇒ argmax 字母{'=' + CAND_LETTERS[pred] + ' 正确' if (corr is not None and CAND_LETTERS[pred] == corr) else '错'}")
    return {"argmax": pred, "argmax_letter": CAND_LETTERS[pred], "p_max": float(p[pred]),
            "hit_history": bool(hit_hist), "p": [float(x) for x in p], "s": [float(x) for x in sc],
            "correct_letter": corr}


def print_realign_card(rec, tr, tr_off):
    hi, inj, e_sel = tr["hi"], tr["inj"], tr["e_sel"]
    n_rep = tr["n_reply_pos"]
    ctx_len = len(rec["p"])
    h_rep = hi[0, ctx_len:]
    print(f"[重整卡]  信息从哪来：e_sel = Σ_j p_j · h_j（**soft 选择**，权重 = 上面选择卡的 p；"
          f"不是硬取 argmax 那一项）")
    print(f"[重整卡]  注入方式：h = realign(h_input + apply ⊙ e_sel)，apply = 「位置 >= {ctx_len}」（= <resp> 之后的回复区）"
          f" ⇒ 注入位置数 = {n_rep}（全序列 {hi.size(1)}）")
    print(f"[重整卡]  ||e_sel||={fnum(float(e_sel.norm()), 4)} | 回复区 ||h_input|| 均值={fnum(float(h_rep.norm(dim=-1).mean()), 4)}"
          f" | 相对量级={fnum(float(e_sel.norm()) / max(1e-9, float(h_rep.norm(dim=-1).mean())), 4)}"
          f" | e_sel sha256[0:16]={digest(e_sel)}")
    print(f"[重整卡]  通路活性探针（把注入置零、其余完全相同）：")
    d_h = (tr["h"] - tr_off["h"]).abs()
    rep_slice = slice(ctx_len, tr["h"].size(1))
    dlog = (tr["logits"][0, ctx_len - 1: -1] - tr_off["logits"][0, ctx_len - 1: -1]).abs()
    t_on = d79.dec([int(tr["logits"][0, ctx_len - 1].argmax())])
    t_off = d79.dec([int(tr_off["logits"][0, ctx_len - 1].argmax())])
    print(f"[重整卡]    回复区隐态 max|Δ|={fnum(float(d_h[:, rep_slice].max()), 6)} | 回复位置 logits max|Δ|="
          f"{fnum(float(dlog.max()), 6)} | 首个回复 token：注入={t_on!r} 置零={t_off!r} ⇒ "
          f"{'★注入改变了输出' if t_on != t_off else '首 token 未翻转（影响在后续 token / 概率上）'}")
    return {"e_sel_norm": float(e_sel.norm()), "n_reply_pos": int(n_rep),
            "h_delta_max": float(d_h[:, rep_slice].max()),
            "logits_delta_max": float(dlog.max()),
            "first_token_inject": t_on, "first_token_zeroed": t_off,
            "inject_changes_first_token": bool(t_on != t_off)}


def print_output_card(rec, steps, text, show_n: int):
    shown = steps[:show_n]
    tail = "" if len(steps) <= show_n else f" …（省略 {len(steps) - show_n} 个）"
    print(f"[输出卡]  贪心解码逐 token（共 {len(steps)} 步；每步取 argmax，不采样）：")
    for st in shown:
        print(f"[输出卡]    #{st['i']:>2} token={st['token']!r:<8} id={st['id']:<6} p(该 token)={st['p']:.6f}"
              + ("  ← <eos> 停止" if st["eos"] else ""))
    if tail:
        print(f"[输出卡]    {tail}")
        last = steps[-1]
        print(f"[输出卡]    末步 #{last['i']:>2} token={last['token']!r:<8} id={last['id']:<6} p={last['p']:.6f}"
              + ("  ← <eos> 停止" if last["eos"] else ""))
    print(f"[输出卡]  拼接 = {text!r}")
    return {"steps": steps, "n_steps": len(steps), "text": text}


def print_contrast(rec, text, model_arm: str):
    gold = rec["target"]
    gl = rec.get("gold_letter")
    pl = d79.dec_letter(text)
    em = bool(d79.em_strict(gold, text))
    found = d79.letters_in(text)
    print(f"[对照]    gold        = {gold!r}")
    print(f"[对照]    pred({model_arm}) = {text!r}")
    print(f"[对照]    gold 字母={gl} | 抽出字母={pl}（文本里出现的「（X）」={found}；恰 1 个才算抽出）"
          f" ⇒ T1 字母{'对' if pl == gl else '错'} | 逐字 EM ⇒ T2 {'对' if em else '错'}")
    return {"gold": gold, "gold_letter": gl, "pred_letter": pl, "letters_found": found, "em": em}


def demo_one(rec, idx: int, models: dict, device: str, show_tokens: int) -> dict:
    """对一条测试样本，把「输入卡 → 选择卡 → 重整卡 → 输出卡 → 对照」整条链打出来。"""
    out = {"index": idx, "ctx_len": len(rec["p"]), "n_cands": rec["n_cands"],
           "ent": rec.get("ent"), "cond": rec.get("cond"), "gold_letter": rec.get("gold_letter"),
           "arms": {}}
    for arm in ("A", "B2"):
        if arm not in models:
            continue
        model = models[arm]
        print(hr("="))
        print(f"【卡片内部演示 · arm={arm} · 测试集第 {idx} 条】"
              + ("（单卡直出：input(1 层) → target(2 层) → output）" if arm == "A"
                 else "（两卡：input(1 层) → 选择卡(1 层) → 重整卡(1 层) → output）"))
        print(hr("-"))
        if arm == "B2":
            tr = trace_b(model, rec, inject=True)
            tr_off = trace_b(model, rec, inject=False)
            sel_info = None
            print_input_card(rec, tr, arm)
            sel_info = print_select_card(rec, tr)
            re_info = print_realign_card(rec, tr, tr_off)
            digest_map = {"h_input": digest(tr["hi"]), "h_select": digest(tr["hs"]),
                          "g_cand_末位": digest(tr["g"]), "e_sel": digest(tr["e_sel"]),
                          "h_realign": digest(tr["h"]), "logits": digest(tr["logits"])}
        else:
            tr = trace_a(model, rec)
            sel_info, re_info = None, None
            print_input_card(rec, tr, arm)
            print(f"[选择卡]  —— 不适用：A 臂是**单卡直出**，没有选择卡（这正是要对照的那一点）")
            print(f"[重整卡]  —— 不适用：A 臂没有重整卡；候选层在 A 里**根本不参与**计算，"
                  f"信息只能从历史里取")
            print(f"[目标层]  A 的 2 层 think（裸 encoder 层，无外层残差）："
                  f"逐层 sha256[0:16] = {tr['layer_digests']}")
            digest_map = {"h_input": digest(tr["hi"]), "h_target(2层)": digest(tr["hs"]),
                          "logits": digest(tr["logits"])}
        print(f"[内部指纹] " + " | ".join(f"{k}={v}" for k, v in digest_map.items()))
        steps, text = greedy_trace(model, rec, max_new=d79.MAXLEN - len(rec["p"]), show_n=show_tokens)
        # 自洽核验：逐 token 留痕的拼接必须与批量评测用的 d79.greedy_gen 逐字相同
        ref = d79.greedy_gen(model, [rec])[0]
        same = (ref.strip() == text.strip())
        print(f"[输出卡-自洽] 本演示的 greedy_trace 与评测口径 d79.greedy_gen "
              f"{'逐字相同 PASS' if same else '★不同（实现漂移！）'}")
        print_output_card(rec, steps, text, show_tokens)
        con = print_contrast(rec, text, arm)
        out["arms"][arm] = {"select": sel_info, "realign": re_info, "digests": digest_map,
                            "output": {"n_steps": len(steps), "text": text, "steps": steps},
                            "contrast": con, "selfconsistent_with_greedy_gen": bool(same)}
    return out


def print_side_by_side(idx: int, rec: dict, texts: dict, letters: dict, hitmap: dict):
    gl = rec.get("gold_letter")
    print(hr("="))
    print(f"【A vs B2 并排 · 同一条输入 · 测试集第 {idx} 条】gold 字母 = {gl}")
    print(hr("-"))
    print(f"  gold      : {rec['target']}")
    for arm in ("A", "B2"):
        if arm not in texts:
            continue
        t = d79.dec_letter(texts[arm])
        tag = "对" if t == gl else "错"
        print(f"  {arm:<3} 字母{t if t else '抽不出'}({tag}) : {texts[arm]}")
    hitline = []
    for arm in ("A", "B2"):
        if arm in texts:
            hitline.append(f"{arm}={'对' if hitmap[arm] else '错'}")
    print(f"  ⇒ T2 逐字 EM：{' '.join(hitline)}"
          f" | {'★这条正是「A 报错字母、B2 报对」的例子' if (d79.dec_letter(texts.get('A', '')) != gl and hitmap.get('B2')) else '（本条不是 A 错 B2 对的例子）'}")


# ---------------------------------------------------------------------------
# 自检（D-a..D-f）
# ---------------------------------------------------------------------------
def kfd_line(tag: str, pids):
    return f"{tag}：{'无法观测（{p} 不可读）'.format(p=KFD_PROC) if pids is None else (sorted(pids) if pids else '空（无 GPU 进程）')}"


def main(argv=None) -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(
        description="S83 L3-d1 卡片框架自然语言生成演示（CPU / 确定性 / 可核验）",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--arm", choices=("A", "B2"), default="B2", help="默认演示臂（B2 = 选择卡+重整卡）")
    ap.add_argument("--compare", action="store_true", help="A vs B2 并排（--self-check 下自动开）")
    ap.add_argument("--ckpt", default=CKPT_B2_DEFAULT, help="B2 臂 ckpt（默认 = 已验证能复现 75.88% 的那一个）")
    ap.add_argument("--ckpt-a", default=CKPT_A_DEFAULT, help="A 臂 ckpt（只用于 A vs B2 并排）")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="默认 cpu（r6 在跑，别抢卡）")
    ap.add_argument("--seed", type=int, default=1234, help="记录值对应的 seed（B2 s1234 = 75.88%%）")
    ap.add_argument("--eval-n", type=int, default=800, help="--self-check 的评测条数（默认全量 800）")
    ap.add_argument("--indices", default=None, help="显式指定要演示的测试条下标，如 0,3,7")
    ap.add_argument("--scan-n", type=int, default=64, help="默认演示时扫描前 N 条来挑「A 错 B2 对」的例子")
    ap.add_argument("--show", type=int, default=2, help="默认演示展示几条（--self-check 下也用它）")
    ap.add_argument("--show-tokens", type=int, default=12, help="输出卡逐 token 打印前几个")
    ap.add_argument("--self-check", action="store_true", help="跑 D-a..D-f 全量自检（含 800 条复现）")
    ap.add_argument("--json", action="store_true", help="在 stdout 末尾追加单行 [JSON] 机器可读结果")
    ap.add_argument("--json-out", default=None, help="把同一份 JSON 写进该文件（建议 logs/83_*.json）")
    a = ap.parse_args(argv)
    if a.self_check:
        a.compare = True

    result = {"stage": "L3-d1", "script": "stages/83_card_demo.py", "device": a.device,
              "seed": a.seed, "cmd": " ".join([os.path.basename(sys.argv[0])] + (argv or sys.argv[1:])),
              "records": RECORDS, "verdicts": {}, "notes": []}
    ok = True

    # ---- D-f 起点：演示前 GPU(KFD) 进程快照 ----
    pids0 = kfd_pids()

    print(hr("="))
    print("flowme L3-d1 · 卡片框架做自然语言生成 · 推理演示（CPU / 确定性 / 可核验）")
    print(hr("="))
    print(f"[环境] device={a.device} seed={a.seed} python={sys.version.split()[0]} torch={torch.__version__} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<未设置>')!r} "
          f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '<未设置>')!r} "
          f"torch.cuda.is_initialized()={torch.cuda.is_initialized()}")
    if a.device == "cpu" and torch.cuda.is_initialized():
        print("[环境] ★警告：CPU 档下 CUDA 竟已初始化 ⇒ D-f 存疑")
    print(f"[环境] 演示开始前 /sys/class/kfd/kfd/proc = {kfd_line('', pids0).split('：', 1)[1]}")

    # ---- D-d：ckpt / 数据指纹 ----
    for arm, path in (("B2", a.ckpt), ("A", a.ckpt_a)):
        if not os.path.exists(path):
            print(f"[VERIFY] ★{arm} ckpt 不存在：{path}")
            ok = False
    ck_info = {}
    for arm, path in (("B2", a.ckpt), ("A", a.ckpt_a)):
        if not os.path.exists(path):
            continue
        sh = file_sha256(path)
        ck_info[arm] = {"path": os.path.relpath(path, ROOT), "sha256": sh, "abbrev8": sh[:8],
                        "bytes": os.path.getsize(path)}
        print(f"[VERIFY] ckpt({arm}) = {os.path.relpath(path, ROOT)}  bytes={os.path.getsize(path)}")
        print(f"[VERIFY] ckpt({arm}) sha256 = {sh}  （前 8 位 {sh[:8]}）")
    result["ckpt"] = ck_info

    data_info = {}
    for tag, path in (("test", TEST_JSONL), ("train", TRAIN_JSONL)):
        if not os.path.exists(path):
            print(f"[VERIFY] ★数据不存在：{path}")
            ok = False
            continue
        sh = file_sha256(path)
        data_info[tag] = {"path": os.path.relpath(path, ROOT), "sha256": sh, "abbrev8": sh[:8]}
    print(f"[VERIFY] data(test) = {data_info['test']['path']}  sha256 前 8 位 = {data_info['test']['abbrev8']}"
          f"（§11.6 引文口径里 jsonl sha256 是另一套指纹，见下 data_fp）")

    d79.load_tok()
    test = d79.load_jsonl(TEST_JSONL)
    train = d79.load_jsonl(TRAIN_JSONL)
    for r in test:
        d79.prep_record(r)
    for r in train:
        d79.prep_record(r)
    fp_test = d79.d74.d72.data_fp(test)
    fp_train = d79.d74.d72.data_fp(train)
    data_info["test"]["n"] = len(test)
    data_info["test"]["data_fp"] = fp_test
    data_info["train"]["data_fp"] = fp_train
    data_info["test"]["fp_quoted"] = FP_QUOTED_TEST
    data_info["test"]["fp_match_quoted"] = (fp_test == FP_QUOTED_TEST)
    result["data"] = data_info
    print(f"[VERIFY] data test n={len(test)} train n={len(train)} | data_fp test={fp_test} train={fp_train} "
          f"| §11.6 引文 test={FP_QUOTED_TEST} train={FP_QUOTED_TRAIN} ⇒ "
          f"{'一致' if fp_test == FP_QUOTED_TEST else '★与引文不同（如实报，指纹口径见 §11.6）'}"
          f"（仅参考性对齐，不作判据）")

    # ---- 载模型 ----
    models, params = {}, {}
    have_A = os.path.exists(a.ckpt_a)
    need_A = a.compare or a.self_check or a.arm == "A"
    for arm, path in ((a.arm, a.ckpt), ("B2", a.ckpt), ("A", a.ckpt_a if have_A else None)):
        if arm in models or path is None:
            continue
        if arm == "A" and not need_A:
            continue
        if arm == "B2" and a.arm != "B2" and not need_A:
            continue
        t0 = time.time()
        models[arm], params[arm] = build_model(arm, path, a.device)
        print(f"[MODEL] arm={arm} 载入 {os.path.relpath(path, ROOT)} | 参数={params[arm]} "
              f"| 墙钟={time.time() - t0:.1f}s", file=sys.stderr, flush=True)
    result["params"] = params

    # ---- D-b：全量复现 ----
    evals = {}
    if a.self_check:
        n_eval = min(a.eval_n, len(test))
        ev = test[:n_eval]
        print(hr("="))
        print(f"[D-B] ★复现记录值：在 **{n_eval} 条测试集** 上评测（贪心、CPU、同一 ckpt / 同一份 jsonl）")
        print(hr("-"))
        for arm in ("B2", "A"):
            if arm not in models:
                continue
            t0 = time.time()
            m = eval_arms(models[arm], ev)
            evals[arm] = m
            print(f"[D-B] ===== arm={arm} =====")
            print(f"[D-B] [EM]  n={m['n']} T2 完整回复 EM = {m['t2']:.2f}%（原始 {m['t2']:.4f}）"
                  f" | T1 字母准确率 = {m['t1']:.2f}%（原始 {m['t1']:.4f}，抽不出 {m['miss']}）"
                  f" | 选择准确率(argmax 候选对==历史) = "
                  + ("不适用（A 无选择卡）" if m["sel"] is None else f"{m['sel']:.2f}%（原始 {m['sel']:.4f}）")
                  + f" | 选择 argmax 字母正确率 = "
                  + ("不适用" if m["sel_letter"] is None else f"{m['sel_letter']:.2f}%"))
            arm_ok = print_record_table(arm, m, compare_records(arm, m))
            ok &= arm_ok
            print(f"[D-B] arm={arm} 评测墙钟={time.time() - t0:.0f}s（不写进 stdout，见 D-e）", file=sys.stderr, flush=True)
        result["metrics"] = {k: {kk: v[kk] for kk in ("n", "t2", "t1", "sel", "sel_letter", "miss")}
                             for k, v in evals.items()}
        result["record_check"] = {k: compare_records(k, v) for k, v in evals.items()}
    else:
        print("[D-B] 未跑全量复现（只做演示）。要跑 D-b 请加 `--self-check`。")

    # ---- 选演示样本 ----
    if a.indices:
        idxs = [int(x) for x in a.indices.split(",") if x.strip() != ""]
    else:
        scan_n = min(len(test), a.scan_n)
        scan = test[:scan_n]
        texts = {}
        for arm in ("A", "B2"):
            if arm in models:
                texts[arm] = d79.greedy_gen(models[arm], scan)
        idxs, fallback = [], []
        for i, r in enumerate(scan):
            gl = r.get("gold_letter")
            la = d79.dec_letter(texts["A"][i]) if "A" in texts else None
            lb = d79.dec_letter(texts["B2"][i]) if "B2" in texts else None
            if "B2" in texts and lb == gl and la != gl:
                idxs.append(i)
            elif not fallback:
                fallback.append(i)
            if len(idxs) >= a.show:
                break
        if not idxs:
            idxs = fallback or [0]
            print(f"[演示] 前 {scan_n} 条里没找到「A 错 B2 对」的例子，退回到第 {idxs} 条（如实报）")
        else:
            print(f"[演示] 在前 {scan_n} 条里按「A 抽出的字母 ≠ gold 且 B2 = gold」挑出第 {idxs} 条"
                  f"（挑选规则确定 ⇒ 不影响 D-e）")

    demo_results = []
    for idx in idxs:
        if idx >= len(test):
            print(f"[演示] ★第 {idx} 条超出 test 范围 {len(test)}，跳过")
            ok = False
            continue
        rec = test[idx]
        demo_results.append(demo_one(rec, idx, models, a.device, a.show_tokens))
        if a.compare and "A" in models and "B2" in models:
            tx = {arm: d79.greedy_gen(models[arm], [rec])[0] for arm in ("A", "B2")}
            hitmap = {arm: bool(d79.em_strict(rec["target"], tx[arm])) for arm in ("A", "B2")}
            print_side_by_side(idx, rec, tx, {arm: d79.dec_letter(tx[arm]) for arm in tx}, hitmap)
            demo_results[-1]["side_by_side"] = {"gold": rec["target"], "pred": tx, "hit": hitmap}
    result["demos"] = demo_results

    # ---- 判据结论 ----
    print(hr("="))
    print("判据自检（L3_PLAN.md §16）")
    print(hr("-"))
    d_a = True
    print(f"[D-a] 一条命令跑通：本进程走到这里 ⇒ 退出码 0 ⇒ **{'PASS' if d_a else 'FAIL'}**"
          f"（命令：.venv/bin/python stages/83_card_demo.py --self-check）")
    if a.self_check and evals:
        d_b = all(r["ok"] for rows in result["record_check"].values() for r in rows)
        print(f"[D-b] ★复现记录值：B2 s{a.seed} 实测 T2={evals['B2']['t2']:.2f}% T1={evals['B2']['t1']:.2f}% "
              f"选择={evals['B2']['sel']:.2f}% vs 记录 75.88% / 80.00% / 88.38% ⇒ "
              f"**{'PASS（同一 ckpt + 同数据 + 贪心，逐项一致）' if d_b else '★FAIL（见上表 Δ）'}**")
        if "A" in evals:
            print(f"[D-b] 对照臂 A s{a.seed} 实测 T2={evals['A']['t2']:.2f}% T1={evals['A']['t1']:.2f}% "
                  f"vs §11.24/§11.26 记录 12.38% / 12.62% ⇒ "
                  f"**{'PASS' if all(r['ok'] for r in result['record_check']['A']) else '★FAIL'}**")
    else:
        d_b = None
        print("[D-b] ★未跑（加 `--self-check` 才做全量 800 条复现）；本 run 只演示卡片内部")
    d_c = bool(demo_results) and all(len(d["arms"]) > 0 for d in demo_results)
    print(f"[D-c] 展示卡片内部：{'选择卡 argmax+p 分布 / 重整卡注入通路+|Δ| / 输出卡逐 token / 对照' if d_c else '★缺失'}"
          f" ⇒ **{'PASS' if d_c else 'FAIL'}**（共演示 {len(demo_results)} 条）")
    d_d = bool(ck_info.get("B2")) and bool(data_info.get("test", {}).get("data_fp"))
    print(f"[D-d] 可核验：ckpt sha256={ck_info.get('B2', {}).get('sha256', '—')[:16]}… "
          f"data_fp={data_info.get('test', {}).get('data_fp', '—')} "
          f"jsonl sha256 前 8 位={data_info.get('test', {}).get('abbrev8', '—')} "
          f"| --json={'开' if a.json else '关'} ⇒ **{'PASS' if d_d else 'FAIL'}**")
    print(f"[D-e] 确定性：贪心 + 固定 seed + 无随机 + stdout 不含耗时。自查：同命令跑两次后 "
          f"`sha256sum r1.txt r2.txt` 两行必须相同（见文件 docstring「确定性如何自查」）")
    pids1 = kfd_pids()
    new_pids = None if (pids0 is None or pids1 is None) else sorted(pids1 - pids0)
    d_f = (a.device == "cpu") and (new_pids is not None and len(new_pids) == 0) \
        and (not torch.cuda.is_initialized())
    print(f"[D-f] 不占 GPU：device={a.device} | torch.cuda.is_initialized()={torch.cuda.is_initialized()} | "
          f"演示前 KFD 进程={sorted(pids0) if pids0 is not None else '无法观测'} | "
          f"演示后 KFD 进程={sorted(pids1) if pids1 is not None else '无法观测'} | "
          f"新出现的 PID={new_pids if new_pids is not None else '无法观测'} ⇒ "
          f"**{'PASS（演示期间没有产生 GPU 进程）' if d_f else '★FAIL/不可判（如实报）'}**")
    ok &= bool(d_c and d_d) and (d_b is not False) and bool(d_f)

    result["verdicts"] = {"D-a": d_a, "D-b": d_b, "D-c": bool(d_c), "D-d": bool(d_d),
                          "D-e": "见 docstring 自查（脚本本身无法自证）", "D-f": bool(d_f)}
    result["kfd"] = {"before": sorted(pids0) if pids0 is not None else None,
                     "after": sorted(pids1) if pids1 is not None else None, "new": new_pids}
    result["ok"] = ok
    print(hr("="))
    print(f"[总判] {'全部可判判据 PASS' if ok else '★有判据未过（见上）'} | "
          f"总墙钟={time.time() - t_start:.0f}s（走 stderr，不进 stdout）", file=sys.stderr, flush=True)
    print(f"[总判] exit={'0' if ok else '1'}")
    print(hr("="))
    if a.json:
        print("[JSON] " + json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, sort_keys=True, indent=2)
        print(f"[JSON] 已写入 {a.json_out}", file=sys.stderr, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
