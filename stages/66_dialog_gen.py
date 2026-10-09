#!/usr/bin/env python3
"""S66 · L2「跨轮指代式单轮回复」合成对话数据生成器（CPU、秒级、确定性 seed）。

设计见 L2_PLAN.md §3；本文件提供 P1–P5 判据的数据面与地板口径。

每条样本 = 轮1 用户话 / 轮1 系统话 / 轮2 用户话 / 轮2 系统话 / 轮3 用户话 + 轮3 目标回复。

三条不变量（构造时逐条断言；违例必须为 0，否则退出码非 0）：
  A  轮1 里出现一个实体（闭合词表 NAMES），且轮3 用户话里绝不出现该实体；
  B  目标回复里必然出现该实体 + 轮3 的一个新信息（v3 槽位）⇒ 不读历史写不出；
  C  目标回复由「族模板 + 槽位」确定性生成 ⇒ ctx→target 是函数（唯一可判）。

产物：logs/66_dialog_train.jsonl / logs/66_dialog_test.jsonl
每条字段： ctx / target / gold / ent / turn3_has_ent(必须 False)
          + t3 / t1_user / t1_sys / t2_user / t2_sys / family / v3（地板与消融用）
          + n_turns / ent_count / ent_pos / t3_chars / t3_tok（计数类地板特征）

指纹口径与 stages/65_textgen_cards.py::data_fp 逐字一致：逐条 len|ids 拼接后 sha256[:16]。

只写 logs/ 与本文件；不 import torch（纯 CPU、秒级）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import sys
from collections import Counter

NS_ROOT = "/home/vesita/coding/my/nanoSeek"
if NS_ROOT not in sys.path:
    sys.path.insert(0, NS_ROOT)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
TOK_PATH = NS_ROOT + "/data/chinese/char_tokenizer.json"
OUT_TMPL = os.path.join(LOG_DIR, "66_dialog_{split}.jsonl")

N_TRAIN = 7000
N_TEST = 800
SEED = 20261010
EOS = "<eos>"

# ---------------- 闭合实体词表（40 个人名；不依赖任何外部语料） ----------------
NAMES = (
    "张伟", "李娜", "王芳", "刘洋", "陈静", "杨帆", "赵磊", "黄敏", "周涛", "吴倩",
    "徐强", "孙丽", "马超", "朱婷", "胡军", "郭涛", "林晓", "何静", "高翔", "罗敏",
    "郑爽", "梁静", "谢军", "宋佳", "唐磊", "许峰", "韩雪", "冯刚", "曹阳", "邓超",
    "彭飞", "曾毅", "萧然", "田甜", "董浩", "潘玮", "蒋雯", "蔡明", "余音", "杜鹃",
)

# ---------------- 轮2 填充语（与任务无关；提供 ctx 唯一性 + 干扰项） ----------------
T2_TOPICS = (
    "今天天气不错，适合出门散步。",
    "最近公司的事情有点多。",
    "我昨天看了一部电影，挺好看的。",
    "周末打算在家休息一下。",
    "早上出门的时候有点堵车。",
    "楼下新开了一家面包店。",
    "我最近在学做菜，还在摸索。",
    "这几天降温挺明显的。",
    "中午和同事一起吃了碗面。",
    "我养的那盆绿萝长出新叶子了。",
    "昨天睡得有点晚，今天没什么精神。",
    "最近在追一部纪录片。",
    "明天要去机场接个朋友。",
    "我家猫又把杯子推下桌子了。",
    "这段时间在整理以前的照片。",
    "晚上准备去楼下跑跑步。",
    "刚泡了杯茶，准备看会儿书。",
    "上个月买的书还没拆封。",
    "隔壁在装修，白天有点吵。",
    "今天地铁上人特别多。",
    "打算下周请两天假出去走走。",
    "我最近在练习早睡早起。",
    "昨天试着做了一道新菜。",
    "阳台上晒的被子收进来了。",
    "手机换了新电池，续航好多了。",
    "楼下的桂花开了，味道很好闻。",
    "周末想去逛逛旧书店。",
    "今天的工作总算告一段落。",
    "刚把房间彻底打扫了一遍。",
    "朋友推荐了一家不错的咖啡店。",
    "我最近迷上了拼图。",
    "早上煮了个鸡蛋当早餐。",
    "这两天在收拾换季的衣服。",
    "同事送了我一盆小多肉。",
    "昨天散步走了很远。",
    "新买的台灯亮度挺合适。",
    "傍晚的天空很好看。",
    "我打算把作息调回正常。",
    "今天路过公园，花都开了。",
    "晚上约了朋友一起吃火锅。",
)

T2_SYS = (
    "听起来挺好的。",
    "嗯，我记下了。",
    "那确实不错。",
    "这样安排挺好。",
    "明白，你先忙。",
    "好的，随时说。",
    "嗯嗯，继续。",
    "收到，谢谢你告诉我。",
)

# ---------------- 5 个「族」= 5 套（轮1 用户/轮1 系统/轮3 用户/目标回复）模板 ----------------
# 每族的 t3_user 只带 v3 槽位、绝不带实体槽位 ⇒ 不变量 A 由构造保证。
FAMILIES = (
    {
        "key": "快递",
        "t1_user": "我叫{ent}，上周寄了一个包裹。",
        "t1_sys": "好的{ent}，包裹已经登记在您名下。",
        "t3_user": "我的包裹什么时候能到{v3}？",
        "t3_sys": "{ent}，您的包裹预计{v3}送达，请留意短信。",
        "v3": ("明天下午", "后天早上", "这周五", "下周一"),
    },
    {
        "key": "预约",
        "t1_user": "我叫{ent}，想预约牙科门诊。",
        "t1_sys": "收到{ent}，预约通道已为您打开。",
        "t3_user": "帮我改成{v3}吧。",
        "t3_sys": "{ent}，您的门诊已改到{v3}，请提前十分钟到。",
        "v3": ("周三上午", "周四下午", "周六上午", "周日下午"),
    },
    {
        "key": "订座",
        "t1_user": "我叫{ent}，订了一个靠窗的位置。",
        "t1_sys": "好的{ent}，靠窗座位已经锁定。",
        "t3_user": "我们一共{v3}，麻烦加个位子。",
        "t3_sys": "{ent}，已为您把座位改成{v3}的桌子，靠窗保留。",
        "v3": ("三个人", "四个人", "五个人", "六个人"),
    },
    {
        "key": "售后",
        "t1_user": "我叫{ent}，买的那台咖啡机有点问题。",
        "t1_sys": "{ent}您好，已为您登记售后工单。",
        "t3_user": "我想让师傅{v3}上门看看。",
        "t3_sys": "{ent}，师傅会在{v3}上门，请保持电话畅通。",
        "v3": ("明天上午", "明天下午", "周六上午", "周日下午"),
    },
    {
        "key": "会员",
        "t1_user": "我叫{ent}，是你们的老会员了。",
        "t1_sys": "欢迎回来{ent}，会员权益已生效。",
        "t3_user": "帮我把积分换成{v3}吧。",
        "t3_sys": "{ent}，已为您兑换{v3}，可在下次消费时使用。",
        "v3": ("十元券", "二十元券", "三十元券", "五十元券"),
    },
)

FAM_BY_KEY = {f["key"]: f for f in FAMILIES}
N_TEMPLATE_FAMILIES = len(FAMILIES)
N_REPLY_TEMPLATES = sum(len(f["v3"]) for f in FAMILIES)   # 族 × 新信息槽 = 实例化回复模板数
N_V3 = sum(len(f["v3"]) for f in FAMILIES)


# ---------------- 渲染：与 nanoSeek DialogueStream(用户：/ <resp> / <eos>) 同款 ----------------
def render_ctx(r1u: str, r1s: str, r2u: str, r2s: str, r3u: str) -> str:
    """手写权威格式；main 里用 DialogueStream 逐条核对（必须是同一条流）。"""
    return (f"用户：{r1u}\n<resp>{r1s}{EOS}\n"
            f"用户：{r2u}\n<resp>{r2s}{EOS}\n"
            f"用户：{r3u}\n<resp>")


def gen_records(seed: int = SEED, n_train: int = N_TRAIN, n_test: int = N_TEST) -> list:
    """确定性生成 n_train+n_test 条**互不相同**的 ctx（组合无放回抽样）。

    组合空间 = 5 族 × 40 实体 × 4 新信息 × 40 填充语 = 32000，先 shuffle 再切前 need 条。
    """
    rng = random.Random(seed)
    combos = [(fi, ent, v3, ti)
              for fi, fam in enumerate(FAMILIES)
              for ent in NAMES
              for v3 in fam["v3"]
              for ti in range(len(T2_TOPICS))]
    rng.shuffle(combos)
    need = n_train + n_test
    if len(combos) < need:
        raise SystemExit(f"[D66] 组合空间不足：{len(combos)} < {need}")
    recs = []
    for fi, ent, v3, ti in combos[:need]:
        fam = FAMILIES[fi]
        t2s = T2_SYS[rng.randrange(len(T2_SYS))]
        r1u = fam["t1_user"].format(ent=ent)
        r1s = fam["t1_sys"].format(ent=ent)
        r3u = fam["t3_user"].format(v3=v3)
        t2u = T2_TOPICS[ti]
        gold = fam["t3_sys"].format(ent=ent, v3=v3)
        ctx = render_ctx(r1u, r1s, t2u, t2s, r3u)
        recs.append(dict(
            ctx=ctx, target=gold, gold=gold, ent=ent, turn3_has_ent=(ent in r3u),
            t3=r3u, t1_user=r1u, t1_sys=r1s, t2_user=t2u, t2_sys=t2s,
            family=fam["key"], v3=v3, n_turns=3,
            ent_count=sum(1 for n in NAMES if n in ctx),
            ent_pos=ctx.index(ent), t3_chars=len(r3u),
            v3_in_t3=(v3 in r3u), ent_in_target=(ent in gold), v3_in_target=(v3 in gold),
        ))
    return recs


# ---------------- 不变量检查（逐条断言式计数） ----------------
def check_invariants(recs, tag="all"):
    """返回 (通过计数, 逐条违例计数)；另做「词表级」静态检查（与具体样本无关）。"""
    ok = Counter()
    vio = Counter()
    for r in recs:
        a = (r["ent"] in r["t1_user"]) and (not r["turn3_has_ent"]) and (r["ent"] not in r["t3"])
        b = r["ent_in_target"] and r["v3_in_target"] and r["v3_in_t3"]
        ok["A"] += int(a)
        ok["B"] += int(b)
        vio["A"] += int(not a)
        vio["B"] += int(not b)
    # C：ctx 唯一 + ctx→target 是函数
    seen = {}
    dup, conflict = 0, 0
    for r in recs:
        if r["ctx"] in seen:
            dup += 1
            if seen[r["ctx"]] != r["target"]:
                conflict += 1
        seen[r["ctx"]] = r["target"]
    ok["C"] += len(seen)
    vio["C_dup"] += dup
    vio["C_conflict"] += conflict
    # 静态：任何实体都不许作为子串出现在任何 (族, v3) 的轮3 文本里；轮2 也要无实体
    static_vio = 0
    for fam in FAMILIES:
        for v3 in fam["v3"]:
            t3 = fam["t3_user"].format(v3=v3)
            static_vio += sum(1 for ent in NAMES if ent in t3)
    for ent in NAMES:
        static_vio += sum(1 for t in T2_TOPICS + T2_SYS if ent in t)
    vio["A_static"] += static_vio
    print(f"[INV:{tag}] A 实体在轮1且轮3无实体：通过={ok['A']} 违例={vio['A']} | "
          f"静态词表级违例(A_static)={vio['A_static']}", flush=True)
    print(f"[INV:{tag}] B 目标含实体+轮3新信息：通过={ok['B']} 违例={vio['B']}", flush=True)
    print(f"[INV:{tag}] C ctx→target 是函数：ctx唯一={ok['C']} 重复={vio['C_dup']} "
          f"同ctx不同target={vio['C_conflict']}（违例）", flush=True)
    total_vio = vio["A"] + vio["B"] + vio["A_static"] + vio["C_dup"] + vio["C_conflict"]
    print(f"[INV:{tag}] 违例合计={total_vio} ⇒ "
          f"{'全部不变量成立' if total_vio == 0 else '★数据无效'}", flush=True)
    return ok, vio, total_vio


# ---------------- 指纹（口径与 stages/65::data_fp 逐字一致） ----------------
def data_fp(recs) -> str:
    """对每条 (len(p), p..., len(t), t...) 求 sha256[:16]。recs 需带 'p'/'t' 两个 id 序列。"""
    h = hashlib.sha256()
    for r in recs:
        for seq in (r["p"], r["t"]):
            h.update(struct.pack("<Q", len(seq)))
            if seq:
                h.update(struct.pack(f"<{len(seq)}i", *seq))
    return h.hexdigest()[:16]


# ---------------- 地板（P2）：只用输入可见信息 + train 统计 ----------------
def _em(gold: str, pred: str) -> int:
    return int((gold or "").strip() == (pred or "").strip())


def _t3_parse(t3: str):
    """只用轮3 文本 + 已知族模板 ⇒ (族 key, v3)；识别不出返回 (None, None)。

    ★ 该函数**不许**读样本的 ent / family 字段：它模拟「看得到模板与轮3 槽位」。
    """
    for fam in FAMILIES:
        head, tail = fam["t3_user"].split("{v3}")
        if t3.startswith(head) and (t3.endswith(tail) if tail else True):
            mid = t3[len(head): len(t3) - len(tail)] if tail else t3[len(head):]
            if mid in fam["v3"]:
                return fam["key"], mid
    return None, None


def _bucket_specs():
    return (
        ("⑤a轮3长度桶(5tok一桶)", lambda r: r.get("t3_tok", r["t3_chars"]) // 5),
        ("⑤b实体个数桶", lambda r: r["ent_count"]),
        ("⑤c轮数桶", lambda r: r["n_turns"]),
        ("⑤d实体位置桶(10字符一桶)", lambda r: r["ent_pos"] // 10),
    )


def _stats(train):
    st = {
        "maj_target": Counter(r["target"] for r in train).most_common(1)[0][0],
        "maj_ent": Counter(r["ent"] for r in train).most_common(1)[0][0],
        "bucket": {},
    }
    for name, keyfn in _bucket_specs():
        per = {}
        for r in train:
            per.setdefault(keyfn(r), []).append(r["target"])
        st["bucket"][name] = {b: Counter(v).most_common(1)[0][0] for b, v in per.items()}
    return st


def _rule_preds():
    def fill_only_t3(r, s):
        key, v3 = _t3_parse(r["t3"])
        if key is None:
            s["miss"] += 1
            return ""
        return FAM_BY_KEY[key]["t3_sys"].format(ent=s["maj_ent"], v3=v3)

    rules = [
        ("①复制轮3用户话", lambda r, s: r["t3"]),
        ("②复制轮1系统话", lambda r, s: r["t1_sys"]),
        ("③最常见回复(train多数类)", lambda r, s: s["maj_target"]),
        ("④模板填空只用轮3(实体位填train多数实体)", fill_only_t3),
    ]
    for name, keyfn in _bucket_specs():
        def mk(name=name, keyfn=keyfn):
            def f(r, s):
                return s["bucket"][name].get(keyfn(r), s["maj_target"])
            return f
        rules.append((name, mk()))
    return rules


def compute_floors(train, test, tag="test", show_train=True):
    """在 test 上按 P2 逐条算 EM；返回 (per-rule EM, max_naive 名, max_naive 值, 门槛)。

    规则 ①–④ 只用「输入可见信息 + train 统计」；⑤ 是计数类（按特征桶取 train 多数类）。
    """
    st = _stats(train)
    rules = _rule_preds()

    def eval_rules(rs):
        rows = {}
        for name, fn in rules:
            per = dict(st)              # 每条规则独立的 miss 计数；st['bucket'] 只读共享
            per["miss"] = 0
            hit = sum(_em(r["target"], fn(r, per)) for r in rs)
            rows[name] = (100.0 * hit / len(rs), per["miss"])
        return rows

    rows_test = eval_rules(test)
    out = {k: v[0] for k, v in rows_test.items()}
    for name, (em, miss) in rows_test.items():
        print(f"[FLOOR] {tag} {name} EM={em:.2f}% (n={len(test)})"
              + (f" | 识别不出族={miss}" if miss else ""), flush=True)
    mx_name = max(out, key=out.get)
    mx = out[mx_name]
    n = len(test)
    se2 = 2.0 * (mx / 100.0 * (1 - mx / 100.0) / n) ** 0.5 * 100.0
    thr = mx + se2
    print(f"[FLOOR] {tag} max_naive={mx:.2f}% 来自={mx_name} | 2SE={se2:.2f}pp ⇒ "
          f"P1 门槛=max_naive+2SE={thr:.2f}%", flush=True)
    if show_train:
        rows_train = eval_rules(train)
        tr = {k: v[0] for k, v in rows_train.items()}
        for name, (em, _miss) in rows_train.items():
            print(f"[FLOOR] train {name} EM={em:.2f}% (n={len(train)})", flush=True)
        tm_name = max(tr, key=tr.get)
        print(f"[FLOOR] train max_naive={tr[tm_name]:.2f}% 来自={tm_name} ⇒ "
              + ("★数据无效（朴素规则在 train 上≥90%）" if tr[tm_name] >= 90.0
                 else "有效（无朴素规则在 train 上≥90%）"), flush=True)
    return out, mx_name, mx, thr


# ---------------- 入口 ----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="S66 跨轮指代式对话数据生成器")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--n-test", type=int, default=N_TEST)
    a = ap.parse_args(argv)

    recs = gen_records(seed=a.seed, n_train=a.n_train, n_test=a.n_test)
    train, test = recs[:a.n_train], recs[a.n_train:]
    print(f"[SPLIT] 固定切分：先 shuffle 组合再切前 {a.n_train}+{a.n_test} 条；"
          f"组合空间={N_TEMPLATE_FAMILIES * len(NAMES) * 4 * len(T2_TOPICS)}", flush=True)

    ok, vio, total_vio = check_invariants(recs, tag="all")

    # ---- tokenizer：逐条编码 + 指纹 + 可加性 + 格式核对 ----
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK_PATH)
    V = tok.get_vocab_size()
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids        # noqa: E731
    dec = lambda ids: tok.decode([int(i) for i in ids], skip_special_tokens=False)  # noqa: E731
    PAD_ID = tok.token_to_id("<pad>")
    EOS_ID = tok.token_to_id("<eos>")
    assert enc("<resp>") == [140] and enc(EOS) == [EOS_ID], (enc("<resp>"), enc(EOS))

    from training.dialogue_stream import DialogueStream
    n_ctx_fmt_mismatch = n_additive_bad = n_rt_bad = 0
    for r in recs:
        r["p"] = enc(r["ctx"])
        r["t"] = enc(r["target"] + EOS)
        r["t3_tok"] = len(enc(r["t3"]))
        if dec(r["p"]) != r["ctx"] or dec(r["t"]) != r["target"] + EOS:
            n_rt_bad += 1
        if enc(r["ctx"] + r["target"] + EOS) != r["p"] + r["t"]:
            n_additive_bad += 1
        st = DialogueStream(enc, window=10 ** 9)
        st.append("用户", r["t1_user"]); st.commit(r["t1_sys"])
        st.append("用户", r["t2_user"]); st.commit(r["t2_sys"])
        st.append("用户", r["t3"])
        if st.prompt() != r["ctx"]:
            n_ctx_fmt_mismatch += 1
    assert max(len(r["p"]) for r in recs) + max(len(r["t"]) for r in recs) <= 512, "ctx+target 超 512"
    train, test = recs[:a.n_train], recs[a.n_train:]
    tr_fp, te_fp = data_fp(train), data_fp(test)
    print(f"[FMT] 与 DialogueStream.prompt() 逐字不一致={n_ctx_fmt_mismatch} | "
          f"enc 可加性破坏={n_additive_bad} | 编解码往返坏={n_rt_bad} | "
          f"V={V} PAD={PAD_ID} EOS={EOS_ID} | ctx tok 中位="
          f"{sorted(len(r['p']) for r in recs)[len(recs)//2]} 最大={max(len(r['p']) for r in recs)} "
          f"| target tok 中位={sorted(len(r['t']) for r in recs)[len(recs)//2]} "
          f"最大={max(len(r['t']) for r in recs)}", flush=True)

    # ---- 落盘（p/t 只在内存里用于指纹与长度统计，不写进 jsonl：67 会用同一 tokenizer 重算） ----
    for split, rs in (("train", train), ("test", test)):
        fp = os.path.join(LOG_DIR, f"66_dialog_{split}.jsonl")
        with open(fp, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps({k: v for k, v in r.items() if k not in ("p", "t")},
                                   ensure_ascii=False) + "\n")
        print(f"[OUT] {split} → {fp} ({len(rs)} 条)", flush=True)

    print(f"[D66] n_train={len(train)} n_test={len(test)} 模板数={N_TEMPLATE_FAMILIES} "
          f"实体词表={len(NAMES)} | train_fp={tr_fp} test_fp={te_fp}", flush=True)
    print(f"[TPL] 族模板={N_TEMPLATE_FAMILIES} 回复模板(族×新信息槽)={N_REPLY_TEMPLATES} "
          f"新信息槽合计={N_V3} | 族=" + " ".join(
              f"{f['key']}({len(f['v3'])}槽)" for f in FAMILIES), flush=True)
    print(f"[CONST] 轮2 填充语={len(T2_TOPICS)} 轮2 系统语={len(T2_SYS)} seed={a.seed} "
          f"| 唯一 ctx={len(set(r['ctx'] for r in recs))}/{len(recs)} | "
          f"turn3_has_ent True 计数={sum(r['turn3_has_ent'] for r in recs)}（必须 0）", flush=True)
    print(f"[ENTCNT] ctx 中实体个数分布="
          f"{dict(Counter(r['ent_count'] for r in recs))}", flush=True)

    # ---- 地板 ----
    compute_floors(train, test, tag="test", show_train=True)
    if total_vio:
        print(f"[D66-VERDICT] ★不变量违例={total_vio} ⇒ 数据无效，停。", flush=True)
        return 4
    print("[D66-VERDICT] 不变量全过；地板特征与指纹已落盘。", flush=True)
    print("[DONE] exit=0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
