#!/usr/bin/env python3
"""syllogism_card 数据生成器（程序生成、受控）+ 断言 + naive 电池口径。

产出 `data.json`：
  items: 逐题 {id, exit, type, sub, premises, gold_tags, label, attr, mention_bases, ...}
  stats: 逐型 × 逐出口的 n / 丢弃率 / 表面规则可解率
  surface: 两出口「无共享表面串」的断言实测（字符集交集、n-gram 交集）

用法：uv run python experiments/syllogism_card/gen_data.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent

# ---------------- 词表（两出口字符集必须不相交 —— 生成器 assert） ----------------
PT = "斐巴基栖梧松青白赤蓝绿紫黄银金铁铜瀚雪岚峰"
ST = "岛兰桐岭川石壁湾洲檀沙滩桥山海原田村"
PP = "麒麟蛟鲲玄朱毕饕貔狴睚狻椒鸱鹤鹫"
SP = "麟凤鹏武雀泽方餮貅犴眦猊图吻羽翎"

FUNC = {
    "tmpl": {"q_all": "所有", "q_some": "有的", "sub": "是", "disj": "不是",
             "pos": "成立", "neg": "不成立", "negent": "逆否", "sep": "。"},
    "para": {"q_all": "每个", "q_some": "某些", "sub": "属于", "sub_flip": "包含",
             "disj": "并非属于", "pos": "生效", "neg": "失效", "negent": "倒装", "sep": "，"},
}
#: 极小否定字集（纯字符串规则可用；不含任何关系方向信息）
NEG_CHARS = set("不非无未失")

TAGS = ["O", "ENT_S", "ENT_O", "REL_P", "REL_N"]
T_O, T_S, T_EO, T_RP, T_RN = 0, 1, 2, 3, 4

N_FORM, N_QUANT, N_ENT = 2, 2, 4
N_LABEL = 1 + N_FORM * N_QUANT * N_ENT * N_ENT   # 65
N_ATTR = 6 * 6 + 1                                # 37 (最后一位 = NONE)
REJECT, ATTR_NONE = 0, 36

ENT_LEN = 2   # 实体词恒为 2 字（接口约定）


def enc_label(form: int, quant: int, a: int, b: int) -> int:
    return 1 + form * (N_QUANT * N_ENT * N_ENT) + quant * (N_ENT * N_ENT) + a * N_ENT + b


def dec_label(x: int):
    if x == REJECT:
        return None
    v = x - 1
    form = v // (N_QUANT * N_ENT * N_ENT)
    v %= N_QUANT * N_ENT * N_ENT
    quant = v // (N_ENT * N_ENT)
    v %= N_ENT * N_ENT
    return form, quant, v // N_ENT, v % N_ENT


def base_of(run: str) -> str:
    """名称段 → 实体词：恒取最后 2 字（前缀 = 2 字量词 / 2 字逆否标记）。"""
    return run[-ENT_LEN:]


def neg_of(rel: str) -> bool:
    return bool(set(rel) & NEG_CHARS)


# ---------------- 渲染 ----------------
def render_premise(exit_: str, spec: dict) -> tuple[str, list[tuple[str, str]]]:
    """逻辑前提 → (表面文本, [(片段, 角色)] )，角色 ∈ S/O/RP/RN。

    角色按**表面顺序**：第一个名称段 = S（施事位），第二个 = O（受事位）——
    方向信息只由关系词承载，交给推理头（识别卡不猜方向）。
    """
    f = FUNC[exit_]
    k = spec["kind"]
    if k == "sub":
        q = f["q_all"] if spec["quant"] == 0 else f["q_some"]
        a, b = spec["a"], spec["b"]
        if exit_ == "tmpl":
            return q + a + f["sub"] + b, [(q + a, "S"), (f["sub"], "RP"), (b, "O")]
        if not spec["flip"]:
            return q + a + f["sub"] + b, [(q + a, "S"), (f["sub"], "RP"), (b, "O")]
        return q + b + f["sub_flip"] + a, [(q + b, "S"), (f["sub_flip"], "RP"), (a, "O")]
    if k == "disj":
        q = f["q_all"]
        a, b = spec["a"], spec["b"]
        if exit_ == "tmpl" or not spec["flip"]:
            return q + a + f["disj"] + b, [(q + a, "S"), (f["disj"], "RN"), (b, "O")]
        return q + b + f["disj"] + a, [(q + b, "S"), (f["disj"], "RN"), (a, "O")]
    if k == "sub_contra":
        # 逻辑意义 b ⊆ c；表面给逆否式 ¬c ⊆ ¬b
        q = f["q_all"]
        b, c = spec["b"], spec["c"]
        m = f["negent"]
        if exit_ == "tmpl" or not spec["flip"]:
            return q + m + c + f["sub"] + m + b, \
                [(q + m + c, "S"), (f["sub"], "RP"), (m + b, "O")]
        return q + m + b + f["sub_flip"] + m + c, \
            [(q + m + b, "S"), (f["sub_flip"], "RP"), (m + c, "O")]
    if k == "unary":
        q = f["q_all"]
        a = spec["a"]
        w = f["pos"] if spec["truth"] else f["neg"]
        return q + a + w, [(q + a, "S"), (w, "RN" if not spec["truth"] else "RP")]
    raise ValueError(k)


def tags_of(text: str, segs: list[tuple[str, str]]) -> list[int]:
    out: list[int] = []
    i = 0
    for s, role in segs:
        assert text[i:i + len(s)] == s, (text, segs, i)
        tag = {"S": T_S, "O": T_EO, "RP": T_RP, "RN": T_RN}[role]
        out += [tag] * len(s)
        i += len(s)
    assert i == len(text), (text, segs, i)
    return out


# ---------------- 结构抽取（识别卡输出 → 命题结构；gold 与预测共用） ----------------
def parse_tags(text: str, tags: list[int]) -> dict | None:
    """标签序列 → 命题结构。返回 None = 识别卡过不了结构门（fail-closed）。"""
    runs: list[dict] = []
    i, n = 0, len(text)
    while i < n:
        t = tags[i]
        if t in (T_S, T_EO):
            j = i
            while j < n and tags[j] == t:
                j += 1
            runs.append({"role": "S" if t == T_S else "O", "start": i, "end": j})
            i = j
        elif t in (T_RP, T_RN):
            j = i
            while j < n and tags[j] == t:
                j += 1
            runs.append({"role": "rel", "start": i, "end": j,
                         "neg": t == T_RN})
            i = j
        else:
            i += 1
    names = [r for r in runs if r["role"] != "rel"]
    rels = [r for r in runs if r["role"] == "rel"]
    if len(rels) != 1:
        return None
    order = sorted(runs, key=lambda r: r["start"])
    if order[0]["role"] == "rel":
        return None
    if len(names) == 2:
        kind = "binary"
    elif len(names) == 1:
        kind = "unary"
    else:
        return None
    # 名称段必须首尾相接「名-关系-名」或「名-关系」
    seq = [r["role"] for r in order]
    expect = ["S", "rel", "O"] if kind == "binary" else ["S", "rel"]
    if seq != expect:
        return None
    for r in names:
        r["text"] = text[r["start"]:r["end"]]
        r["base"] = base_of(r["text"])
    rel = rels[0]
    rel["text"] = text[rel["start"]:rel["end"]]
    return {"kind": kind, "names": names, "rel": rel}


# ---------------- 题型 ----------------
SUBS = ["chain", "v_neg", "v_quant", "v_contra", "affirm", "negaff", "undist", "distractor"]
TYPE_OF = {"chain": "chain", "v_neg": "variant", "v_quant": "variant",
           "v_contra": "variant", "affirm": "fallacy", "negaff": "fallacy",
           "undist": "fallacy", "distractor": "distractor"}


def build_logic(sub: str, e: list[str]) -> tuple[list[dict], tuple | None]:
    """→ (逻辑前提列表, gold(form,quant,a,b) | None)；e = [A,B,C,D] 语义实体（字符串）"""
    A, B, C, D = e
    if sub == "chain":
        return [{"kind": "sub", "a": A, "b": B, "quant": 0},
                {"kind": "sub", "a": B, "b": C, "quant": 0}], (0, 0, A, C)
    if sub == "v_neg":
        return [{"kind": "sub", "a": A, "b": B, "quant": 0},
                {"kind": "disj", "a": B, "b": C, "quant": 0}], (1, 0, A, C)
    if sub == "v_quant":
        return [{"kind": "sub", "a": A, "b": B, "quant": 1},
                {"kind": "sub", "a": B, "b": C, "quant": 0}], (0, 1, A, C)
    if sub == "v_contra":
        return [{"kind": "sub", "a": A, "b": B, "quant": 0},
                {"kind": "sub_contra", "b": B, "c": C, "quant": 0}], (0, 0, A, C)
    if sub == "affirm":
        return [{"kind": "sub", "a": A, "b": B, "quant": 0},
                {"kind": "unary", "a": B, "truth": True}], None
    if sub == "negaff":
        return [{"kind": "sub", "a": A, "b": B, "quant": 0},
                {"kind": "unary", "a": B, "truth": False}], None
    if sub == "undist":
        return [{"kind": "sub", "a": A, "b": B, "quant": 0},
                {"kind": "sub", "a": C, "b": B, "quant": 0}], None
    if sub == "distractor":
        return [{"kind": "sub", "a": A, "b": B, "quant": 0},
                {"kind": "sub", "a": C, "b": D, "quant": 0}], None
    raise ValueError(sub)


def make_item(rng: random.Random, exit_: str, sub: str, eid: int, lex: list[str]) -> dict | None:
    ents = rng.sample(lex, 4)
    logic, gold = build_logic(sub, ents)
    flip = [bool(rng.getrandbits(1)) for _ in logic] if exit_ == "para" else [False] * len(logic)
    for s, fl in zip(logic, flip):
        s["flip"] = fl
    surf = [render_premise(exit_, s) for s in logic]
    order = list(range(2))
    if exit_ == "para" and rng.getrandbits(1):
        rng.shuffle(order)
    premises = [surf[i][0] for i in order]
    segs = [surf[i][1] for i in order]
    tags = [tags_of(p, sg) for p, sg in zip(premises, segs)]

    structs = [parse_tags(p, t) for p, t in zip(premises, tags)]
    if any(s is None for s in structs):
        return None
    # mention 序列（跨前提按表面顺序）→ 实体 id（首次出现序）
    mentions: list[dict] = []
    for pi, st in enumerate(structs):
        for ni, nm in enumerate(st["names"]):
            mentions.append({"p": pi, "n": ni, "base": nm["base"],
                             "role": nm["role"], "start": nm["start"], "end": nm["end"]})
    ids: dict[str, int] = {}
    for m in mentions:
        if m["base"] not in ids:
            ids[m["base"]] = len(ids)
        m["eid"] = ids[m["base"]]
    if len(ids) > N_ENT:
        return None

    if gold is None:
        label, attr = REJECT, ATTR_NONE
    else:
        form, quant, xa, xb = gold
        if xa not in ids or xb not in ids:
            return None
        label = enc_label(form, quant, ids[xa], ids[xb])
        ia = next(m for m in mentions if m["base"] == xa)
        ib = next(m for m in mentions if m["base"] == xb)
        attr = mentions.index(ia) * 6 + mentions.index(ib)

    return {
        "id": eid, "exit": exit_, "type": TYPE_OF[sub], "sub": sub,
        "premises": premises, "gold_tags": tags,
        "label": label, "attr": attr,
        "mention_bases": [m["base"] for m in mentions],
        "n_mentions": len(mentions),
        "sem": ents,
    }


# ---------------- 断言：两出口无共享表面串 ----------------
def surface_audit(items: list[dict]) -> dict:
    tset = [w for it in items if it["exit"] == "tmpl" for w in it["premises"]]
    pset = [w for it in items if it["exit"] == "para" for w in it["premises"]]
    ct, cp = set("".join(tset)), set("".join(pset))
    def grams(ss, k):
        out = set()
        for s in ss:
            for i in range(len(s) - k + 1):
                out.add(s[i:i + k])
        return out
    res = {"char_t": len(ct), "char_p": len(cp), "char_common": sorted(ct & cp)}
    for k in (2, 3, 4, 6):
        res[f"gram{k}_common"] = len(grams(tset, k) & grams(pset, k))
    res["sent_common"] = len(set(tset) & set(pset))
    res["func_t"] = sorted(set("".join(v for v in FUNC["tmpl"].values())))
    res["func_p"] = sorted(set("".join(v for v in FUNC["para"].values())))
    res["func_common"] = sorted(set(res["func_t"]) & set(res["func_p"]))
    res["entity_common"] = sorted(set(PT + ST) & set(PP + SP))
    res["assert_ok"] = (res["char_common"] == [] and res["gram2_common"] == 0
                        and res["sent_common"] == 0 and res["func_common"] == []
                        and res["entity_common"] == [])
    return res


# ---------------- naive 电池（口径见 PREREG §4） ----------------
Q_LEX = {"所有": 0, "每个": 0, "有的": 1, "某些": 1}


def _split(p: str) -> tuple[str, str, str, str]:
    """纯字符串拆分：量词(前2字) + 主(下2字) + 关系(中) + 宾(后2字)。

    关系区为空 ⇒ 单前提。**不查任何关系词表**（这是「符号层可解」的定义）。
    """
    if len(p) < 3:
        return "", "", "", ""
    q = p[:2]
    body = p[2:]
    if len(body) < 5:                    # 单前提：主 + 2 字谓词
        return q, body, "", ""
    first, rel, last = body[:2], body[2:-2], body[-2:]
    if not rel:
        return q, first, "", ""
    return q, first, rel, last


def _idmap(premises: list[str]) -> dict:
    """实体 id = 按表面顺序首次出现（与卡的接口约定一致，规则自己从字符串提取）。"""
    ids: dict[str, int] = {}
    for p in premises:
        _, first, rel, last = _split(p)
        for nm in ([first, last] if rel else [first]):
            if nm and nm not in ids:
                ids[nm] = len(ids)
    return ids


def r_chain(premises: list[str], two_way: bool = False):
    """R-chain：模板出口表面协议（首段=主、末段=宾），无关系方向词表。"""
    if len(premises) != 2:
        return None
    sp = [_split(p) for p in premises]
    if any(not s[2] for s in sp):        # 任一为单前提 ⇒ 无链
        return None
    (_, a1, r1, b1), (_, a2, r2, b2) = sp
    if b1 == a2:
        x, y = a1, b2
    elif two_way and b2 == a1:
        x, y = a2, b1
    else:
        return None
    neg = neg_of(r1) or neg_of(r2)
    quant = min(Q_LEX.get(_[0], 0) for _ in sp)
    return (1 if neg else 0, quant, x, y)


def r_keyword(premises: list[str]):
    """R-keyword：词重叠匹配（无序）—— 两前提实体集恰差 1 才连链。"""
    if len(premises) != 2:
        return None
    sets = []
    for p in premises:
        _, first, rel, last = _split(p)
        if not rel:
            return None
        sets.append((p[:2], rel, {first, last}))
    if len(sets[0][2] & sets[1][2]) != 1:
        return None
    x = (sets[0][2] - sets[1][2]).pop()
    y = (sets[1][2] - sets[0][2]).pop()
    neg = neg_of(sets[0][1]) or neg_of(sets[1][1])
    quant = min(Q_LEX.get(sets[0][0], 0), Q_LEX.get(sets[1][0], 0))
    return (1 if neg else 0, quant, x, y)


#: 关系方向词表（两出口全量）—— **披露项 R-lexdir 专用，不计入 max_naive**
REL_DIR = {"是": (0, 1), "属于": (0, 1), "包含": (1, 0)}   # (lo, hi) 相对表面顺序
REL_NEG = {"不是", "并非属于"}


def r_lexdir(premises: list[str]):
    """带关系方向词表的链规则：含量词传播，**不含逆否转换**（非免费项，单列披露）。"""
    parsed = []
    for p in premises:
        _, a, rel, b = _split(p)
        if not rel:
            return None                    # 单前提
        q = p[:2]
        qv = Q_LEX.get(q, 0)
        if rel in REL_NEG:
            parsed.append(("disj", a, b, qv))
        elif rel in REL_DIR:
            i, j = REL_DIR[rel]
            lo, hi = (a, b) if i == 0 else (b, a)
            parsed.append(("sub", lo, hi, qv))
        else:
            return None                    # 关系不在词表 ⇒ 判不出
    if len(parsed) != 2:
        return None
    p1, p2 = parsed
    if p1[0] == "sub" and p2[0] == "sub":
        if p1[2] == p2[1]:
            return (0, min(p1[3], p2[3]), p1[1], p2[2])
        if p2[2] == p1[1]:
            return (0, min(p1[3], p2[3]), p2[1], p1[2])
        return None
    if p1[0] == "sub" and p2[0] == "disj":
        sub, dis = p1, p2
    elif p2[0] == "sub" and p1[0] == "disj":
        sub, dis = p2, p1
    else:
        return None
    if sub[2] == dis[1]:
        return (1, sub[3], sub[1], dis[2])
    if sub[2] == dis[2]:
        return (1, sub[3], sub[1], dis[1])
    return None


_RULES = {"R-chain": r_chain, "R-chain2": lambda p: r_chain(p, True),
          "R-keyword": r_keyword, "R-lexdir": r_lexdir}


def rule_label(kind: str, premises: list[str]) -> int:
    """规则名 → 65 类标签（实体 id 按表面首次出现序归一，与卡同一口径）。"""
    out = _RULES[kind](premises)
    if out is None:
        return REJECT
    form, quant, n1, n2 = out
    ids = _idmap(premises)
    if n1 not in ids or n2 not in ids:
        return REJECT
    return enc_label(form, quant, ids[n1], ids[n2])


def majority_rule(train_labels: Counter) -> callable:
    best = train_labels.most_common(1)[0][0]
    return lambda premises: best


def nslots_rule(tab: Counter, fallback: int) -> callable:
    """n_slots 查表：键 = (前提1字长, 前提2字长) → 训练集多数标签。"""
    return lambda premises: tab.get((len(premises[0]), len(premises[1])), fallback)


# ---------------- 主流程 ----------------
def main() -> int:
    rng = random.Random(20261007)
    n_train, n_test = 3000, 1200
    lex = {"tmpl": [a + b for a in PT for b in ST],
           "para": [a + b for a in PP for b in SP]}
    items: list[dict] = []
    stats: dict = {}
    dropped = Counter()
    seen: dict = Counter()
    plan = []
    for exit_, (nt, ne) in (("tmpl", (n_train, n_test)), ("para", (n_train, n_test))):
        total = nt + ne
        # 四型各 1/4；变体/谬误内部 3 子类均分
        quota = []
        for i in range(total):
            k = i % 4
            if k == 0:
                quota.append("chain")
            elif k == 1:
                quota.append(["v_neg", "v_quant", "v_contra"][(i // 4) % 3])
            elif k == 2:
                quota.append(["affirm", "negaff", "undist"][(i // 4) % 3])
            else:
                quota.append("distractor")
        rng.shuffle(quota)
        for sub in quota:
            plan.append((exit_, sub))

    eid = 0
    for exit_, sub in plan:
        for attempt in range(40):
            it = make_item(rng, exit_, sub, eid, lex[exit_])
            if it is None:
                dropped[(exit_, sub, "struct")] += 1
                continue
            key = (exit_, tuple(it["premises"]))
            if seen[key]:
                dropped[(exit_, sub, "dup")] += 1
                continue
            seen[key] += 1
            it["id"] = eid
            eid += 1
            items.append(it)
            break
        else:
            dropped[(exit_, sub, "exhaust")] += 1

    # 划分：每个出口前 n_train 条为 train，其余 test（生成顺序已打乱）
    per_exit: dict = {"tmpl": [], "para": []}
    for it in items:
        per_exit[it["exit"]].append(it)
    for exit_ in ("tmpl", "para"):
        for i, it in enumerate(per_exit[exit_]):
            it["split"] = "train" if i < n_train else "test"
    items = [it for it in items if it["split"] in ("train", "test")]

    # ---- 逐型统计 ----
    for exit_ in ("tmpl", "para"):
        for ty in ("chain", "variant", "fallacy", "distractor"):
            sel = [it for it in items if it["exit"] == exit_ and it["type"] == ty
                   and it["split"] == "test"]
            g = [it["label"] for it in sel]
            stats[f"{exit_}/{ty}"] = {
                "n": len(sel),
                "drop": sum(v for (e, s, r), v in dropped.items() if e == exit_ and
                            TYPE_OF[s] == ty),
                "drop_den": sum(1 for e, s in plan if e == exit_ and TYPE_OF[s] == ty),
                "reject_rate_gold": round(sum(1 for x in g if x == REJECT) / max(1, len(g)), 4),
            }
    # 表面规则可解率 = R-chain 在该格上的准确率（先算 gold，见 eval；此处占位）
    surface = surface_audit(items)

    out = {"items": items, "stats": stats, "surface": surface,
           "dropped": {f"{e}/{s}/{r}": v for (e, s, r), v in dropped.items()},
           "n_label": N_LABEL, "n_attr": N_ATTR,
           "tags": TAGS, "recipe": {"n_train": n_train, "n_test": n_test}}
    (HERE / "data.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[gen] items={len(items)} train="
          f"{sum(1 for i in items if i['split']=='train')} "
          f"test={sum(1 for i in items if i['split']=='test')}")
    print("[gen] dropped:", dict(dropped) or "无")
    print("[gen] surface assert_ok =", surface["assert_ok"],
          "char_common =", surface["char_common"],
          "gram_common =", [surface[f"gram{k}_common"] for k in (2, 3, 4, 6)],
          "sent_common =", surface["sent_common"])
    if not surface["assert_ok"]:
        print("[gen] 断言失败：两出口存在共享表面串", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
