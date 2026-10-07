#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P12b 两臂数据构造：X-ir（提及对零共享字）+ X-cross（实体级划分·未训实体）。

常量照抄 experiments/entity_identity/PREREG.md（mtime 2026-10-07 13:20:26 +0800，12923 B）：
  §1.1 X-ir   训练 2000（SAME/DIFF 各 1000）、held-out 1200（SAME/DIFF 各 600）。
              正例 = B 型（全名↔代称，零共享字）；负例 = (i) 异名对（不同实体）
              (ii) 代称错配（代称绑定 e1，查询对是 代称↔e2 的名字 ⇒ 不同）。
  §1.1 X-cross 训练 4000（1000/型）、held-out 1200（300/型）；**实体级划分**（EA 训 / EB 评），
              不许 fs[:N] 式切分 ⇒ 实体名池按 seed42 洗牌对半分。
  §1   每条 2 实体 / 3 提及 / 2 目标提及；同句异句两标签各半；禁词；提及定位 m1/m2 规则；
       定位失败 / 无「是」/ 句数 / 截断超长 / 文本重复 ⇒ assert 丢弃并计数（口径同 P12a）。
复用 P12a 生成器 gen_data.py 的 tpl / compose / localize / find_verb / draw_slot / stats，
断言口径逐条沿用；不改 gen_data.py、不改 src/。
只写本目录（data/、logs/）。
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import gen_data as G

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

SEED = 42
BANNED = G.BANNED
MAX_LEN = G.MAX_LEN
N_ENTITIES, N_MENTIONS = 2, 3

# X-ir 分组规模（PREREG §1.1）
IR_TRAIN = {"SAME": 1000, "DIFF": 1000}
IR_TEST = {"SAME": 600, "DIFF": 600}
# X-cross 分组规模（PREREG §1.1）
XC_TRAIN_PER_TYPE = 1000
XC_TEST_PER_TYPE = 300


# ---------- 属性槽位（两标签同构，同 P12a） ----------
def attrs(person: bool, rng: random.Random) -> dict:
    if person:
        c1, c2 = rng.sample(G.CITIES, 2)
        d1, d2 = rng.sample(G.DEPTS, 2)
        return {"c1": c1, "d1": d1, "r1": rng.choice(G.ROLES),
                "c2": c2, "d2": d2, "r2": rng.choice(G.ROLES)}
    c1, c2 = rng.sample(G.CITIES, 2)
    k1, k2 = rng.sample(G.ORG_KINDS, 2)
    return {"c1": c1, "k1": k1, "c2": c2, "k2": k2}


def _slot(person: bool, n1, n2, n3, s1, s2, roles, eid1, eid2, rng) -> dict:
    d = {"person": person, "n1": n1, "n2": n2, "n3": n3, "s1": s1, "s2": s2,
         "roles": roles, "eid1": eid1, "eid2": eid2}
    d.update(attrs(person, rng))
    return d


# ---------- X-ir 三个构建器 ----------
def ir_same(kind: str, rng, prefix):
    """SAME = B 型：全名 ↔ 代称（零共享字），代称绑定 e1，查询对 (e1 名, 代称)。"""
    slot, ents, m1, m2 = G.draw_slot("B", kind, rng, prefix)
    return slot, ents, m1, m2, "SAME-B"


def ir_diff_n1(kind: str, rng, prefix):
    """DIFF (ii) 代称错配：文本骨架与 SAME-B 完全同构（e1 名 / e2 名 / 代称→e1），
    只有查询对不同：(代称, e2 的名字) ⇒ 实体 id 不等 = DIFF。零共享字（代称 vs 名字）。"""
    person = kind == "person"
    e1, e2 = f"{prefix}e1", f"{prefix}e2"
    if person:
        for _ in range(200):
            full1 = rng.choice(G.SURNAMES) + rng.choice(G.GIVEN_A)
            full2 = rng.choice(G.SURNAMES) + rng.choice(G.GIVEN_A)
            if full1 != full2:
                break
        else:
            return None, None, None, None, "实体名重复"
        pron = rng.choice(["他", "她"])
        ents = [{"eid": e1, "surface": full1, "pron": pron, "role": "target"},
                {"eid": e2, "surface": full2, "role": "distractor"}]
        slot = _slot(True, full1, full2, pron, pron, full2,
                     {"n1": "extra", "n2": "m2", "n3": "m1"}, e1, e2, rng)
    else:
        x1 = rng.choice(G.ORG_A_X)
        x2 = rng.choice([y for y in G.ORG_A_X if y != x1])
        full1, full2 = G.org_full(x1), G.org_full(x2)
        ents = [{"eid": e1, "surface": full1, "pron": "该校", "role": "target"},
                {"eid": e2, "surface": full2, "role": "distractor"}]
        slot = _slot(False, full1, full2, "该校", "该校", full2,
                     {"n1": "extra", "n2": "m2", "n3": "m1"}, e1, e2, rng)
    return slot, ents, e1, e2, "DIFF-N1"


def ir_diff_n2(kind: str, rng, prefix):
    """DIFF (i) 异名对：两个不同实体的名字，**零共享字**；额外提及 = e1 名复现。
    实测可构造性（check_arms K5 枚举）：人名侧 70300/79800 对零共享字；
    机构侧仅 55/780 对零共享字且**全部**是「学院/中学 ↔ 大學」跨后缀风格 ⇒ 后缀风格与
    标签相关（传统「大學」只出现在该子型）会引入字面线索 ⇒ **不纳入**，如实报出（不降级成假的不可构造）。"""
    if kind != "person":
        return None, None, None, None, "机构不可构造"
    e1, e2 = f"{prefix}e1", f"{prefix}e2"
    for _ in range(200):
        full1 = rng.choice(G.SURNAMES) + rng.choice(G.GIVEN_A)
        full2 = rng.choice(G.SURNAMES) + rng.choice(G.GIVEN_A)
        if full1 != full2 and not (set(full1) & set(full2)):
            break
    else:
        return None, None, None, None, "零重叠取样失败"
    ents = [{"eid": e1, "surface": full1, "role": "target"},
            {"eid": e2, "surface": full2, "role": "target"}]
    slot = _slot(True, full1, full2, full1, full1, full2,
                 {"n1": "m1", "n2": "m2", "n3": "extra"}, e1, e2, rng)
    return slot, ents, e1, e2, "DIFF-N2"


# ---------- 通用构建（断言口径 = P12a gen_data.build_one） ----------
def build_one(typ, i, variant, kind, rng, prefix, arm, builder, seen: set | None = None):
    slot, ents, eid_m1, eid_m2, subtype = builder(kind, rng, prefix)
    if slot is None:
        return None, subtype
    s1, s2 = slot["s1"], slot["s2"]
    # 零共享字只对 X-ir 臂是硬约束（PREREG §1.1）；A/C/D 型本就要求字面重叠/相同
    if arm == "x-ir" and (set(s1) & set(s2)):
        return None, "非零重叠"
    parts = G.tpl(slot, variant)
    text, spans = G.compose(parts)
    if any(w in text for w in BANNED):
        return None, "禁词"
    loc = G.localize(text, s1, s2)
    if loc is None:
        return None, "定位失败"
    if loc != (spans["m1"], spans["m2"]):
        return None, "定位漂移"
    m1, m2 = loc
    verb = G.find_verb(text, m1, m2)
    if verb is None:
        return None, "无是"
    if len(ents) != N_ENTITIES or len(spans) != N_MENTIONS:
        return None, "实体提及数"
    if text.count("。") != (1 if variant == "same_sent" else 2):
        return None, "句数"
    if seen is not None and text in seen:
        return None, "文本重复"
    if len(text) > MAX_LEN:
        return None, "截断超长"
    label = "SAME" if eid_m1 == eid_m2 else "DIFF"
    item = {
        "id": f"{arm}-{typ}-{i:05d}", "arm": arm, "type": typ, "subtype": subtype,
        "label": label, "kind": kind, "variant": variant,
        "s1": s1, "s2": s2, "text": text,
        "m1_span": [m1[0], m1[1]], "m2_span": [m2[0], m2[1]],
        "verb_span": [verb[0], verb[1]],
        "extra_span": [spans["extra"][0], spans["extra"][1]],
        "entities": ents, "eid_m1": eid_m1, "eid_m2": eid_m2,
        "n_mentions": len(spans), "n_sent": text.count("。"),
        "n_entities": len(ents), "text_len": len(text),
        "zero_overlap": int(not (set(s1) & set(s2))),
    }
    if arm == "x-ir" and item["zero_overlap"] != 1:
        return None, "非零重叠"
    return item, None


# ---------- X-ir ----------
def gen_ir(split: str, n_by_label: dict, seed: int, seen: set):
    """SAME 全部 = B 型（person/org 各半）；DIFF = person 按 N1/N2 对半、org 全 N1
    ⇒ DIFF 内 person/org 仍各半（kind 不构成标签线索）。"""
    drops, attempts = Counter(), Counter()
    kept = {"SAME": [], "DIFF": []}
    for label in ("SAME", "DIFF"):
        n_target = n_by_label[label]
        i = 0
        while len(kept[label]) < n_target and attempts[label] < n_target * 100:
            attempts[label] += 1
            variant = "same_sent" if (i // 2) % 2 == 0 else "diff_sent"
            kind = "person" if i % 2 == 0 else "org"
            prefix = f"ir{label}{kind[0]}{i}_"
            if label == "SAME":
                builder = ir_same
            elif kind == "person":
                builder = ir_diff_n1 if (i % 4 == 0) else ir_diff_n2
            else:
                builder = ir_diff_n1
            item, reason = build_one(
                label, i, variant, kind,
                random.Random(f"{SEED}-{seed}-{split}-{label}-{i}"),
                prefix, "x-ir", builder, seen=seen)
            if item is None:
                drops[(label, kind, reason)] += 1
                i += 1
                continue
            if seen is not None:
                seen.add(item["text"])
            kept[label].append(item)
            i += 1
    return kept, drops, attempts


# ---------- X-cross（实体级划分） ----------
def split_pools() -> dict:
    """实体名池按 seed42 洗牌对半分：EA（训练）/ EB（评测）。属性池不动（上下文配平）。

    预处理：ORG_C 里与 ORG_A_X/ORG_D_X **派生名**同字的条目（如「白石大学」= D 池白石+大学）
    会跨半撞车 ⇒ 先剔除，保证两半**派生 surface 不相交**（check_arms K3 实测断言）。
    """
    derived = {G.org_full(x) for x in G.ORG_A_X} | {x + "大学" for x in G.ORG_D_X} \
        | {x + "大學" for x in G.ORG_D_X} | {x[0] + "大" for x in G.ORG_A_X}
    G.ORG_C = [n for n in G.ORG_C if n not in derived]
    rng = random.Random(42)
    out = {}
    for name in ("SURNAMES", "GIVEN_A", "C_NAMES", "D_PAIRS", "ORG_A_X", "ORG_D_X", "ORG_C"):
        pool = list(getattr(G, name))
        rng.shuffle(pool)
        h = (len(pool) + 1) // 2
        out[name] = {"EA": pool[:h], "EB": pool[h:]}
    return out


def apply_pools(which: str, pools: dict) -> None:
    for name, halves in pools.items():
        setattr(G, name, halves[which])


def gen_cross(split: str, n_per_type: int, seed: int, seen: set, which: str, pools: dict):
    apply_pools(which, pools)
    drops, attempts = Counter(), Counter()
    kept = {t: [] for t in "ABCD"}
    for typ in "ABCD":
        rng = random.Random(f"{SEED}-{seed}-{which}-{split}-{typ}")
        i = 0
        while len(kept[typ]) < n_per_type and attempts[typ] < n_per_type * 100:
            attempts[typ] += 1
            variant = "same_sent" if (i // 2) % 2 == 0 else "diff_sent"
            kind = "person" if i % 2 == 0 else "org"
            prefix = f"xc{which}{typ}{kind[0]}{i}_"
            def _builder(k, r, p, _t=typ):
                slot, ents, m1, m2 = G.draw_slot(_t, k, r, p)
                return slot, ents, m1, m2, _t
            item, reason = build_one(
                typ, i, variant, kind, rng, prefix, "x-cross", _builder, seen=seen)
            if item is None:
                drops[(typ, kind, reason)] += 1
                i += 1
                continue
            if seen is not None:
                seen.add(item["text"])
            kept[typ].append(item)
            i += 1
    return kept, drops, attempts


def write_split(path: Path, groups) -> None:
    with path.open("w", encoding="utf-8") as f:
        for items in groups:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")


def summarize(name: str, kept, drops, attempts) -> dict:
    flat = [it for v in kept.values() for it in v]
    att, dr = sum(attempts.values()), sum(drops.values())
    sub = Counter(it["subtype"] for it in flat)
    lab = Counter(it["label"] for it in flat)
    kind = Counter(it["kind"] for it in flat)
    var = Counter(it["variant"] for it in flat)
    print(f"[{name}] n={len(flat)} 尝试={att} 丢弃={dr} 丢弃率={dr / max(att, 1):.4%}")
    print(f"  label={dict(lab)} kind={dict(kind)} variant={dict(var)} subtype={dict(sub)}")
    for k, v in sorted(drops.items()):
        print(f"  drop {k} = {v}")
    for lab_name in ("SAME", "DIFF"):
        sub_items = [it for it in flat if it["label"] == lab_name]
        if not sub_items:
            continue
        bs = Counter(it["text_len"] // 8 for it in sub_items)
        print(f"  {lab_name}: n={len(sub_items)} 长度桶={dict(sorted(bs.items()))} "
              f"len={round(sum(i['text_len'] for i in sub_items) / len(sub_items), 2)}")
    return {"n": len(flat), "attempts": att, "drops": dr,
            "drop_rate": dr / max(att, 1),
            "drops_detail": {str(k): v for k, v in sorted(drops.items())},
            "label": dict(lab), "kind": dict(kind), "variant": dict(var),
            "subtype": dict(sub)}


def main():
    seen: set = set()
    meta = {"prereg": {"path": "experiments/entity_identity/PREREG.md",
                       "mtime": "2026-10-07 13:20:26 +0800", "size": 12923},
            "arms": {}}

    for split, n in (("train", IR_TRAIN), ("test", IR_TEST)):
        seed = SEED if split == "train" else SEED + 1
        kept, drops, att = gen_ir(split, n, seed, seen)
        write_split(DATA / f"xir_{split}.jsonl", [kept["SAME"], kept["DIFF"]])
        meta["arms"][f"x-ir/{split}"] = summarize(f"x-ir/{split}", kept, drops, att)

    pools = split_pools()
    meta["entity_pool_split"] = {k: {"EA": v["EA"], "EB": v["EB"]} for k, v in pools.items()}
    snap = {n: list(getattr(G, n)) for n in pools}
    try:
        for split, n, which in (("train", XC_TRAIN_PER_TYPE, "EA"),
                                ("test", XC_TEST_PER_TYPE, "EB")):
            seed = SEED if split == "train" else SEED + 1
            kept, drops, att = gen_cross(split, n, seed, seen, which, pools)
            write_split(DATA / f"xcross_{split}.jsonl", [kept[t] for t in "ABCD"])
            meta["arms"][f"x-cross/{split}"] = summarize(f"x-cross/{split}", kept, drops, att)
    finally:
        for n_, v in snap.items():
            setattr(G, n_, v)

    (DATA / "arms_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK gen_arms")


if __name__ == "__main__":
    main()
