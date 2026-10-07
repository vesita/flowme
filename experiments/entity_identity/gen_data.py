#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P12a 数据构造：四型实体同一性数据（程序生成、可核对）。

常量照抄 experiments/entity_identity/PREREG.md（mtime 2026-10-07 13:20:26 +0800，12923 B）：
  - §1   四型各 1/4；标签来源 = 生成器内部实体 id 分配表（独立于任何字面规则）
  - §1   场景：每条文本 2 个实体 / 3 个提及 / 2 个目标提及；同句异句两标签内各半；长度桶配平
  - §1   禁显式同一性关键词：简称/又称/同一人/另一所/两人/即/也就是
  - §1   提及定位：m1 = s1 首现；m2 = s2 首现且与 m1 不重叠；定位失败 assert 丢弃
  - §1.1 X 臂：train 均衡 4000（1000/型），test 同分布 held-out 1200（300/型）
统一骨架（三子句，两标签同构 ⇒ 句数/提及数/长度桶配平）：
  <名字1>是<属性1>，<名字2>是<属性2>，<名字3>当年发布了新公告。
  same_sent = 三子句同句（1 句）；diff_sent = 子句1 后断句（2 句）
不训练、不建卡、不改 src/。只写本目录。
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

# ---------- PREREG 照抄常量 ----------
SEED = 42
TRAIN_PER_TYPE = 1000          # PREREG §1.1 X 臂 4000（1000/型）
TEST_PER_TYPE = 300            # PREREG §1.1 held-out 1200（300/型）
BANNED = ["简称", "又称", "同一人", "另一所", "两人", "即", "也就是"]
N_ENTITIES = 2
N_MENTIONS = 3
TAIL = "当年发布了新公告"
EXPECTED_TYPE = {"A": "SAME", "B": "SAME", "C": "DIFF", "D": "DIFF"}
MAX_LEN = 60

# ---------- 实体池 ----------
SURNAMES = list("王李张刘陈杨赵黄周吴徐孙马朱胡郭何高林罗")
GIVEN_A = ["建国", "明远", "志强", "秀英", "淑珍", "建华", "立新", "俊杰", "慧敏", "学良",
           "佩兰", "伟民", "雅琴", "文博", "若彤", "学礼", "振华", "丹青", "国安", "兰芳"]
D_PAIRS = [("小明", "晓明"), ("雨辰", "雨晨"), ("泽楷", "泽凯"), ("佳怡", "佳琪"),
           ("浩宇", "浩禹"), ("子豪", "子毫"), ("梦瑶", "梦遥"), ("嘉和", "嘉禾"),
           ("若溪", "若熙"), ("雅静", "雅靖")]
C_NAMES = ["张伟", "李静", "王芳", "刘洋", "陈明", "杨帆", "赵敏", "黄磊",
           "周涛", "徐娟", "孙浩", "马丽", "朱琳", "胡兵", "郭鑫", "何敏"]
ORG_A_X = ["南京", "清华", "武汉", "复旦", "中山", "厦门", "兰州", "河北", "浙江", "江苏",
           "安徽", "福建", "广东", "云南", "贵州", "四川", "山东", "辽宁", "陕西", "湖南"]
ORG_D_X = ["北京", "南洋", "天河", "玉山", "文江", "明湖", "平海", "白石", "竹梅", "金江"]
ORG_C = ["海河大学", "玉泉学院", "青麓大学", "云岭中学", "白石大学",
         "金江大学", "明湖学院", "竹梅学院", "松江学院", "锦城学院"]
CITIES = ["南京", "杭州", "武汉", "成都", "西安", "长沙", "青岛", "苏州", "合肥", "福州",
          "昆明", "贵阳", "南昌", "无锡", "宁波", "温州", "徐州", "洛阳", "保定", "唐山"]
DEPTS = ["技术部", "市场部", "研发部", "财务部", "法务部", "运营部", "采购部", "质量部",
         "客服部", "行政部"]
ROLES = ["研究员", "工程师", "教师", "医生", "律师", "记者", "设计师", "会计师", "编辑", "主管"]
ORG_KINDS = ["公立院校", "民办院校", "公办分校", "独立学院", "市属高校", "省属高校"]


def org_full(x: str) -> str:
    return x + "大学"


def org_abbr(x: str) -> str:
    return x[0] + "大"


def ov(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def localize(text: str, s1: str, s2: str):
    """PREREG §1 提及定位：m1=s1 首现；m2=s2 首现且与 m1 不重叠。失败返回 None。"""
    i = text.find(s1)
    if i < 0:
        return None
    m1 = (i, i + len(s1))
    j = text.find(s2)
    while j >= 0:
        sp = (j, j + len(s2))
        if not ov(sp, m1):
            return m1, sp
        j = text.find(s2, j + 1)
    return None


def find_verb(text: str, m1, m2):
    """PREREG §1：第一个不与两提及重叠的「是」。"""
    i = text.find("是")
    while i >= 0:
        sp = (i, i + 1)
        if not ov(sp, m1) and not ov(sp, m2):
            return sp
        i = text.find("是", i + 1)
    return None


def draw_slot(typ, kind, rng, eid_prefix):
    """实体 id 分配表 + 槽位。slot 含 s1/s2/n1/n2/n3 与三子句属性。"""
    person = kind == "person"
    ents = []
    if typ == "A":
        if person:
            sn, gv = rng.choice(SURNAMES), rng.choice(GIVEN_A)
            full, abbr = sn + gv, gv
            dsn = rng.choice([x for x in SURNAMES if x != sn])
            distr = dsn + rng.choice(GIVEN_A)
            while abbr in distr:
                dsn = rng.choice([x for x in SURNAMES if x != sn])
                distr = dsn + rng.choice(GIVEN_A)
            e1, e2 = f"{eid_prefix}e1", f"{eid_prefix}e2"
            ents = [{"eid": e1, "surface": full, "abbr": abbr, "role": "target"},
                    {"eid": e2, "surface": distr, "role": "distractor"}]
            n1, n2, n3, s1, s2 = full, distr, abbr, full, abbr
        else:
            x = rng.choice(ORG_A_X)
            full, abbr = org_full(x), org_abbr(x)
            dx = rng.choice([y for y in ORG_A_X if y != x])
            distr = org_full(dx)
            while abbr in distr:
                dx = rng.choice([y for y in ORG_A_X if y != x])
                distr = org_full(dx)
            e1, e2 = f"{eid_prefix}e1", f"{eid_prefix}e2"
            ents = [{"eid": e1, "surface": full, "abbr": abbr, "role": "target"},
                    {"eid": e2, "surface": distr, "role": "distractor"}]
            n1, n2, n3, s1, s2 = full, distr, abbr, full, abbr
        roles = {"n1": "m1", "n2": "extra", "n3": "m2"}
        eid_m1, eid_m2 = e1, e1
    elif typ == "B":
        if person:
            sn, gv = rng.choice(SURNAMES), rng.choice(GIVEN_A)
            full = sn + gv
            dsn = rng.choice([x for x in SURNAMES if x != sn])
            distr = dsn + rng.choice(GIVEN_A)
            pron = rng.choice(["他", "她"])
            e1, e2 = f"{eid_prefix}e1", f"{eid_prefix}e2"
            ents = [{"eid": e1, "surface": full, "pron": pron, "role": "target"},
                    {"eid": e2, "surface": distr, "role": "distractor"}]
            n1, n2, n3, s1, s2 = full, distr, pron, full, pron
        else:
            x = rng.choice(ORG_A_X)
            full = org_full(x)
            distr = org_full(rng.choice([y for y in ORG_A_X if y != x]))
            e1, e2 = f"{eid_prefix}e1", f"{eid_prefix}e2"
            ents = [{"eid": e1, "surface": full, "pron": "该校", "role": "target"},
                    {"eid": e2, "surface": distr, "role": "distractor"}]
            n1, n2, n3, s1, s2 = full, distr, "该校", full, "该校"
        roles = {"n1": "m1", "n2": "extra", "n3": "m2"}
        eid_m1, eid_m2 = e1, e1
    elif typ == "C":
        if person:
            name = rng.choice(C_NAMES)
            e1, e2 = f"{eid_prefix}e1", f"{eid_prefix}e2"
            ents = [{"eid": e1, "surface": name, "role": "target"},
                    {"eid": e2, "surface": name, "role": "target"}]
        else:
            name = rng.choice(ORG_C)
            e1, e2 = f"{eid_prefix}e1", f"{eid_prefix}e2"
            ents = [{"eid": e1, "surface": name, "role": "target"},
                    {"eid": e2, "surface": name, "role": "target"}]
        n1 = n2 = n3 = s1 = s2 = name
        roles = {"n1": "m1", "n2": "m2", "n3": "extra"}
        eid_m1, eid_m2 = e1, e2
    elif typ == "D":
        if person:
            sn = rng.choice(SURNAMES)
            g1, g2 = rng.choice(D_PAIRS)
            n1, n2 = sn + g1, sn + g2
            e1, e2 = f"{eid_prefix}e1", f"{eid_prefix}e2"
            ents = [{"eid": e1, "surface": n1, "role": "target"},
                    {"eid": e2, "surface": n2, "role": "target"}]
            s1, s2, n3 = n1, n2, n2
        else:
            x = rng.choice(ORG_D_X)
            n1, n2 = org_full(x), x + "大學"
            e1, e2 = f"{eid_prefix}e1", f"{eid_prefix}e2"
            ents = [{"eid": e1, "surface": n1, "role": "target"},
                    {"eid": e2, "surface": n2, "role": "target"}]
            s1, s2, n3 = n1, n2, n2
        roles = {"n1": "m1", "n2": "m2", "n3": "extra"}
        eid_m1, eid_m2 = e1, e2
    else:
        raise ValueError(typ)

    # 子句属性（两标签同构）
    if person:
        c1, c2 = rng.sample(CITIES, 2)
        d1, d2 = rng.sample(DEPTS, 2)
        r1, r2 = rng.choice(ROLES), rng.choice(ROLES)
        slot = {"c1": c1, "d1": d1, "r1": r1, "c2": c2, "d2": d2, "r2": r2}
    else:
        c1, c2 = rng.sample(CITIES, 2)
        k1, k2 = rng.sample(ORG_KINDS, 2)
        slot = {"c1": c1, "k1": k1, "c2": c2, "k2": k2}
    slot.update({"person": person, "n1": n1, "n2": n2, "n3": n3,
                 "s1": s1, "s2": s2, "roles": roles})
    return slot, ents, eid_m1, eid_m2


def tpl(slot, variant):
    """统一三子句骨架；返回 [(part_kind, str), ...]。"""
    n1, n2, n3 = slot["n1"], slot["n2"], slot["n3"]
    if slot["person"]:
        c1 = n1 + "是" + slot["c1"] + slot["d1"] + "的" + slot["r1"]
        c2 = n2 + "是" + slot["c2"] + slot["d2"] + "的" + slot["r2"]
    else:
        c1 = n1 + "是" + slot["c1"] + "的" + slot["k1"]
        c2 = n2 + "是" + slot["c2"] + "的" + slot["k2"]
    c3 = n3 + TAIL
    parts = []
    key1, key2, key3 = slot["roles"]["n1"], slot["roles"]["n2"], slot["roles"]["n3"]
    if variant == "same_sent":
        parts = [(key1, n1), ("x", c1[len(n1):] + "，"),
                 (key2, n2), ("x", c2[len(n2):] + "，"),
                 (key3, n3), ("x", c3[len(n3):] + "。")]
    else:
        parts = [(key1, n1), ("x", c1[len(n1):] + "。"),
                 (key2, n2), ("x", c2[len(n2):] + "，"),
                 (key3, n3), ("x", c3[len(n3):] + "。")]
    return parts


def compose(parts):
    text = ""
    spans = {}
    for kind, s in parts:
        if kind in ("m1", "m2", "extra") and kind not in spans:
            spans[kind] = (len(text), len(text) + len(s))
        text += s
    return text, spans


def build_one(typ, i, variant, kind, rng, seen):
    slot, ents, eid_m1, eid_m2 = draw_slot(typ, kind, rng, f"{typ}{kind[0].upper()}{i}_")
    s1, s2 = slot["s1"], slot["s2"]
    parts = tpl(slot, variant)
    text, spans = compose(parts)
    if any(w in text for w in BANNED):
        return None, "禁词"
    loc = localize(text, s1, s2)
    if loc is None:
        return None, "定位失败"
    if loc != (spans["m1"], spans["m2"]):
        return None, "定位漂移"
    m1, m2 = loc
    verb = find_verb(text, m1, m2)
    if verb is None:
        return None, "无是"
    if len(ents) != N_ENTITIES or len(spans) != N_MENTIONS:
        return None, "实体提及数"
    if text.count("。") != (1 if variant == "same_sent" else 2):
        return None, "句数"
    if text in seen:
        return None, "文本重复"
    if len(text) > MAX_LEN:
        return None, "截断超长"
    label = "SAME" if eid_m1 == eid_m2 else "DIFF"
    if label != EXPECTED_TYPE[typ]:
        return None, "标签不符"
    item = {
        "id": f"{typ}{i:05d}", "type": typ, "label": label, "kind": kind,
        "variant": variant, "s1": s1, "s2": s2, "text": text,
        "m1_span": [m1[0], m1[1]], "m2_span": [m2[0], m2[1]],
        "verb_span": [verb[0], verb[1]], "extra_span": [spans["extra"][0], spans["extra"][1]],
        "entities": ents, "eid_m1": eid_m1, "eid_m2": eid_m2,
        "n_mentions": len(spans), "n_sent": text.count("。"), "n_entities": len(ents),
        "text_len": len(text),
    }
    return item, None


def gen_split(n_per_type, seed, seen_texts, tag):
    drops = Counter()
    kept = {t: [] for t in "ABCD"}
    attempts = Counter()
    for typ in "ABCD":
        rng = random.Random(f"{SEED}-{seed}-{tag}-{typ}")
        i, retries = 0, 0
        while len(kept[typ]) < n_per_type and attempts[typ] < n_per_type * 100:
            attempts[typ] += 1
            variant = "same_sent" if (i // 2) % 2 == 0 else "diff_sent"
            kind = "person" if i % 2 == 0 else "org"
            item, reason = build_one(typ, i, variant, kind, rng, seen_texts)
            if item is None:
                drops[(typ, reason)] += 1
                retries += 1
                if retries > 50:          # 同一 (variant,kind) 槽位重试耗尽 → 推进并计数
                    drops[(typ, "重试耗尽")] += 1
                    i += 1
                    retries = 0
                continue
            seen_texts.add(item["text"])
            kept[typ].append(item)
            i += 1
            retries = 0
    return kept, drops, attempts


def stats(items):
    n = len(items)
    ls = [it["text_len"] for it in items]
    mean = sum(ls) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in ls) / n if n else 0.0
    buckets = Counter(it["text_len"] // 8 for it in items)
    return {
        "n": n,
        "label": items[0]["label"] if n else None,
        "person": sum(1 for it in items if it["kind"] == "person"),
        "org": sum(1 for it in items if it["kind"] == "org"),
        "same_sent": sum(1 for it in items if it["variant"] == "same_sent"),
        "diff_sent": sum(1 for it in items if it["variant"] == "diff_sent"),
        "n_sent": dict(Counter(it["n_sent"] for it in items)),
        "text_len_mean": round(mean, 2),
        "text_len_sd": round(var ** 0.5, 2),
        "len_bucket(len//8)": {str(k): v for k, v in sorted(buckets.items())},
    }


def main():
    seen = set()
    train, tdrops, tatt = gen_split(TRAIN_PER_TYPE, SEED, seen, "train")
    test, vdrops, vatt = gen_split(TEST_PER_TYPE, SEED + 1, seen, "test")
    meta = {
        "prereg": {"path": "experiments/entity_identity/PREREG.md",
                   "mtime": "2026-10-07 13:20:26 +0800", "size": 12923,
                   "constants": {"SEED": SEED, "train_per_type": TRAIN_PER_TYPE,
                                 "test_per_type": TEST_PER_TYPE, "banned": BANNED}},
        "split": {}, "drops": {}, "attempts": {},
    }
    for split, kept, drops, att in (("train", train, tdrops, tatt), ("test", test, vdrops, vatt)):
        meta["split"][split] = {t: stats(kept[t]) for t in "ABCD"}
        meta["drops"][split] = {f"{t}:{r}": c for (t, r), c in sorted(drops.items())}
        meta["attempts"][split] = dict(att)
        with (DATA / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for t in "ABCD":
                for it in kept[t]:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (DATA / "gen_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    splits = {"train": train, "test": test}
    for split in ("train", "test"):
        kept = splits[split]
        tot = sum(meta["split"][split][t]["n"] for t in "ABCD")
        dr = sum(meta["drops"][split].values())
        att = sum(meta["attempts"][split].values())
        print(f"[{split}] 总 n={tot} 尝试={att} 丢弃={dr} 丢弃率={dr / max(att, 1):.4%}")
        for t in "ABCD":
            s = meta["split"][split][t]
            print(f"  {t} ({s['label']}): n={s['n']} person/org={s['person']}/{s['org']} "
                  f"同句/异句={s['same_sent']}/{s['diff_sent']} "
                  f"len={s['text_len_mean']}±{s['text_len_sd']} buckets={s['len_bucket(len//8)']}")
        for k, v in meta["drops"][split].items():
            print(f"  drop {k} = {v}")
        for lab in ("SAME", "DIFF"):
            sub = [it for t in "ABCD" for it in kept[t] if it["label"] == lab]
            bs = Counter(it["text_len"] // 8 for it in sub)
            print(f"  [{split}] {lab}: n={len(sub)} 长度桶={dict(sorted(bs.items()))} "
                  f"len={round(sum(i['text_len'] for i in sub)/len(sub),2)}")
    print("OK gen_data")


if __name__ == "__main__":
    main()
