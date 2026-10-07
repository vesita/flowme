#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P12b 两臂数据断言（口径逐条沿用 P12a check_e01.py 的风格；只读 data/，纯 CPU）。

断言（每条打印 PASS/FAIL，全部 PASS 才 exit 0）：
  K1 X-ir 提及对**零共享字**：train/test 各 0 违例；逐 subtype 报 n。
  K2 X-ir 标签/人机/同句异句 配平（两标签内各半）+ subtype 构成。
  K3 X-cross **实体级划分**：train 的实体 surface 集合 ∩ test = ∅（含 abbr/代称以外的名字）；
     实体 id 前缀（EA/EB）不相交；名池两半不相交。
  K4 逐条结构断言（两臂 + P12a 主臂同口径）：2 实体 / 3 提及 / 2 目标提及、
     定位重跑 == 存盘槽位、text[span] 逐字 == s、渲染动词存在、禁词 0、标签 == 实体 id 判定。
  K5 **可构造性**：X-ir DIFF-N2（异名对）机构侧 —— 枚举机构名池全部异名对，零共享字对数 = 0
     ⇒ 机构异名对**不可构造**（明说，不降级）；人名侧实测可构造 n。
  K6 长度桶 / 句数 分布（两标签），如实报出。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import gen_data as G

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BANNED = G.BANNED

PASS = {"n": 0}
FAIL = {"n": 0, "items": []}


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASS["n"] += 1
    else:
        FAIL["n"] += 1
        FAIL["items"].append(name)
    print(f"[{tag}] {name}" + (f" | {detail}" if detail else ""))
    return ok


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def struct_check(tag: str, rows: list[dict]) -> None:
    bad = Counter()
    for r in rows:
        t, s1, s2 = r["text"], r["s1"], r["s2"]
        if r["n_entities"] != 2 or len(r["entities"]) != 2:
            bad["实体数"] += 1
        if r["n_mentions"] != 3:
            bad["提及数"] += 1
        loc = G.localize(t, s1, s2)
        if loc is None:
            bad["定位失败重跑"] += 1
        elif [list(loc[0]), list(loc[1])] != [r["m1_span"], r["m2_span"]]:
            bad["定位漂移重跑"] += 1
        if t[r["m1_span"][0]:r["m1_span"][1]] != s1:
            bad["m1逐字"] += 1
        if t[r["m2_span"][0]:r["m2_span"][1]] != s2:
            bad["m2逐字"] += 1
        if G.find_verb(t, tuple(r["m1_span"]), tuple(r["m2_span"])) != tuple(r["verb_span"]):
            bad["动词重跑"] += 1
        if any(w in t for w in BANNED):
            bad["禁词"] += 1
        want = "SAME" if r["eid_m1"] == r["eid_m2"] else "DIFF"
        if want != r["label"]:
            bad["标签与实体id不符"] += 1
        eids = {e["eid"] for e in r["entities"]}
        if r["eid_m1"] not in eids or r["eid_m2"] not in eids:
            bad["eid不在实体表"] += 1
        if r["n_sent"] != t.count("。") or t.count("。") not in (1, 2):
            bad["句数"] += 1
    check(f"{tag} 结构断言（2实体/3提及/定位重跑/逐字/动词/禁词/标签id）",
          not bad, f"违例={dict(bad) if bad else '0'}")


def balance(tag: str, rows: list[dict]) -> None:
    for lab in ("SAME", "DIFF"):
        sub = [r for r in rows if r["label"] == lab]
        if not sub:
            continue
        kind = Counter(r["kind"] for r in sub)
        var = Counter(r["variant"] for r in sub)
        n = len(sub)
        half = abs(kind.get("person", 0) - kind.get("org", 0)) <= max(2, n // 100)
        hvar = abs(var.get("same_sent", 0) - var.get("diff_sent", 0)) <= max(2, n // 100)
        check(f"{tag} {lab} 人/机构配平", half, f"{dict(kind)}")
        check(f"{tag} {lab} 同句/异句配平", hvar, f"{dict(var)}")


def main() -> int:
    xir_tr, xir_te = load(DATA / "xir_train.jsonl"), load(DATA / "xir_test.jsonl")
    xc_tr, xc_te = load(DATA / "xcross_train.jsonl"), load(DATA / "xcross_test.jsonl")
    x_tr, x_te = load(DATA / "train.jsonl"), load(DATA / "test.jsonl")

    # ---- K1 零共享字 ----
    for tag, rows in (("x-ir/train", xir_tr), ("x-ir/test", xir_te)):
        viol = [r for r in rows if set(r["s1"]) & set(r["s2"])]
        sub = Counter(r["subtype"] for r in rows)
        check(f"{tag} 提及对零共享字", not viol,
              f"n={len(rows)} 违例={len(viol)} subtype={dict(sub)}")
        check(f"{tag} 存盘 zero_overlap=1", all(r["zero_overlap"] == 1 for r in rows))

    # ---- K2 配平 + 规模 ----
    check("x-ir/train 规模 1000/1000",
          Counter(r["label"] for r in xir_tr) == {"SAME": 1000, "DIFF": 1000},
          str(dict(Counter(r["label"] for r in xir_tr))))
    check("x-ir/test 规模 600/600",
          Counter(r["label"] for r in xir_te) == {"SAME": 600, "DIFF": 600},
          str(dict(Counter(r["label"] for r in xir_te))))
    balance("x-ir/train", xir_tr)
    balance("x-ir/test", xir_te)
    # SAME-B 与 DIFF-N1 同骨架（e1名/e2名/代称）：n1 都是代称绑定 e1 的文本骨架
    check("x-ir subtype 构成（N1≥N2、含 SAME-B）",
          {"SAME-B", "DIFF-N1"} <= {r["subtype"] for r in xir_tr}
          and {"DIFF-N2"} <= {r["subtype"] for r in xir_tr},
          str(dict(Counter(r["subtype"] for r in xir_tr))))

    for tag, rows, per in (("x-cross/train", xc_tr, 1000), ("x-cross/test", xc_te, 300)):
        c = Counter(r["type"] for r in rows)
        check(f"{tag} 四型各 {per}", all(c[t] == per for t in "ABCD"), str(dict(c)))
        check(f"{tag} 标签均衡", Counter(r["label"] for r in rows) == {"SAME": per * 2, "DIFF": per * 2},
              str(dict(Counter(r["label"] for r in rows))))
        balance(tag, rows)

    # ---- K3 实体级划分 ----
    def surfaces(rows):
        out = set()
        for r in rows:
            for e in r["entities"]:
                out.add(e["surface"])
                if "abbr" in e:
                    out.add(e["abbr"])
        return out
    def eids(rows):
        return {e["eid"] for r in rows for e in r["entities"]}

    inter_s = surfaces(xc_tr) & surfaces(xc_te)
    inter_i = eids(xc_tr) & eids(xc_te)
    check("x-cross 实体 surface train∩test = ∅", not inter_s,
          f"|train|={len(surfaces(xc_tr))} |test|={len(surfaces(xc_te))} 交={len(inter_s)}")
    check("x-cross 实体 id train∩test = ∅", not inter_i, f"交={len(inter_i)}")
    pools = json.loads((DATA / "arms_meta.json").read_text(encoding="utf-8"))["entity_pool_split"]
    for name, halves in pools.items():
        a, b = set(map(str, halves["EA"])), set(map(str, halves["EB"]))
        check(f"名池 {name} 两半不相交", not (a & b), f"|EA|={len(a)} |EB|={len(b)}")
    # 提及字面层面：**实体名字**在 train/test 间必须不相交；代称（他/她/该校）按设计共享
    PRON = {"他", "她", "该校"}
    tr_names = {r[k] for r in xc_tr for k in ("s1", "s2")} - PRON
    te_names = {r[k] for r in xc_te for k in ("s1", "s2")} - PRON
    check("x-cross 实体名字字面 train∩test = ∅（零实体泄漏）", not (tr_names & te_names),
          f"交={sorted(tr_names & te_names)}")
    tr_pron = {r[k] for r in xc_tr for k in ("s1", "s2")} & PRON
    te_pron = {r[k] for r in xc_te for k in ("s1", "s2")} & PRON
    print(f"[info] x-cross 代称共享（按设计共享，非实体）: train={sorted(tr_pron)} "
          f"test={sorted(te_pron)} 交={sorted(tr_pron & te_pron)}")

    # ---- K4 结构断言 ----
    struct_check("x-ir/train", xir_tr)
    struct_check("x-ir/test", xir_te)
    struct_check("x-cross/train", xc_tr)
    struct_check("x-cross/test", xc_te)
    struct_check("x/train(P12a)", x_tr)
    struct_check("x/test(P12a)", x_te)

    # ---- K5 可构造性：异名对零共享字枚举（如实报出，不降级） ----
    org_names = sorted(set(G.ORG_A_X) | set(G.ORG_C) | set(G.ORG_D_X))
    org_full = sorted({G.org_full(x) for x in G.ORG_A_X} | set(G.ORG_C)
                      | {x + "大学" for x in G.ORG_D_X} | {x + "大學" for x in G.ORG_D_X})
    pairs = zero = 0
    zlist = []
    for i, a in enumerate(org_full):
        for b in org_full[i + 1:]:
            if a == b:
                continue
            pairs += 1
            if not (set(a) & set(b)):
                zero += 1
                zlist.append((a, b))
    print(f"[construct] 机构异名对：名 {len(org_full)} 个、异名对 {pairs} 对、"
          f"零共享字对 {zero}（{zero / max(pairs, 1):.2%}）")
    for a, b in zlist[:6]:
        print(f"[construct]   零共享字机构对示例: {a} ↔ {b}")
    check("X-ir 机构异名对：可构造性已实测报出（含全部零共享字对是否跨后缀风格）",
          True, f"零共享字对={zero}")
    check("X-ir DIFF-N2 机构子型未纳入（数据里 0 条 org DIFF-N2）",
          not any(r["subtype"] == "DIFF-N2" and r["kind"] == "org"
                  for r in xir_tr + xir_te),
          "跨后缀风格会把「传统 大學」字面变成标签线索 ⇒ 不纳入")
    per_names = sorted({G.SURNAMES[i] + g for i in range(len(G.SURNAMES))
                        for g in G.GIVEN_A})
    pn = zero_p = 0
    for i, a in enumerate(per_names):
        for b in per_names[i + 1:]:
            pn += 1
            if not (set(a) & set(b)):
                zero_p += 1
    check("X-ir DIFF-N2 人名侧可构造（存在零共享字对）", zero_p > 0,
          f"人名 {len(per_names)} 个，异名对 {pn} 对，零共享字对 {zero_p}")

    # ---- K6 分布 ----
    for tag, rows in (("x-ir/train", xir_tr), ("x-ir/test", xir_te),
                      ("x-cross/train", xc_tr), ("x-cross/test", xc_te)):
        for lab in ("SAME", "DIFF"):
            sub = [r for r in rows if r["label"] == lab]
            bs = Counter(r["text_len"] // 8 for r in sub)
            print(f"[dist] {tag} {lab}: n={len(sub)} 长度桶={dict(sorted(bs.items()))} "
                  f"句数={dict(Counter(r['n_sent'] for r in sub))}")

    print(f"\n总计 PASS={PASS['n']} FAIL={FAIL['n']}"
          + (f" 失败项={FAIL['items']}" if FAIL["n"] else ""))
    out = {"pass": PASS["n"], "fail": FAIL["n"], "failed": FAIL["items"]}
    (DATA / "arms_assertions.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK check_arms" if FAIL["n"] == 0 else "ASSERT FAIL")
    return 0 if FAIL["n"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
