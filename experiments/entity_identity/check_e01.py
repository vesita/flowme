#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E0-1：语句级断言（只读 data/*.jsonl，全部必过，不过则退出码 1）。

判据常量照抄 PREREG.md：
  - §1 四型各 1/4；标签 = 实体 id 是否相等（来源独立于字面规则）
  - §1 关键：正例字面不同、负例字面相同或极近 ⇒「字面相同/不同」都不是答案
  - §1 A 型距离配平：人 全名↔去姓 编辑距离 1；机构 X大学↔X大 距离 2
  - §1 B 型零共享字；C 型距离 0；D 型距离 1（近名/简繁）
  - §1 场景：每条 2 实体 / 3 提及 / 2 目标提及；同句异句两标签内各半
  - §1 禁关键词；§1 提及定位取回构造槽位
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BANNED = ["简称", "又称", "同一人", "另一所", "两人", "即", "也就是"]


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load(split):
    p = DATA / f"{split}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    results = []
    ok_all = True

    def A(name, cond, detail=""):
        nonlocal ok_all
        ok = bool(cond)
        ok_all &= ok
        results.append({"assert": name, "pass": ok, "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    train, test = load("train"), load("test")
    both = train + test

    # A6 四型各 1/4
    for split, items, want in (("train", train, 1000), ("test", test, 300)):
        c = Counter(it["type"] for it in items)
        A(f"A6-{split}-四型各1/4", all(c[t] == want for t in "ABCD") and len(items) == 4 * want,
          f"n={dict(sorted(c.items()))} total={len(items)}")

    # A1 正例字面不同
    pos = [it for it in both if it["label"] == "SAME"]
    A("A1-正例字面不同", all(it["s1"] != it["s2"] for it in pos),
      f"SAME n={len(pos)} s1==s2 的条数={sum(1 for it in pos if it['s1'] == it['s2'])}")

    # A2 负·同形：字面相同（距离 0）
    c_items = [it for it in both if it["type"] == "C"]
    A("A2-C型字面相同(距离0)", len(c_items) > 0 and all(it["s1"] == it["s2"] and lev(it["s1"], it["s2"]) == 0
                                                      for it in c_items),
      f"C n={len(c_items)} 距离集合={sorted(set(lev(it['s1'], it['s2']) for it in c_items))}")

    # A3 负·相似：字面不同且距离 ≤ 1
    d_items = [it for it in both if it["type"] == "D"]
    A("A3-D型字面不同且距离≤1", len(d_items) > 0 and all(
        it["s1"] != it["s2"] and lev(it["s1"], it["s2"]) <= 1 for it in d_items),
      f"D n={len(d_items)} 距离集合={sorted(set(lev(it['s1'], it['s2']) for it in d_items))}")

    # A4 PREREG 距离配平：A 人 = 1，A 机构 = 2
    a_p = [it for it in both if it["type"] == "A" and it["kind"] == "person"]
    a_o = [it for it in both if it["type"] == "A" and it["kind"] == "org"]
    A("A4-A人距离=1", all(lev(it["s1"], it["s2"]) == 1 for it in a_p),
      f"n={len(a_p)} 距离={sorted(set(lev(it['s1'], it['s2']) for it in a_p))}")
    A("A4-A机构距离=2", all(lev(it["s1"], it["s2"]) == 2 for it in a_o),
      f"n={len(a_o)} 距离={sorted(set(lev(it['s1'], it['s2']) for it in a_o))}")

    # A5 B 型零共享字
    b_items = [it for it in both if it["type"] == "B"]
    A("A5-B型零共享字", len(b_items) > 0 and all(
        not (set(it["s1"]) & set(it["s2"])) for it in b_items),
      f"B n={len(b_items)} 有共享字的条数={sum(1 for it in b_items if set(it['s1']) & set(it['s2']))}")

    # A7 场景：2 实体 / 3 提及 / 2 目标提及（2 个目标提及由 s1、s2 各一保证）
    A("A7-每条2实体3提及", all(it["n_entities"] == 2 and it["n_mentions"] == 3 for it in both),
      f"实体数分布={dict(Counter(it['n_entities'] for it in both))} "
      f"提及数分布={dict(Counter(it['n_mentions'] for it in both))}")

    # A8 同句/异句 两标签内各半
    for lab in ("SAME", "DIFF"):
        sub = [it for it in both if it["label"] == lab]
        ss = sum(1 for it in sub if it["variant"] == "same_sent")
        A(f"A8-{lab}-同句异句各半", ss * 2 == len(sub),
          f"同句={ss} 异句={len(sub) - ss} n={len(sub)}")
    # 句数分布跨标签配平
    sc = {lab: dict(Counter(it["n_sent"] for it in both if it["label"] == lab)) for lab in ("SAME", "DIFF")}
    A("A8-句数分布跨标签一致", sc["SAME"] == sc["DIFF"], f"{sc}")

    # A9 禁词
    A("A9-禁词0出现", all(not any(w in it["text"] for w in BANNED) for it in both),
      f"违例={sum(1 for it in both if any(w in it['text'] for w in BANNED))}")

    # A10 提及定位 == 构造槽位 + 有不重叠的「是」+ 标签 = 实体 id 相等
    from gen_data import localize          # 复用 PREREG §1 的定位规则本身
    bad_loc = bad_verb = bad_label = 0
    for it in both:
        t = it["text"]
        m1, m2 = tuple(it["m1_span"]), tuple(it["m2_span"])
        if localize(t, it["s1"], it["s2"]) != (m1, m2):   # 规则重跑 != 存盘槽位
            bad_loc += 1
            continue
        if m1[0] < m2[1] and m2[0] < m1[1]:   # 区间重叠 ⇒ 违例
            bad_loc += 1
            continue
        if t[m1[0]:m1[1]] != it["s1"] or t[m2[0]:m2[1]] != it["s2"]:
            bad_loc += 1
        v = tuple(it["verb_span"])
        if t[v[0]:v[1]] != "是" or (v[0] < m1[1] and m1[0] < v[1]) or (v[0] < m2[1] and m2[0] < v[1]):
            bad_verb += 1
        if ("SAME" if it["eid_m1"] == it["eid_m2"] else "DIFF") != it["label"]:
            bad_label += 1
    A("A10-定位取回构造槽位", bad_loc == 0, f"违例={bad_loc}/{len(both)}")
    A("A10-是不与两提及重叠", bad_verb == 0, f"违例={bad_verb}/{len(both)}")
    A("A10-标签=实体id相等", bad_label == 0, f"违例={bad_label}/{len(both)}")

    # A11 字面不可分（语句级）：相同字面只在 DIFF 出现；不同字面在 SAME 与 DIFF 都出现
    eq_labels = sorted(set(it["label"] for it in both if it["s1"] == it["s2"]))
    neq_labels = sorted(set(it["label"] for it in both if it["s1"] != it["s2"]))
    A("A11-字面相同⇒只在DIFF下", eq_labels == ["DIFF"],
      f"字面相同的标签集合={eq_labels} 条数={sum(1 for it in both if it['s1'] == it['s2'])}")
    A("A11-字面不同⇒SAME与DIFF都出现", neq_labels == ["DIFF", "SAME"],
      f"字面不同的标签集合={neq_labels} "
      f"SAME={sum(1 for it in both if it['s1'] != it['s2'] and it['label']=='SAME')} "
      f"DIFF={sum(1 for it in both if it['s1'] != it['s2'] and it['label']=='DIFF')}")

    # A12 实体数/提及数/同句配平跨 split 一致（比例口径）
    comp = {}
    for lab in ("SAME", "DIFF"):
        comp[lab] = {
            "person": sum(1 for it in both if it["label"] == lab and it["kind"] == "person"),
            "org": sum(1 for it in both if it["label"] == lab and it["kind"] == "org"),
        }
    A("A12-两标签 person/org 构成一致", comp["SAME"] == comp["DIFF"], f"{comp}")

    out = {"all_pass": ok_all, "n_train": len(train), "n_test": len(test), "assertions": results}
    (DATA / "e01_assertions.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(f"[{'ALL PASS' if ok_all else 'FAILED'}] E0-1 断言 {sum(r['pass'] for r in results)}/{len(results)}")
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
