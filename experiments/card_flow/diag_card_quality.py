#!/usr/bin/env python3
"""①候选区的**后验诊断**（不进判据，只报数）：真卡吐的切片到底像不像本卡要找的东西。

判据：把每条**通过②筛选**的候选，拿去跟**这张卡自己的训练真值词典**对一遍
（pronoun→人称代词表 / negation→NEG_MARKERS / relation→成语表 / sentiment→情绪词典 /
 person→姓名表∪人称代词表（**称谓未纳入 ⇒ 这是下界**））。
这条诊断**不参与 G1–G4 判定**，只用来解释"为什么生成的句子不像话"。
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from dtseek.tasks.builtin.idiom.lexicon import IDIOMS  # noqa: E402
from dtseek.tasks.builtin.negation.dataset import NEG_MARKERS  # noqa: E402
from dtseek.tasks.builtin.person.dataset import FEMALE_NAMES, MALE_NAMES  # noqa: E402
from dtseek.tasks.builtin.sentiment.dataset import LEXICON_BY_CAT  # noqa: E402

PRONOUNS = {"我", "我们", "咱们", "你", "您", "你们", "他", "她", "它", "他们", "她们"}
SENTIMENT = {w for ws in LEXICON_BY_CAT.values() for w in ws}
IDIOM_SET = set(IDIOMS)
PERSON = set(MALE_NAMES) | set(FEMALE_NAMES) | PRONOUNS

LEXICON = {
    "pronoun": PRONOUNS,
    "negation": set(NEG_MARKERS),
    "relation": IDIOM_SET,
    "sentiment": SENTIMENT,
    "person": PERSON,
}


def main() -> int:
    data = json.loads((HERE / "results_real.json").read_text(encoding="utf-8"))
    records = data["batch_real"]["records"]
    tot = collections.Counter()
    valid = collections.Counter()
    hit = collections.Counter()
    fail_reason = collections.Counter()
    empty_span = collections.Counter()
    maxlen = collections.Counter()
    bad_example: dict[str, list] = collections.defaultdict(list)

    for rec in records:
        for c in rec.get("candidates", []):
            card = c["card"]
            tot[card] += 1
            if (c["s0"], c["e0"]) and c["s0"] == c["e0"]:
                empty_span[card] += 1
            if c["valid"]:
                valid[card] += 1
                maxlen[card] = max(maxlen[card], len(c["surface"]))
                ok = c["surface"] in LEXICON.get(card, set())
                if ok:
                    hit[card] += 1
                elif len(bad_example[card]) < 5:
                    bad_example[card].append(
                        {"surface": c["surface"], "class": c["class_name"],
                         "input": rec["input"][:24]})
            else:
                for f in c["failed"]:
                    fail_reason[(card, f)] += 1

    out = {"n_sentences": len(records), "per_card": {}}
    for card in sorted(tot):
        out["per_card"][card] = {
            "anchors": tot[card],
            "valid": valid[card],
            "valid_rate": round(valid[card] / tot[card], 4) if tot[card] else 0.0,
            "empty_span_s0_eq_e0": empty_span[card],
            "filter_failures": {f: n for (c, f), n in fail_reason.items() if c == card},
            "lexicon_hit_among_valid": hit[card],
            "lexicon_hit_rate": round(hit[card] / valid[card], 4) if valid[card] else None,
            "max_span_len": maxlen[card],
            "bad_examples": bad_example[card],
        }
    out["note"] = ("后验诊断，不进 G1–G4 判据；person 的词典只含姓名+人称代词（称谓未纳入）"
                   "⇒ person 命中率是下界")
    path = HERE / "diag_card_quality.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[写出] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
