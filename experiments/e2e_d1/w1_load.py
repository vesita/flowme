#!/usr/bin/env python3
"""W1（E1）—— 端到端路径实际加载什么：基座 sha / 卡文件 sha / negation 是否真在 Plan 里。

用法：
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 uv run python experiments/e2e_d1/w1_load.py
"""
from __future__ import annotations

import json
import sys

from common import (  # noqa: E402
    CAPS, E_BASES, EVAL_FILES, HERE, ROOT, SEEDS,
    doc_sha, file_sha, split_of,
)

import torch  # noqa: E402

from dtseek.tasks.dialogue import DEFAULT_ATTACH, respond  # noqa: E402
from dtseek.tasks.engine import DEFAULT_BASE, MultiTaskEngine  # noqa: E402

EXPECT_ONLINE = "dc27db5337d16f185dd08cc50505390e53d03606b9d307457828f4ecbfdf6b1a"
N_TRIGGER = 128  # 触发比例的抽样条数（W3 的全量比例在 w0_e2e.py 里报）


def build_engine(base_rel: str) -> MultiTaskEngine:
    """与 `examples/example_dialogue.py:54-56` 逐字同构：默认四卡 + DEFAULT_ATTACH。"""
    eng = MultiTaskEngine(base_path=str(ROOT / base_rel))
    for path in DEFAULT_ATTACH:
        eng.attach(path)
    return eng


def main() -> int:
    out: dict = {
        "prereg": "experiments/e2e_d1/PREREG.md",
        "entry": "dtseek.tasks.dialogue.respond(engine, text, type_='plain')",
        "inner": "run_cards -> engine.predict (分段解码 _run_segment)",
        "src_literals": {"engine.DEFAULT_BASE": DEFAULT_BASE,
                         "dialogue.DEFAULT_ATTACH": list(DEFAULT_ATTACH)},
        "bases": E_BASES, "eval_files": EVAL_FILES, "seeds": list(SEEDS),
        "engines": {}, "trigger_probe": {},
    }

    print(f"[src] engine.DEFAULT_BASE    = {DEFAULT_BASE}")
    print(f"[src] dialogue.DEFAULT_ATTACH = {list(DEFAULT_ATTACH)}")
    print(f"[src] example_dialogue 建引擎 = MultiTaskEngine(base_path=DEFAULT_BASE); attach(DEFAULT_ATTACH)")

    # ---- 两个引擎实际加载的基座 + 卡文件 sha ----
    engines: dict[str, MultiTaskEngine] = {}
    for key, rel in E_BASES.items():
        eng = build_engine(rel)
        engines[key] = eng
        ds = doc_sha(eng._base["doc_encoder"])
        cards = {name: {"path": path, "file_sha256": file_sha(ROOT / path)}
                 for name, path in sorted(eng.card_paths.items())}
        out["engines"][key] = {
            "base_path": eng.base_path, "doc_sha256": ds,
            "attached": sorted(eng.attached), "cards": cards, "device": str(eng.device),
        }
        print(f"[engine {key}] base_path={eng.base_path}")
        print(f"              doc_sha256 = {ds}")
        print(f"              device={eng.device}  attached={sorted(eng.attached)}")
        for n, c in cards.items():
            print(f"              card {n:10s} {c['path']}  file_sha256={c['file_sha256'][:16]}…")

    a = out["engines"]["A_online"]["doc_sha256"]
    b = out["engines"]["B_own"]["doc_sha256"]
    out["selfcheck"] = {
        "A_is_online_dc27": a == EXPECT_ONLINE,
        "A_vs_B_same_doc_sha": a == b,
        "negation_card_sha_same_both": (
            out["engines"]["A_online"]["cards"]["negation"]["file_sha256"]
            == out["engines"]["B_own"]["cards"]["negation"]["file_sha256"]),
    }
    print(f"[selfcheck] A==dc27db53…: {out['selfcheck']['A_is_online_dc27']}  "
          f"A==B: {out['selfcheck']['A_vs_B_same_doc_sha']}  "
          f"negation 卡两引擎同文件: {out['selfcheck']['negation_card_sha_same_both']}")

    # ---- negation 是否真在 Plan 里 / 真被调用（抽样）----
    # 计划行 `describe()` 只打印**第一个**授权动作，所以不能靠字符串找 "negation"；
    # 「Plan 里是否含它」的权威口径 = 初始状态的授权集 `admissible_actions`。
    from dtseek.tasks.dispatch import (  # noqa: E402
        MAX_STEPS, Conf, InfoState, Signals, admissible_actions,
    )
    from dtseek.tasks.dialogue import (  # noqa: E402
        DIALOGUE_BITS, DIALOGUE_BIT_KINDS, DIALOGUE_REQUIRED_BITS,
    )
    s0 = Signals("plain", (InfoState.UNCHECKED,) * 4, Conf.LOW, MAX_STEPS)
    authorized0 = [a.card for a in admissible_actions(
        s0, registry=None, info_bits=DIALOGUE_BITS,
        required_bits=DIALOGUE_REQUIRED_BITS, bit_kinds=DIALOGUE_BIT_KINDS)]
    out["authorized_initial"] = authorized0
    print(f"[plan] 初始状态授权集 admissible_actions = {authorized0}  "
          f"⇒ negation 在 Plan 授权集内 = {'negation' in authorized0}")

    probe = split_of("negation", 42)
    texts = [it["text"] for it in probe[:N_TRIGGER]]
    for key, eng in engines.items():
        first = respond(eng, texts[0], type_="plain")
        called = 0
        plan_has_neg = 0
        for t in texts:
            rec = respond(eng, t, type_="plain")
            if "negation" in rec["cards_run"]:
                called += 1
            if any("negation" in ln for ln in rec["plan"]):
                plan_has_neg += 1
        out["trigger_probe"][key] = {
            "n": len(texts), "called_negation": called,
            "call_ratio": called / max(1, len(texts)),
            "plan_line_literal_negation": plan_has_neg,
            "plan_line_literal_ratio": plan_has_neg / max(1, len(texts)),
            "authorized_initial": authorized0,
        }
        print(f"[trigger {key}] n={len(texts)} cards_run 含 negation = {called} "
              f"({called / max(1, len(texts)):.4f})  plan 行字面含 'negation' = {plan_has_neg}（describe() 只打第一个授权动作，权威口径见 admissible_actions）")
    print("\n[plan 样本 0 / A_online]")
    rec0 = respond(engines["A_online"], texts[0], type_="plain")
    for i, ln in enumerate(rec0["plan"], 1):
        print(f"  plan[{i}]: {ln}")
    print(f"  cards_run = {rec0['cards_run']}  kind={rec0['kind']}  terminal={rec0['terminal']}")
    print(f"  text      = {rec0['text']}")
    print(f"  input     = {texts[0]!r}")

    res = HERE / "results" / "w1_load.json"
    res.parent.mkdir(parents=True, exist_ok=True)
    res.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {res}")
    print("W1_DONE")
    del engines
    return 0 if out["selfcheck"]["A_is_online_dc27"] else 3


if __name__ == "__main__":
    sys.exit(main())
