"""否定标记任务卡：圈出句子里的否定标记（不 / 没 / 别 / 未 …），背景类零标记。

类别只有两个：`背景`（index 0）/ `否定`；`span` 是标记本身在原文上的 0-based 半开区间。
`classes[0]` 是背景类，`annotate_all=True` —— 句中出现的标记必须全部标出，
数据集侧保证不超 `max_steps`（超了直接在构建期响亮失败）。

探针纪律（dev-notes/06 §7）：
  - 每个标记配**与训练载体句式不重合**的探针载体；
  - 类别错与定位错分开报（`run_probe` 的 `class_acc` / `span_acc`）；
  - 每个单元自带「无终止标点、标记顶句尾」的裸词形态（dev-notes/05 §3.4）。
`probe_units()` 每次调用都会先跑 `validate_probe_carriers()` —— 坏载体 fail-closed。
"""
from __future__ import annotations

from dtseek.tasks.builtin.negation.dataset import (
    LABEL_NEG,
    NEG_MARKERS,
    PROBE_CARRIERS_BY_MARKER,
    build_negation_dataset,
    validate_probe_carriers,
)
from dtseek.tasks.plugin import DEFAULT_SEED, ProbeUnit, TaskClass, TaskSpec, register

SPEC = TaskSpec(
    name="negation",
    label="否定标记",
    classes=(
        TaskClass("背景"),
        TaskClass("否定", color="\033[1;95;40m"),
    ),
    max_steps=4,
    max_len=64,
    emission="single",
    segment_policy="sentence",
    annotate_all=True,
)


class NegationCard:
    spec = SPEC

    def build_dataset(self, target_samples: int, seed: int = DEFAULT_SEED) -> list[dict]:
        return build_negation_dataset(target_samples=target_samples, seed=seed)

    def probe_units(self) -> list[ProbeUnit]:
        """每个标记一个单元；载体先过 fail-closed 校验，坏载体绝不带进探针。"""
        validate_probe_carriers()
        return [
            ProbeUnit(key=word, expected_class=LABEL_NEG, words=(word,),
                      carriers=PROBE_CARRIERS_BY_MARKER[word])
            for word in NEG_MARKERS
        ]

    def sanity_cases(self) -> list[tuple[str, int]]:
        """已知答案对照：探针先在这些句子上给出预期结论，否则先怀疑探针。"""
        return [
            ("我不同意这个方案。", 1),
            ("别着急，我马上到。", 1),
            ("没有找到对应的文件。", 1),
            ("数据库集群写入延迟保持在五毫秒以内。", 0),
        ]


CARD = NegationCard()
register(CARD)
