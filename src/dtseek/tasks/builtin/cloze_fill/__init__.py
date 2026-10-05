"""完形填空任务卡：`带空位的句子 + 4 个候选词`，选出该填的那个词。

类别与择优回复同族（dev-notes/13 §1）：`classes[0]` = **无合适候选**（背景类，
样本吐 0 切片），`classes[1..4]` = 候选编号；锚点是**被选中候选正文**在原文上的
0-based 半开区间（空位 `__` 只是输入里的一处标记，锚点不落在它身上 ——
下游组合算子要的是「选中了哪个词」，不是「空在哪」）。

`segment_policy` 必须是 `window`（候选列表不能被标点切开）；`max_len=96`，
数据侧用 `TEXT_LIMIT = max_len - 8`（引擎整段阈值）卡住样本。

唯一性校验与空位位置配额都在 `dataset.build_cloze_dataset` 里 fail-closed。

探针纪律（dev-notes/06 §7 / dev-notes/13 §3）：
  - 探针句子是**手写**的，与训练用的自然语料句子不重合；
  - 每条 (句子, 正确词) 在 4 个槽位各考一遍，类别头与指针头分开可判；
  - 覆盖句首空位、句尾空位顶段尾、`，__` 标点相邻、问句形态等边界组合。
"""
from __future__ import annotations

from dtseek.tasks.builtin.cloze_fill.dataset import N_CANDIDATES, build_cloze_dataset
from dtseek.tasks.builtin.reply_pick.frame import (
    BLANK,
    SEPARATOR,
    SLOT_CLOSE,
    validate_markers,
)
from dtseek.tasks.plugin import DEFAULT_SEED, ProbeUnit, TaskClass, TaskSpec, register

SPEC = TaskSpec(
    name="cloze_fill",
    label="完形填空",
    classes=(
        TaskClass("无合适候选"),
        TaskClass("候选1", color="\033[1;92;40m"),
        TaskClass("候选2", color="\033[1;91;40m"),
        TaskClass("候选3", color="\033[1;94;40m"),
        TaskClass("候选4", color="\033[1;96;44m"),
    ),
    max_steps=1,
    max_len=96,
    emission="single",
    segment_policy="window",
)

#: 换帧探针：(带空位的句子, 正确词)。句式与训练语料不重合，
#: 且覆盖 句首 / 句中 / 句末 / `，__` 标点相邻 / 问句 等形态。
PROBE_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("这份报告写得很" + BLANK + "，一眼就能看懂", "清楚"),
    ("" + BLANK + "得越仔细，返工就越少", "检查"),
    ("这件事我早就跟他" + BLANK + "了", "说明"),
    ("时间太紧，我们" + BLANK + "出发吧", "马上"),
    ("" + BLANK + "之前记得把门窗关好", "离开"),
    ("你先" + BLANK + "一下这个数字对不对", "核对"),
    ("他每天都" + BLANK + "半小时的英语", "朗读"),
    ("雨这么大，还是" + BLANK + "出门为好", "别"),
)

#: 干扰项池：通顺但填不进去的词（与正确词同为常用词，答非所「空」）。
PROBE_DISTRACTORS: tuple[str, ...] = (
    "复杂", "旅行", "音乐", "邮局", "昨天", "漂亮", "同意", "钢笔",
)


def probe_frame(scenario_idx: int, slot: int) -> str:
    """把正确词（`{w}` 占位）放进 `slot`（1..4），其余槽填干扰项。

    返回**载体模板**：模板里不得出现正确词字面（`probe.build_cases` 会因此抛错）。
    """
    sentence = PROBE_SCENARIOS[scenario_idx][0]
    parts = [sentence]
    # 干扰项按 scenario 轮转：同一 scenario 换槽位时**干扰项保持不变**，
    # 这样 4 个槽位之间的差异就只有正确词的位置（位置探针的变量控制）。
    base = scenario_idx % len(PROBE_DISTRACTORS)
    pool = [PROBE_DISTRACTORS[(base + k) % len(PROBE_DISTRACTORS)] for k in range(N_CANDIDATES - 1)]
    di = 0
    for n in range(1, N_CANDIDATES + 1):
        if n == slot:
            cand = "{w}"
        else:
            cand = pool[di]
            di += 1
        parts.append(f"{SEPARATOR}{n}{SLOT_CLOSE}{cand}")
    return "".join(parts)


class ClozeFillCard:
    spec = SPEC

    def build_dataset(self, target_samples: int, seed: int = DEFAULT_SEED) -> list[dict]:
        return build_cloze_dataset(target_samples=target_samples, seed=seed)

    def probe_units(self) -> list[ProbeUnit]:
        """每个 (句子, 槽位) 一个单元；同一个正确词在 4 个槽位各考一遍。"""
        validate_markers()
        return [
            ProbeUnit(key=f"候选{slot}#{s_idx}", expected_class=slot, words=(answer,),
                      carriers=(probe_frame(s_idx, slot),))
            for s_idx, (_sent, answer) in enumerate(PROBE_SCENARIOS)
            for slot in range(1, N_CANDIDATES + 1)
        ]

    def sanity_cases(self) -> list[tuple[str, int]]:
        """已知答案对照：探针先在这些帧上给出预期结论，否则先怀疑探针。"""
        return [
            (probe_frame(0, 1).format(w="清楚"), 1),
            (probe_frame(2, 3).format(w="说明"), 3),
            (probe_frame(7, 4).format(w="别"), 4),
        ]


CARD = ClozeFillCard()
register(CARD)
