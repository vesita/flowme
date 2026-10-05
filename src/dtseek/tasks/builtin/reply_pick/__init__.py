"""择优回复任务卡：`问句 + 4 条候选回复`，选最合适的一条并给出它的原文区间。

类别（dev-notes/13 §1）：`classes[0]` = **无合适候选**（背景类，样本吐 0 切片），
`classes[1..4]` = 候选编号；指针头给出**被选中候选正文**的 0-based 半开区间 ——
这个锚点就是阶段三组合算子 `chain/filter` 要消费的东西（dev-notes/13 §7）。

`segment_policy` 必须是 `window`：`sentence` 会按标点把候选列表切开，
候选结构在句边界丢失（dev-notes/13 §2）。`max_len=128` 已顶到基座窗口上限，
数据侧用 `TEXT_LIMIT = max_len - 8`（引擎整段阈值）把样本卡在不会被切的位置。

探针纪律（dev-notes/06 §7 / dev-notes/13 §3）：
  - 探针问句是**手写换帧句式**，与训练用的自然语料问句不重合；
  - 每条 (问句, 正确答案) 在 4 个槽位各考一遍 —— 类别头与指针头分开可判
    （`run_probe` 的 `class_acc` / `span_acc`），位置偏置也直接暴露；
  - 载体覆盖标记相邻形态：`？` 紧接 `|1）`、问句无终止标点、答案顶在段尾。
`probe_units()` 每次调用都先跑 `validate_markers()`，坏标记 fail-closed。
"""
from __future__ import annotations

from dtseek.tasks.builtin.reply_pick.dataset import (
    N_CANDIDATES,
    build_reply_pick_dataset,
)
from dtseek.tasks.builtin.reply_pick.frame import (
    SEPARATOR,
    SLOT_CLOSE,
    validate_markers,
)
from dtseek.tasks.plugin import DEFAULT_SEED, ProbeUnit, TaskClass, TaskSpec, register

SPEC = TaskSpec(
    name="reply_pick",
    label="择优回复",
    classes=(
        TaskClass("无合适候选"),
        TaskClass("候选1", color="\033[1;92;40m"),
        TaskClass("候选2", color="\033[1;91;40m"),
        TaskClass("候选3", color="\033[1;94;40m"),
        TaskClass("候选4", color="\033[1;96;44m"),
    ),
    max_steps=1,
    max_len=128,
    emission="single",
    segment_policy="window",
)

# ---------------------------------------------------------------------------
# 换帧探针（dev-notes/13 §5 P3）：每条 scenario = (问句, 正确答案)。
# 覆盖的边界形态（dev-notes/09 §2：组合形态必须在训练数据里出现过）：
#   问句无终止标点（`问：…|1）` 紧邻）、答案带终止标点、
#   问句里出现 `：` 与 `“”` —— 训练语料这两类符号大量出现，组合形态不悬空。
# 干扰项与答案同为通顺中文短句，但答非所问（跑题负例的探针版）。
# ---------------------------------------------------------------------------
PROBE_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("明早的班车几点发，改没改时刻", "售票窗口六点开门。"),
    ("这道菜要不要先焯一下水", "水开以后再下锅。"),
    ("你看这份说明书写得清楚吗", "第三步少画了一个箭头。"),
    ("周五之前能把报告交上来吗", "周四晚上给你初稿。"),
    ("屋子里怎么突然这么冷", "窗户一直开着呢。"),
    ("这两个方案你更倾向哪一个", "我选第二个，风险小一些。"),
    ("他为什么没来开会", "临时被叫去客户那边了。"),
    ("今天穿这件外套合适吗", "风大，还是加件厚的。"),
)

PROBE_DISTRACTORS: tuple[str, ...] = (
    "红烧肉要炖两个小时。",
    "文件我已经放共享盘了。",
    "下周三之前都可以退换。",
    "这篇散文一共分成四段。",
)


def probe_frame(scenario_idx: int, slot: int) -> str:
    """把 scenario 的正确答案（`{w}` 占位）放进 `slot`（1..4），其余槽填干扰项。

    返回的是**载体模板**：探针框架自己负责填 `{w}`，模板里不得出现答案字面
    （`probe.build_cases` 会因此直接抛错）。
    """
    q, _answer = PROBE_SCENARIOS[scenario_idx]
    distractors = [d for d in PROBE_DISTRACTORS]
    parts = [f"问：{q}"]
    di = 0
    for n in range(1, N_CANDIDATES + 1):
        if n == slot:
            cand = "{w}"
        else:
            cand = distractors[di]
            di += 1
        parts.append(f"{SEPARATOR}{n}{SLOT_CLOSE}{cand}")
    return "".join(parts)


class ReplyPickCard:
    spec = SPEC

    def build_dataset(self, target_samples: int, seed: int = DEFAULT_SEED) -> list[dict]:
        return build_reply_pick_dataset(target_samples=target_samples, seed=seed)

    def probe_units(self) -> list[ProbeUnit]:
        """每个 (问句, 槽位) 一个单元；同一句正确答案在 4 个槽位各考一遍。"""
        validate_markers()
        return [
            ProbeUnit(key=f"候选{slot}#{s_idx}", expected_class=slot, words=(answer,),
                      carriers=(probe_frame(s_idx, slot),))
            for s_idx, (_q, answer) in enumerate(PROBE_SCENARIOS)
            for slot in range(1, N_CANDIDATES + 1)
        ]

    def sanity_cases(self) -> list[tuple[str, int]]:
        """已知答案对照：探针先在这些帧上给出预期结论，否则先怀疑探针。"""
        return [
            (probe_frame(0, 1).format(w="售票窗口六点开门。"), 1),
            (probe_frame(5, 3).format(w="我选第二个，风险小一些。"), 3),
            (probe_frame(7, 4).format(w="风大，还是加件厚的。"), 4),
        ]


CARD = ReplyPickCard()
register(CARD)
