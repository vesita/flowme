"""候选选择卡族（reply_pick / cloze_fill）的契约单测。

对应 dev-notes/06 §10 加任务检查清单与 dev-notes/13 §3/§5 里可自动判定的条目：
类别 index 0 是背景、`max_steps=1`、`segment_policy=window`、长度卡在引擎分段阈值内、
标记符号词表内无 UNK 无碰撞、锚点逐字等于被选中候选、位置打散、
训练/评测折不相交、干扰项同类同频同字长、唯一性校验 fail-closed、
探针句式与训练不重合、同 seed 确定性。

语料扫描统一压到 `max_lines=60000`：跑得完、又足以让 10% 评测折拿到 ≥200 条轮对
（更小的窗口会触发数据集 fail-closed，那本身也是一条测试）。
"""
import pytest

# 被测的两张卡：直接 import 其包（包末尾 `register(CARD)` 即注册），
# 不依赖 builtin/__init__ 的导入顺序。
import dtseek.tasks.builtin.cloze_fill
import dtseek.tasks.builtin.reply_pick  # noqa: F401
from dtseek.tasks.builtin.cloze_fill.dataset import (
    MARGIN,
    MIN_SCORE,
    POS_LEXICON,
    _score,
    blank_report,
    build_cloze_dataset,
    word_class,
)
from dtseek.tasks.builtin.cloze_fill.dataset import (
    TEXT_LIMIT as CLOZE_LIMIT,
)
from dtseek.tasks.builtin.reply_pick.dataset import (
    TEXT_LIMIT as REPLY_LIMIT,
)
from dtseek.tasks.builtin.reply_pick.dataset import (
    build_reply_pick_dataset,
    difficulty_report,
    position_report,
)
from dtseek.tasks.builtin.reply_pick.frame import (
    BLANK,
    FRAME_TOKENS,
    QUESTION_HEAD,
    SEPARATOR,
    SLOT_CLOSE,
    check_text_length,
    content_problems,
    frame_token_problems,
    render_choice_list,
)
from dtseek.tasks.plugin import all_tasks, probe_units_of, resolve_tasks
from dtseek.tasks.probe import build_cases
from dtseek.tasks.runtime import GenericTaskDataset

SCAN = 120_000
SMALL = 200


# ---- spec -----------------------------------------------------------------

@pytest.mark.parametrize("name,max_len", [("reply_pick", 128), ("cloze_fill", 96)])
def test_spec_shape(name, max_len):
    spec = resolve_tasks([name])[name].spec
    assert spec.class_names == ("无合适候选", "候选1", "候选2", "候选3", "候选4")
    assert spec.num_classes == 5                 # classes[0] 必须是背景类
    assert spec.max_steps == 1                   # 选择任务只发射一次
    assert spec.max_len == max_len
    assert spec.emission == "single"
    assert spec.segment_policy == "window"       # sentence 会切开候选列表
    assert not spec.annotate_all
    assert not spec.identity_labels
    assert spec.problems() == []


@pytest.mark.parametrize("name,limit", [("reply_pick", REPLY_LIMIT), ("cloze_fill", CLOZE_LIMIT)])
def test_text_limit_is_engine_window(name, limit):
    """文本上限必须等于引擎 window 的整段阈值 `max_len - 8`，否则训练口径与推理口径分叉。"""
    assert limit == all_tasks()[name].spec.max_len - 8


@pytest.mark.parametrize("name", ["reply_pick", "cloze_fill"])
def test_registered_like_every_builtin_card(name):
    card = all_tasks()[name]
    assert card is all_tasks()[name]
    assert len({c.color for c in card.spec.classes}) == card.spec.num_classes


# ---- 标记符号纪律（dev-notes/13 §3） ---------------------------------------

def test_frame_markers_have_deterministic_vocab_ids():
    """标记必须逐字 1 token、非 UNK/特殊 token、互不碰撞（`｜`/`＿＿` 实测是 UNK）。"""
    assert frame_token_problems() == []
    assert set(FRAME_TOKENS) == {"问", "：", SEPARATOR, SLOT_CLOSE, BLANK[0],
                                 "1", "2", "3", "4"}


def test_design_glyphs_are_unk_in_this_tokenizer():
    """设计稿里的 `｜` / `＿＿` 就是 UNK —— 这条钉住「为什么要换形」的事实前提。"""
    from nano_char_tokenizer import NanoCharTokenizer

    tok = NanoCharTokenizer()
    for glyph in ("｜", "＿"):
        ids = tok.encode(glyph, max_length=4, padding=False)["input_ids"]
        assert ids == [tok.unk_token_id], glyph


# ---- 框架渲染 --------------------------------------------------------------

def test_render_choice_list_spans_map_back():
    text, spans = render_choice_list(["甲", "乙丙", "丁", "戊己庚"])
    assert len(spans) == 4
    for (s, e), cand in zip(spans, ["甲", "乙丙", "丁", "戊己庚"]):
        assert text[s:e] == cand
    assert text.startswith(f"{SEPARATOR}1{SLOT_CLOSE}")


@pytest.mark.parametrize("bad", [
    ["a", "b", "c"],            # 数量不对
    ["a", "b", "c", "a"],        # 重复候选
    ["a", "b", "c", "x|y"],      # 内容含分隔符
    ["a", "b", "c", "＿"],       # 内容含空位符号
])
def test_render_choice_list_fails_closed(bad):
    with pytest.raises(ValueError):
        render_choice_list(bad)


def test_content_and_length_gates():
    assert content_problems("今天天气不错") == []
    assert content_problems("答案是 | 4）")
    with pytest.raises(ValueError, match="窗口阈值"):
        check_text_length("长" * (REPLY_LIMIT + 1), REPLY_LIMIT)


# ---- reply_pick 数据集 -----------------------------------------------------

@pytest.fixture(scope="module")
def reply_train():
    return build_reply_pick_dataset(target_samples=SMALL, seed=11, max_lines=SCAN)


@pytest.fixture(scope="module")
def reply_eval():
    return build_reply_pick_dataset(target_samples=SMALL, seed=12, split="eval",
                                    max_lines=SCAN)


def test_reply_dataset_shape(reply_train):
    assert len(reply_train) == SMALL
    for item in reply_train:
        meta = item["meta"]
        assert len(item["spans"]) <= 1
        assert len(item["text"]) <= REPLY_LIMIT
        if meta["kind"] == "solvable":
            assert len(item["spans"]) == 1
            sp = item["spans"][0]
            assert 1 <= sp["label"] <= 4
            # 锚点必须逐字等于被选中候选（P5 的前提）
            assert item["text"][sp["start"]:sp["end"]] == meta["cands"][sp["label"] - 1]
            assert item["text"][sp["start"]:sp["end"]] == meta["true"]
            assert meta["answer_pos"] == sp["label"]
        else:
            assert item["spans"] == []           # 无合适候选 → 背景样本吐 0 切片
            assert meta["answer_pos"] == 0
        # 内容（问句）里不得出现标记字符 —— 结构只能有一种读法
        assert content_problems(meta["q"]) == []
        head, _, tail = item["text"].rpartition(f"{SEPARATOR}4{SLOT_CLOSE}")
        assert head and tail


def test_reply_positions_cover_all_slots(reply_train):
    dist = position_report(reply_train)
    assert sorted(dist) == [1, 2, 3, 4]
    counts = list(dist.values())
    assert max(counts) - min(counts) <= 0.25 * sum(counts)   # 位置大体均匀


def test_reply_difficulty_mix(reply_train):
    report = difficulty_report(reply_train)
    assert "档1跑题" in report and sum(report.values()) == SMALL


def test_reply_split_disjoint(reply_train, reply_eval):
    tr = {x["meta"]["q"] for x in reply_train}
    ev = {x["meta"]["q"] for x in reply_eval}
    assert not (tr & ev), f"训练折与评测折问句重叠 {len(tr & ev)} 条"


def test_reply_same_seed_deterministic():
    a = build_reply_pick_dataset(target_samples=100, seed=7, max_lines=SCAN)
    b = build_reply_pick_dataset(target_samples=100, seed=7, max_lines=SCAN)
    key = lambda ds: [(x["text"], tuple(x["spans"])) for x in ds]
    assert key(a) == key(b)


def test_reply_fixed_and_shuffled_are_paired():
    """P1 的配对前提：固定序与打乱序是同一批轮对、同一批负例，只差正解位置。"""
    shuf = build_reply_pick_dataset(target_samples=100, seed=21, max_lines=SCAN)
    fixed = build_reply_pick_dataset(target_samples=100, seed=21, fixed_answer_pos=1,
                                     max_lines=SCAN)
    assert [x["meta"]["q"] for x in shuf] == [x["meta"]["q"] for x in fixed]
    assert [x["meta"]["cands"] for x in shuf if x["meta"]["kind"] == "solvable"] != \
           [x["meta"]["cands"] for x in fixed if x["meta"]["kind"] == "solvable"]
    assert all(x["meta"]["answer_pos"] == 1 for x in fixed
               if x["meta"]["kind"] == "solvable")


def test_reply_samples_encode_without_unk():
    """整段必须逐字 1 token 且无 UNK —— 标记符号一旦是 UNK，锚点坐标就整体错位。"""
    from nano_char_tokenizer import NanoCharTokenizer

    tok = NanoCharTokenizer()
    ds = build_reply_pick_dataset(target_samples=60, seed=13, max_lines=SCAN)
    for item in ds:
        ids = tok.encode(item["text"], max_length=512, padding=False)["input_ids"]
        assert len(ids) == len(item["text"])
        assert tok.unk_token_id not in ids
        assert len(ids) <= all_tasks()["reply_pick"].spec.max_len


def test_reply_encodes_through_generic_dataset(reply_train):
    from nano_char_tokenizer import NanoCharTokenizer

    spec = all_tasks()["reply_pick"].spec
    encoded = GenericTaskDataset(reply_train[:16], NanoCharTokenizer(), spec)
    for i in range(len(encoded)):
        encoded[i]      # 不抛即通过


def test_reply_too_small_fails_closed():
    """扫描窗口太小 / 轮对不够时必须响亮失败，不允许静默出缩水数据集。"""
    with pytest.raises(ValueError, match="太小|不完整"):
        build_reply_pick_dataset(target_samples=5000, seed=3, max_lines=6_000)


# ---- cloze_fill 数据集 -----------------------------------------------------

@pytest.fixture(scope="module")
def cloze_train():
    return build_cloze_dataset(target_samples=SMALL, seed=11, max_lines=SCAN)


@pytest.fixture(scope="module")
def cloze_eval():
    return build_cloze_dataset(target_samples=SMALL, seed=12, split="eval", max_lines=SCAN)


def test_cloze_dataset_shape(cloze_train):
    assert len(cloze_train) == SMALL
    for item in cloze_train:
        assert len(item["text"]) <= CLOZE_LIMIT
        assert BLANK in item["text"]
        if item["meta"]["kind"] == "solvable":
            assert len(item["spans"]) == 1
            sp = item["spans"][0]
            assert item["text"][sp["start"]:sp["end"]] == item["meta"]["word"]
            assert sp["label"] == item["meta"]["answer_pos"]
        else:
            assert item["spans"] == []
        # 空位两侧不能是候选词本身（否则「找词」不用看空位）
        assert BLANK in item["meta"]["ctx"]
        assert item["meta"]["ctx"] in item["text"]


def test_cloze_distractors_same_class_and_length(cloze_train):
    """干扰项必须同类同字长（同频在构造里按词频桶卡，这里复核词类与字长）。"""
    for item in cloze_train:
        cands = item["meta"]["cands"]
        assert len({len(c) for c in cands}) == 1, cands
        classes = {word_class(c) for c in cands}
        assert len(classes) == 1, (cands, classes)
        assert classes <= set(POS_LEXICON)


def test_cloze_uniqueness_rule_holds_for_solvable(cloze_train):
    """正解样本必须满足预注册的唯一性判据（用缓存语料池复算，不是只信构造时的数字）。"""
    from dtseek.tasks.builtin.cloze_fill.dataset import load_cloze_pool

    pool = load_cloze_pool(SCAN)
    checked = 0
    for item in cloze_train:
        meta = item["meta"]
        if meta["kind"] != "solvable":
            continue
        head = item["text"].split(f"{SEPARATOR}1{SLOT_CLOSE}")[0]
        left, _, right = head.partition(BLANK)
        prev = left[-1] if left else None
        nxt = right[0] if right else None
        s_true = _score(pool, meta["word"], prev, nxt)
        s_neg = [_score(pool, w, prev, nxt) for w in meta["cands"] if w != meta["word"]]
        assert s_true >= MIN_SCORE, (item["text"], s_true)
        assert s_true >= MARGIN * max(s_neg), (item["text"], s_true, s_neg)
        checked += 1
    assert checked > 0


def test_cloze_bg_candidates_all_weak(cloze_train):
    from dtseek.tasks.builtin.cloze_fill.dataset import load_cloze_pool

    pool = load_cloze_pool(SCAN)
    n = 0
    for item in cloze_train:
        if item["meta"]["kind"] != "bg":
            continue
        head = item["text"].split(f"{SEPARATOR}1{SLOT_CLOSE}")[0]
        left, _, right = head.partition(BLANK)
        prev = left[-1] if left else None
        nxt = right[0] if right else None
        for w in item["meta"]["cands"]:
            assert _score(pool, w, prev, nxt) < MIN_SCORE, (item["text"], w)
        n += 1
    assert n > 0


def test_cloze_blank_position_quotas(cloze_train):
    kinds = blank_report(cloze_train)
    n = sum(kinds.values())
    assert kinds.get("句首", 0) >= 0.15 * n - 1
    assert kinds.get("句末", 0) >= 0.15 * n - 1


def test_cloze_split_disjoint(cloze_train, cloze_eval):
    tr = {x["meta"]["ctx"] for x in cloze_train}
    ev = {x["meta"]["ctx"] for x in cloze_eval}
    assert not (tr & ev), f"训练折与评测折句子重叠 {len(tr & ev)} 条"


def test_cloze_same_seed_deterministic():
    a = build_cloze_dataset(target_samples=100, seed=7, max_lines=SCAN)
    b = build_cloze_dataset(target_samples=100, seed=7, max_lines=SCAN)
    key = lambda ds: [(x["text"], tuple(x["spans"])) for x in ds]
    assert key(a) == key(b)


def test_cloze_probe_frames_disjoint_from_training(cloze_train):
    from dtseek.tasks.builtin.cloze_fill import PROBE_SCENARIOS

    heads = {x["meta"]["ctx"] for x in cloze_train}
    for sentence, _word in PROBE_SCENARIOS:
        assert sentence not in heads, f"探针句子与训练句子重合：{sentence}"


# ---- 探针纪律（dev-notes/06 §7） -------------------------------------------

@pytest.mark.parametrize("name", ["reply_pick", "cloze_fill"])
def test_probe_units_pass_build_cases(name):
    units = probe_units_of(all_tasks()[name])
    assert units, name
    cases = build_cases(units)                     # 载体含待测词字面会在这里抛错
    assert len(cases) == len(units)
    for unit, case in zip(units, cases):
        assert case["expected_class"] == unit.expected_class
        assert 1 <= case["expected_class"] <= 4     # 类别 0 是无合适，探针不考它
        (w, s, e) = case["expected_spans"][0]
        assert case["text"][s:e + 1] == w          # 闭区间端点必须落回词本身
        assert len(case["text"]) <= all_tasks()[name].spec.max_len - 8


def test_reply_probe_frames_disjoint_from_training(reply_train):
    from dtseek.tasks.builtin.reply_pick import PROBE_SCENARIOS

    heads = {x["meta"]["q"] for x in reply_train}
    for q, _a in PROBE_SCENARIOS:
        assert q not in heads, f"探针问句与训练问句重合：{q}"


def test_reply_probe_covers_marker_adjacency():
    """探针要覆盖「问句紧接候选列表」的标记相邻形态（dev-notes/09 §2 的组合形态）。"""
    from dtseek.tasks.builtin.reply_pick import PROBE_SCENARIOS, probe_frame

    frame = probe_frame(0, 1)
    # 问句无终止标点，`问：{问句}` 之后**紧贴** `|1）` —— 标记相邻形态在探针里必须出现
    assert QUESTION_HEAD + PROBE_SCENARIOS[0][0] + SEPARATOR + "1" + SLOT_CLOSE in frame
    # 且探针问句本身不以问号收尾（训练里也有这种形态，组合不悬空）
    assert not PROBE_SCENARIOS[0][0].endswith(("？", "?"))
