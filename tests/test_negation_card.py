"""否定任务卡的契约单测：spec / 提取器 / fail-closed 门禁 / 探针纪律 / 数据集形状。

对应 dev-notes/06 §10 加任务检查清单里可自动判定的条目：
类别 index 0 是背景、span 是 0-based 半开、载体纯净、背景零标记、
探针句式与训练不重合且覆盖裸词形态、样本切片数不超 max_steps。
"""
import pytest

from dtseek.tasks.builtin.negation.dataset import (
    BACKGROUND_POOL,
    CARRIERS_BY_MARKER,
    MAX_STEPS_PER_SAMPLE,
    NEG_MARKERS,
    PROBE_CARRIERS_BY_MARKER,
    build_negation_dataset,
    extract_negation_spans,
    reject_reason,
    style_problems,
    validate_carriers,
    validate_probe_carriers,
    word_coverage_report,
)
from dtseek.tasks.plugin import all_tasks, probe_units_of, resolve_tasks
from dtseek.tasks.runtime import GenericTaskDataset

TERMINAL = set("。！？!?；;，,")


def _norm(tpl: str) -> str:
    return tpl.replace("{e}", "\0").replace("{w}", "\0")


# ---- spec -----------------------------------------------------------------

def test_spec_shape():
    spec = resolve_tasks(["negation"])["negation"].spec
    assert spec.class_names == ("背景", "否定")      # classes[0] 必须是背景类
    assert spec.num_classes == 2
    assert spec.max_steps == MAX_STEPS_PER_SAMPLE
    assert spec.max_len == 64
    assert spec.emission == "single"
    assert spec.segment_policy == "sentence"
    assert spec.annotate_all


def test_registered_like_every_builtin_card():
    card = all_tasks()["negation"]
    assert card is all_tasks()["negation"]
    assert card.spec.problems() == []
    assert len({c.color for c in card.spec.classes}) == card.spec.num_classes


# ---- 提取器（唯一真相源） ---------------------------------------------------

def test_longest_first_matching():
    got = extract_negation_spans("你不要这样。")
    assert [(s["word"], s["start"], s["end"]) for s in got] == [("不要", 1, 3)]
    got = extract_negation_spans("我没有时间。")
    assert [(s["word"], s["start"], s["end"]) for s in got] == [("没有", 1, 3)]


def test_span_half_open_and_maps_back_to_text():
    text = "我不知道，他也没来。"
    spans = extract_negation_spans(text)
    assert [s["word"] for s in spans] == ["不", "没"]      # 按 start 升序
    for s in spans:
        assert s["label"] == 1
        assert 0 <= s["start"] < s["end"] <= len(text)
        assert text[s["start"]:s["end"]] == s["word"]


def test_every_marker_extractable():
    for word in NEG_MARKERS:
        spans = extract_negation_spans(f"他{word}一下")
        assert [s["word"] for s in spans] == [word], word


# ---- 丢弃规则：同形异词 / 虚化固定式 / A-not-A ------------------------------

@pytest.mark.parametrize("text", [
    "是不是你", "有没有时间", "去不去啊", "知不知道", "吃没吃",
])
def test_a_not_a_question_rejected(text):
    assert reject_reason(text) is not None


@pytest.mark.parametrize("text", [
    # 同形异词
    "别人已经到了", "未来很美好", "这两个人有区别", "识别图像", "特别好用",
    "性别不同", "出没无常", "莫斯科很冷",
    # 虚化固定式
    "不得不承认", "这方案不错", "差不多得了", "不管你怎么说", "怪不得是他",
    # 长词优先的切分歧义（真标记是「不」，圈进「要/用」就是 span 错位）
    "不必要加班", "不要紧", "不要求双向联系", "不用心做事",
])
def test_homograph_grammaticalized_or_scope_ambiguous_rejected(text):
    assert reject_reason(text) is not None, text


@pytest.mark.parametrize("text", [
    "我不同意这个方案", "没有找到文件", "别着急", "尚未提交", "何必呢",
])
def test_genuine_negation_stays_usable(text):
    assert reject_reason(text) is None


# ---- 载体门禁（fail-closed） ------------------------------------------------

def test_all_validators_pass_on_shipped_tables():
    validate_carriers()
    validate_probe_carriers()


def test_training_carrier_with_marker_in_static_part_rejected():
    with pytest.raises(ValueError, match="静态部分含标记"):
        validate_carriers({"不": ("我{e}去，就是不。",)})


def test_probe_carrier_yielding_extra_marker_rejected():
    # 「我{w}用」填入「不」→ 抽出「不用」而不是「不」，定位/类别都会被带偏
    with pytest.raises(ValueError, match="恰好抽出"):
        validate_probe_carriers({"不": ("我{w}用。",)})


def test_probe_frames_disjoint_from_training():
    train = {_norm(t) for ts in CARRIERS_BY_MARKER.values() for t in ts}
    probe = {_norm(t) for ts in PROBE_CARRIERS_BY_MARKER.values() for t in ts}
    assert not (train & probe), f"句式重合: {train & probe}"
    train_text = {t.format(e=m) for m, ts in CARRIERS_BY_MARKER.items() for t in ts}
    probe_text = {t.format(w=m) for m, ts in PROBE_CARRIERS_BY_MARKER.items() for t in ts}
    assert not (train_text & probe_text), f"填充句重合: {train_text & probe_text}"


# ---- 探针单元 ---------------------------------------------------------------

def test_probe_units_cover_every_marker_with_bare_tail():
    units = probe_units_of(all_tasks()["negation"])
    assert {u.key for u in units} == set(NEG_MARKERS)
    assert all(u.expected_class == 1 for u in units)
    for u in units:
        word = u.words[0]
        assert all(word not in tpl for tpl in u.carriers)
        bare_tail = [
            tpl for tpl in u.carriers
            if not (set(tpl.format(w=word)) & TERMINAL) and tpl.format(w=word).endswith(word)
        ]
        assert bare_tail, f"{word} 缺少无终止标点且标记顶句尾的载体"


def test_sanity_cases_match_extractor_ground_truth():
    cases = all_tasks()["negation"].sanity_cases()
    assert len(cases) >= 3
    assert any(want == 0 for _, want in cases)
    for text, want in cases:
        assert reject_reason(text) is None
        assert bool(extract_negation_spans(text)) == bool(want), text


# ---- 背景与语体门禁 ---------------------------------------------------------

def test_background_pool_is_marker_free_and_styled():
    for s in BACKGROUND_POOL:
        assert extract_negation_spans(s) == [], s
        assert reject_reason(s) is None, s
        assert style_problems(s) == [], s


def test_style_gate_blocks_english_and_code():
    assert style_problems("def foo(): return 1")
    assert style_problems("Returns: int or float")
    assert not style_problems("今天天气晴朗，气温约二十二度。")


# ---- 数据集构建 -------------------------------------------------------------

@pytest.fixture(scope="module")
def small_ds():
    return build_negation_dataset(target_samples=1000, seed=7, max_lines=60000)


def test_dataset_contract(small_ds):
    assert len(small_ds) == 1000
    n_pos = 0
    for item in small_ds:
        text = item["text"]
        spans = item["spans"]
        assert len(spans) <= MAX_STEPS_PER_SAMPLE
        if not spans:
            assert extract_negation_spans(text) == [], f"背景句含未标注否定词: {text!r}"
            continue
        n_pos += 1
        for s in spans:
            assert s["label"] == 1
            assert 0 <= s["start"] < s["end"] <= len(text)
            assert text[s["start"]:s["end"]] == s["word"] in set(NEG_MARKERS)
    assert n_pos == 650          # POS_RATIO = 0.65


def test_every_marker_has_coverage_floor(small_ds):
    cover = word_coverage_report(small_ds)
    assert all(n > 0 for n in cover.values()), cover


def test_annotate_all_encoding_passes(small_ds):
    from nano_char_tokenizer import NanoCharTokenizer

    spec = all_tasks()["negation"].spec
    encoded = GenericTaskDataset(small_ds[:16], NanoCharTokenizer(), spec)
    for i in range(len(encoded)):
        encoded[i]                                  # 不抛即通过（annotate_all 会在超限时响亮失败）


def test_target_too_small_fails_closed():
    with pytest.raises(ValueError, match="装不下"):
        build_negation_dataset(target_samples=500, seed=1)


def test_same_seed_is_deterministic():
    a = build_negation_dataset(target_samples=1000, seed=7, max_lines=60000)
    b = build_negation_dataset(target_samples=1000, seed=7, max_lines=60000)
    key = lambda ds: [(x["text"], tuple((s["start"], s["end"]) for s in x["spans"])) for x in ds]
    assert key(a) == key(b)
