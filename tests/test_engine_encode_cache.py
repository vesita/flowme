"""基座编码缓存的守卫：开/关缓存逐位相同 + 调用次数降到"不同键数" + 键错了必须被抓。

缺陷与修复（`src/dtseek/tasks/engine.py`）：`predict` 的 docstring 曾写「基座只编码一次，
各卡共享」，实现却是**每张卡 × 每个段各编码一次**。修复后 `predict` 在**一次调用内**按
`(input_ids, attention_mask)` 缓存 `doc_memory`，同键直接复用同一个张量；键的含义与
"为什么不统一各卡的 max_len"写在 `_encode_cache_key` 的 docstring 里。

判别力（守卫为什么长这样）：
- 逐位守卫比较**开缓存**与**关缓存**（桩件关掉复用）的锚点 —— 关掉复用时两条路都退化成
  "每次重算"，输出当然相同 ⇒ 守卫通过；键算错时缓存会把**别的段**的 `doc_memory` 喂给
  当前段 ⇒ 锚点变 ⇒ 守卫失败（文末的反例测试就是拿它验判别力的）。
- 替身解码器的发射**由 doc_memory 的数值决定**（真解码器用 `bmm(doc_memory)` 取指针
  logits，同样吃内容）：只看形状的替身会让"喂错 memory"完全隐形，守卫就抓不到错键。

为什么用替身解码器：`checkpoints/` 在 .gitignore 里、不在版本控制中，测试不能依赖它。
基座走真实的 `MultiTaskEngine.__init__`（临时产物），只有各卡的网络被换掉 ——
`predict` → `_run_segment` 的产锚点循环、tokenizer、切段策略、spec、`doc_encoder`、
缓存全是真代码。仓库里真 4 张卡的同款守卫另见文末（checkpoints 在场才跑）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from nano_char_tokenizer import NanoCharTokenizer

from dtseek.encoder.nano_doc_encoder import NanoDocEncoder
from dtseek.encoder.segmenter import split_with_global_offsets
from dtseek.tasks import engine as engine_mod
from dtseek.tasks.artifacts import save_base
from dtseek.tasks.engine import MultiTaskEngine
from dtseek.tasks.plugin import all_tasks

HIDDEN = 32
REPO = Path(__file__).resolve().parents[1]
CARDS_DIR = REPO / "checkpoints" / "cards"

# 两段文本，都长到足以让 window 卡（reply_pick 的 limit=120）也切开：
# 每张已挂载卡都至少切出 2 段（见 test_every_card_sees_at_least_two_segments）。
TEXT_A = (
    "他说不难答案真棒，其实我很高兴。明天我们一起去公园散步，看看蓝天和白云。"
    "她并不觉得这个方案有什么问题，反而认为它很稳妥，所以大家就照着做了。"
    "可是到了下午，天气突然变了，雨下得很大，于是我们只好取消计划。"
    "有人说应该等一等，也有人说干脆改天再说，讨论了很久也没有结论。"
)
TEXT_B = (
    "模型上线之后必须持续监控，否则数据漂移会让指标悄悄下滑没人发现。"
    "团队每两周复盘一次线上的错误样本，把结论原样写进下一轮的训练集里。"
    "我们先在小流量上验证效果，再逐步放量，最后才把旧服务安安静静地退役。"
    "这样做虽然慢一些，却能在出问题时快速回滚，不影响用户的正常使用。"
)
TEXTS = [TEXT_A, TEXT_B]


# ---- 替身：内容敏感的解码器 -----------------------------------------------

class _ContentStubDecoder:
    """确定性替身：发射的类别 / 跨度 / 置信度都由 `doc_memory` 的**数值**决定。

    真解码器的指针 logits 是 `bmm(query, doc_memory)`，cls 也吃 doc_memory 的表示 ——
    换一段文本进去，发射就该变。替身照着这个性质写，"缓存键错了 ⇒ 喂进去的是别的段的
    memory"才会体现成锚点差异，逐位守卫才有判别力。
    """

    def __init__(self, num_classes: int, device, hidden_dim: int = HIDDEN):
        # bos_query 与基座同设备：`_run_segment` 拿它当 q_seq 的起点
        self.bos_query = torch.zeros(1, 1, hidden_dim, device=device)
        self._c = num_classes

    def forward_step(self, q_seq, doc_memory, doc_mask=None):
        step = q_seq.shape[1] - 1           # 第 k 步 = 第 k 个 query token
        dev = doc_memory.device
        mem = doc_memory.float().squeeze(0)  # [L_pad, D]
        L = mem.shape[0]
        valid = (doc_mask[0].nonzero(as_tuple=True)[0] if doc_mask is not None
                 else torch.arange(L, device=dev))
        n = int(valid.numel())
        # 内容指纹：换一段 doc_memory 就换一组数
        pos = (mem * torch.linspace(0.5, 1.5, mem.shape[1], device=dev)).sum(-1)
        v = pos[valid]
        base = int(v.argmax().item())
        shift = 1 + (int(abs(float(v.sum())) * 1e6) % max(1, n - 1)) if n > 1 else 0
        s_i = int(valid[(base + step) % n].item())
        e_i = int(valid[(base + step + shift) % n].item())
        fp = float(mem[valid].abs().mean())

        start = torch.full((1, L), -1e4, device=dev)
        end = torch.full((1, L), -1e4, device=dev)
        start[0, s_i] = 1.0
        end[0, e_i] = 1.0
        cls = torch.zeros((1, self._c), device=dev)
        cls[0, 1 + int(fp * 1e6) % (self._c - 1)] = 3.0 + (fp * 1000.0) % 1.0
        act = torch.zeros((1, 2), device=dev)
        act[0, 1] = 8.0                      # 一直 <cont>，由 max_steps 收口
        return {"cls_logits": cls, "action_logits": act,
                "start_logits": start, "end_logits": end,
                "last_hidden": torch.zeros(1, 1, HIDDEN, device=dev)}

    def get_step_input(self, *, prev_hidden, prev_cls, prev_start, prev_end):
        return torch.zeros(1, 1, prev_hidden.shape[-1], device=prev_hidden.device)


@pytest.fixture(scope="module")
def engine(tmp_path_factory) -> MultiTaskEngine:
    """真基座（临时产物）+ 每张已注册卡一个内容敏感的替身解码器，全部挂上。"""
    tok = NanoCharTokenizer()
    base = tmp_path_factory.mktemp("base") / "base_encoder.pt"
    save_base(base, NanoDocEncoder(vocab_size=tok.vocab_size, hidden_dim=HIDDEN,
                                   num_layers=1, num_heads=4, max_len=128, dropout=0.0),
              hidden_dim=HIDDEN, vocab_size=tok.vocab_size,
              encoder_kwargs={"num_layers": 1, "num_heads": 4, "max_len": 128})
    eng = MultiTaskEngine(base_path=str(base), cards_dir=None, auto_attach=False)
    for name, card in all_tasks().items():
        eng.specs[name] = card.spec
        eng.decoders[name] = _ContentStubDecoder(card.spec.num_classes, eng.device)
    return eng


# ---- 工具 -----------------------------------------------------------------

def _segments_for(eng: MultiTaskEngine, task: str, text: str) -> list[str]:
    """照抄 `predict` 的切段分支 —— 用来独立算"缓存关"应有的编码次数。"""
    spec = eng.specs[task]
    limit = max(16, spec.max_len - 8)
    if spec.segment_policy == "window" and len(text) <= limit:
        return [text]
    return [s["text"] for s in split_with_global_offsets(text, max_chunk_len=limit)]


def _distinct_keys(eng: MultiTaskEngine, text: str) -> set[tuple]:
    """独立于产品代码算键（键的定义在测试里重写一遍，免得跟着实现一起错）。"""
    keys = set()
    for task in eng.attached:
        spec = eng.specs[task]
        for seg in _segments_for(eng, task, text):
            enc = eng.tokenizer.encode(seg, max_length=spec.max_len, padding=True)
            keys.add((tuple(enc["input_ids"]), tuple(enc["attention_mask"])))
    return keys


def _canonical(result: dict) -> dict:
    """去掉只给人看的 `display`/`color`，其余字段原样保留。"""
    return {"text": result.get("text"), "num_segments": result.get("num_segments"),
            "error": result.get("error"),
            "tasks": {task: [{k: v for k, v in a.items() if k not in ("display", "color")}
                             for a in anchors]
                      for task, anchors in result.get("tasks", {}).items()}}


def _digest(result: dict) -> str:
    blob = json.dumps(_canonical(result), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _patch_cache_off(m) -> None:
    """关掉复用的第一种桩：`_run_segment` 不收缓存 ⇒ 每卡每段都重算（= 修复前的行为）。"""
    orig = MultiTaskEngine._run_segment

    def uncached(self, task, segment_text, encode_cache=None):
        return orig(self, task, segment_text, None)

    m.setattr(MultiTaskEngine, "_run_segment", uncached)


def _predict_cache_off(eng: MultiTaskEngine, monkeypatch, text: str) -> dict:
    with monkeypatch.context() as m:
        _patch_cache_off(m)
        return eng.predict(text)


def _predict_always_miss(eng: MultiTaskEngine, monkeypatch, text: str) -> dict:
    """关掉复用的第二种桩：缓存照常传，但键每次都不同 ⇒ 永远 miss、永远重算。"""
    counter = iter(range(1 << 30))
    with monkeypatch.context() as m:
        m.setattr(engine_mod, "_encode_cache_key", lambda enc: (next(counter),))
        return eng.predict(text)


def _predict_broken_key(eng: MultiTaskEngine, monkeypatch, text: str) -> dict:
    """反例：键只留长度 ⇒ 同一 max_len 的所有段互相串味（键错了的最小实现）。"""
    with monkeypatch.context() as m:
        m.setattr(engine_mod, "_encode_cache_key", lambda enc: (len(enc["input_ids"]),))
        return eng.predict(text)


class _CountingEncoder:
    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._inner(*args, **kwargs)


def _count_encodes(eng: MultiTaskEngine, monkeypatch, text: str,
                   cache_off: bool = False) -> tuple[dict, int]:
    """把 `doc_encoder` 换成计数器，返回 (predict 结果, 调用次数)。

    patch 直接落在传进来的 MonkeyPatch 上：调用方用 `with monkeypatch.context()`
    圈定作用域，两个口径（开/关缓存）互不串。
    """
    counter = _CountingEncoder(eng.doc_encoder)
    monkeypatch.setattr(eng, "doc_encoder", counter)
    if cache_off:
        _patch_cache_off(monkeypatch)
    return eng.predict(text), counter.calls


# ---- 断言 1：覆盖（≥2 段 × 全部已挂载卡，且不空转）------------------------

@pytest.mark.parametrize("text", TEXTS, ids=["A", "B"])
def test_every_card_sees_at_least_two_segments(engine, text):
    attached = engine.attached
    assert len(attached) >= 2, f"守卫只挂了 {attached} 张卡"
    for task in attached:
        assert len(_segments_for(engine, task, text)) >= 2, \
            f"{task} 只切出 1 段，'≥2 段 × 全部卡' 不成立"
    result = engine.predict(text)
    assert result["num_segments"] >= 2
    for task in attached:
        assert result["tasks"][task], f"{task}: 一条锚点都没产出，守卫空转"


# ---- 断言 2：逐位一致（开缓存 vs 关缓存）----------------------------------

@pytest.mark.parametrize("text", TEXTS, ids=["A", "B"])
def test_cache_on_is_bitwise_identical_to_cache_off(engine, monkeypatch, text):
    on = engine.predict(text)
    off = _predict_cache_off(engine, monkeypatch, text)
    assert _canonical(on) == _canonical(off), "开/关缓存的锚点字段不一致"
    assert _digest(on) == _digest(off), \
        f"开/关缓存的 sha256 不同：{_digest(on)} != {_digest(off)}"


@pytest.mark.parametrize("text", TEXTS, ids=["A", "B"])
def test_cache_on_is_bitwise_identical_to_always_miss(engine, monkeypatch, text):
    """第二种关复用的方式（缓存在、键永远 miss）也必须逐位相同。"""
    on = engine.predict(text)
    miss = _predict_always_miss(engine, monkeypatch, text)
    assert _canonical(on) == _canonical(miss)
    assert _digest(on) == _digest(miss)


# ---- 断言 3：判别力（键错了必须让守卫失败）--------------------------------

@pytest.mark.parametrize("text", TEXTS, ids=["A", "B"])
def test_wrong_key_is_caught_by_the_guard(engine, monkeypatch, text):
    """反例：把键改成"只看长度"，同 max_len 的段互相串味 ⇒ 守卫必须判不一致。"""
    off = _predict_cache_off(engine, monkeypatch, text)
    broken = _predict_broken_key(engine, monkeypatch, text)
    assert _digest(broken) != _digest(off), \
        "错键没被抓到 —— 替身解码器没有吃 doc_memory 的内容，守卫没有判别力"


# ---- 断言 4：缓存确实生效（调用次数）--------------------------------------

@pytest.mark.parametrize("text", TEXTS, ids=["A", "B"])
def test_doc_encoder_calls_drop_to_distinct_key_count(engine, monkeypatch, text):
    with monkeypatch.context() as m:
        _, calls_on = _count_encodes(engine, m, text, cache_off=False)
    with monkeypatch.context() as m:
        _, calls_off = _count_encodes(engine, m, text, cache_off=True)
    per_card = {t: len(_segments_for(engine, t, text)) for t in engine.attached}
    expect_off = sum(per_card.values())          # 卡数 × 段数
    expect_on = len(_distinct_keys(engine, text))  # 不同键数
    assert calls_off == expect_off, \
        f"关缓存 {calls_off} 次 != 卡数×段数 {expect_off}（{per_card}）"
    assert calls_on == expect_on, \
        f"开缓存 {calls_on} 次 != 不同键数 {expect_on}"
    assert calls_on < calls_off, "缓存没有省掉任何一次编码"


def test_cache_scope_is_a_single_predict_call(engine, monkeypatch):
    """作用域 = 一次 predict()：不跨调用 memoize，两次调用的编码次数一样。"""
    with monkeypatch.context() as m:
        _, first = _count_encodes(engine, m, TEXT_A)
    with monkeypatch.context() as m:
        _, second = _count_encodes(engine, m, TEXT_A)
    expect = len(_distinct_keys(engine, TEXT_A))
    assert first == expect and second == expect, \
        f"跨调用 memoize 了？第 1 次 {first} 次、第 2 次 {second} 次、应为 {expect} 次"


# ---- 断言 5：键的构成 -------------------------------------------------------

def test_encode_cache_key_covers_ids_and_mask():
    """键 = (input_ids, attention_mask)：任一项变了都必须是不同的键。"""
    base = {"input_ids": [5, 6, 0], "attention_mask": [1, 1, 0]}
    same = {"input_ids": [5, 6, 0], "attention_mask": [1, 1, 0]}
    other_ids = {"input_ids": [5, 7, 0], "attention_mask": [1, 1, 0]}
    other_mask = {"input_ids": [5, 6, 0], "attention_mask": [1, 1, 1]}
    key = engine_mod._encode_cache_key
    assert key(base) == key(same), "输入完全相同 ⇒ 同键（否则缓存永远 miss）"
    assert key(base) != key(other_ids), "input_ids 变了还同键 ⇒ 会复用错的 memory"
    assert key(base) != key(other_mask), "attention_mask 变了还同键 ⇒ 会复用错的 memory"
    assert isinstance(key(base), tuple) and len(key(base)) == 2


# ---- 仓库里的真 4 张卡（checkpoints/ 不在版本控制里，在场才跑）-------------

def _has_real_cards() -> bool:
    return (REPO / "checkpoints" / "base_encoder.pt").is_file() and bool(
        list(CARDS_DIR.glob("*.pt")))


@pytest.mark.skipif(not _has_real_cards(), reason="checkpoints/ 被 .gitignore，不在版本控制里")
def test_real_cards_cache_guard():
    """真基座 + 真 4 张卡（含 person 的 NDB 支路）：开/关缓存逐位相同、调用次数下降。"""
    eng = MultiTaskEngine(base_path=str(REPO / "checkpoints" / "base_encoder.pt"),
                          cards_dir=str(CARDS_DIR))
    assert len(eng.attached) >= 2, f"只挂了 {eng.attached}"
    for task in eng.attached:
        assert len(_segments_for(eng, task, TEXT_A)) >= 2

    with pytest.MonkeyPatch.context() as m:
        off_result, calls_off = _count_encodes(eng, m, TEXT_A, cache_off=True)
    with pytest.MonkeyPatch.context() as m:
        on_result, calls_on = _count_encodes(eng, m, TEXT_A, cache_off=False)

    assert _canonical(on_result) == _canonical(off_result), "真卡开/关缓存锚点不一致"
    assert _digest(on_result) == _digest(off_result)
    per_card = {t: len(_segments_for(eng, t, TEXT_A)) for t in eng.attached}
    assert calls_off == sum(per_card.values()), \
        f"关缓存 {calls_off} 次 != 卡数×段数 {sum(per_card.values())}（{per_card}）"
    assert calls_on == len(_distinct_keys(eng, TEXT_A))
    assert calls_on < calls_off
