"""锚点类别字段的契约守卫：`class_name` 是机器唯一该匹配的键，`display` 只给人读。

背景（缺陷本体）：锚点里曾经只发一个 `category` = `TaskClass.display`（展示名，带副标题，
如 sentiment 的「积极/喜悦」），而 `flip_rule()` 的默认 `apply_to=("积极",)` 按**规范名**写
⇒ 默认规则静默零效果，只能靠调用方自己截 `/` 前一段绕过。契约改成两个字段后，
`category` 被彻底删掉（不留别名），任何还在读它的消费者会当场 KeyError。

为什么用替身解码器：守卫要覆盖**每张已注册卡**，而仓库里只有 5 张 ckpt；
更关键的是未训练权重的 argmax 会落在背景类 ⇒ 一条锚点都不产，断言会空转。
基座走真实的 `MultiTaskEngine.__init__`（临时产物），只有网络前向被替身换掉 ——
产锚点的循环（`predict` → `_run_segment`）、tokenizer、切段策略、spec 全是真代码。
"""
from __future__ import annotations

import pytest
import torch
from nano_char_tokenizer import NanoCharTokenizer

from dtseek.encoder.nano_doc_encoder import NanoDocEncoder
from dtseek.tasks.artifacts import save_base
from dtseek.tasks.compose import flip_rule, to_items
from dtseek.tasks.engine import MultiTaskEngine
from dtseek.tasks.plugin import all_tasks

HIDDEN = 32
TEXT = "他说不难答案真棒，其实我很高兴。"


class _StubDecoder:
    """确定性替身：只吐 logits；第 k 步发第 (k mod (C-1)) + 1 类、跨度互不重复。"""

    def __init__(self, num_classes: int):
        self.bos_query = torch.zeros(1, 1, HIDDEN)
        self._c = num_classes

    def forward_step(self, q_seq, doc_memory, doc_mask=None):
        step = q_seq.shape[1] - 1
        cls = torch.full((1, self._c), -8.0)
        cls[0, step % (self._c - 1) + 1] = 8.0        # 背景类永不发射
        act = torch.full((1, 2), -8.0)
        act[0, 1] = 8.0                               # 一直 <cont>，由 max_steps 收口
        length = doc_memory.shape[1]
        s0 = step % max(1, length - 1)
        e0 = min(s0 + 1, length - 1)
        start, end = torch.full((1, length), -8.0), torch.full((1, length), -8.0)
        start[0, s0] = 8.0
        end[0, e0] = 8.0
        return {"cls_logits": cls, "action_logits": act,
                "start_logits": start, "end_logits": end,
                "last_hidden": torch.zeros(1, 1, HIDDEN)}

    def get_step_input(self, *, prev_hidden, prev_cls, prev_start, prev_end):
        return torch.zeros(1, 1, HIDDEN)


@pytest.fixture(scope="module")
def engine(tmp_path_factory) -> MultiTaskEngine:
    """真基座（临时产物）+ 每张已注册卡一个替身解码器。"""
    tok = NanoCharTokenizer()
    base = tmp_path_factory.mktemp("base") / "base_encoder.pt"
    save_base(base, NanoDocEncoder(vocab_size=tok.vocab_size, hidden_dim=HIDDEN,
                                   num_layers=1, num_heads=4, max_len=128, dropout=0.0),
              hidden_dim=HIDDEN, vocab_size=tok.vocab_size,
              encoder_kwargs={"num_layers": 1, "num_heads": 4, "max_len": 128})
    eng = MultiTaskEngine(base_path=str(base), cards_dir=None, auto_attach=False)
    for name, card in all_tasks().items():
        eng.specs[name] = card.spec
        eng.decoders[name] = _StubDecoder(card.spec.num_classes)
    return eng


def _anchors(engine: MultiTaskEngine, task: str) -> list[dict]:
    out = engine.predict(TEXT, tasks=[task])["tasks"][task]
    assert out, f"{task}: 守卫空转 —— 一条锚点都没产出"
    return out


# ---- 断言 1/2/3：class_name 合法、与 class_id 一致、没有 category ------------
@pytest.mark.parametrize("task", sorted(all_tasks()))
def test_anchor_class_fields_satisfy_contract(engine, task):
    spec = all_tasks()[task].spec
    names = {c.name for c in spec.classes}
    for a in _anchors(engine, task):
        assert "category" not in a, f"{task}: category 回退了（锚点键 {sorted(a)}）"
        assert a["class_name"] in names, f"{task}: {a['class_name']!r} 不在 spec.classes 里"
        assert spec.classes[a["class_id"]].name == a["class_name"], \
            f"{task}: class_id={a['class_id']} 与 class_name={a['class_name']!r} 不一致"
        assert a["display"] == spec.classes[a["class_id"]].display


def test_sentiment_emits_canonical_name_and_display_separately(engine):
    """缺陷现场：sentiment 的展示名带副标题，规范名必须原样出现在 class_name。"""
    spec = all_tasks()["sentiment"].spec
    assert spec.classes[1].name == "积极" and spec.classes[1].display == "积极/喜悦"
    hit = [a for a in _anchors(engine, "sentiment") if a["class_id"] == 1]
    assert hit, "替身没发出积极类，断言空转"
    assert all(a["class_name"] == "积极" and a["display"] == "积极/喜悦" for a in hit)


def test_to_items_default_label_key_is_class_name():
    """compose 侧默认取规范名 —— 默认 flip_rule 的 apply_to=("积极",) 才能命中。"""
    anchors = [{"class_name": "积极", "display": "积极/喜悦",
                "s0": 0, "e0": 1, "confidence": 0.9}]
    assert [i["label"] for i in to_items(anchors)] == ["积极"]
    assert "积极" in flip_rule()["apply_to"]
