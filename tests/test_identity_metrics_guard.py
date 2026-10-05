"""身份指标的 fail-closed 守卫：`identity_labels=True` 就必须给出实数。

背景（缺陷本体）：`TaskSpec.to_snapshot/from_snapshot` 曾经不序列化 `identity_labels`
与 `annotate_all`，`build_card_decoder` 用快照重建 spec 时这两个字段悄悄变回 False，
于是 `evaluate_task` 连身份指标的键都不产生 —— 下游 `.get(k, float("nan"))` 拿到一列
NaN 且不报错。守卫写在 `runtime._assert_identity_metrics`（唯一实现点），不只写在测试里。
"""
import math
from dataclasses import replace

import pytest
import torch
from nano_char_tokenizer import NanoCharTokenizer
from torch.utils.data import DataLoader

from dtseek.decoder.robust_ar_model import RobustARSliceDecoder
from dtseek.encoder.nano_doc_encoder import NanoDocEncoder
from dtseek.tasks.artifacts import build_card_decoder, read_card, save_card
from dtseek.tasks.plugin import all_tasks
from dtseek.tasks.runtime import (
    IDENTITY_METRICS,
    GenericTaskDataset,
    IdentityMetricsUnavailable,
    _assert_identity_metrics,
    evaluate_task,
)


@pytest.fixture(scope="module")
def person_eval(tmp_path_factory):
    """一张真实落盘再读回的 person 卡 + 冻结基座 + 验证 loader（CPU，未训练权重）。"""
    card = all_tasks()["person"]
    spec = card.spec
    torch.manual_seed(0)
    decoder = RobustARSliceDecoder(hidden_dim=32, num_classes=spec.num_classes,
                                   num_heads=4, num_layers=1)
    path = tmp_path_factory.mktemp("card") / "person.pt"
    save_card(path, decoder, task="person", spec=spec, hidden_dim=32,
              decoder_kwargs={"num_heads": 4, "num_layers": 1})

    ck = read_card(path)
    dec, snap_spec = build_card_decoder(ck, "cpu")

    tok = NanoCharTokenizer()
    enc = NanoDocEncoder(vocab_size=tok.vocab_size, hidden_dim=32, num_layers=1,
                         num_heads=4, max_len=128, dropout=0.0)
    loader = DataLoader(GenericTaskDataset(card.build_dataset(64), tok, snap_spec),
                        batch_size=32)
    return {"card": card, "snap_spec": snap_spec, "dec": dec, "enc": enc, "loader": loader}


def test_ckpt_rebuilt_person_card_reports_non_nan_identity_metrics(person_eval):
    """② ckpt 重建的 person 卡跑 evaluate_task，身份指标必须在场且非 NaN（修复前整块缺失）。"""
    spec = person_eval["snap_spec"]
    assert spec.identity_labels is True and spec.annotate_all is True

    rep = evaluate_task(person_eval["enc"], person_eval["dec"], person_eval["loader"],
                        torch.device("cpu"), spec)
    missing = [k for k in IDENTITY_METRICS if k not in rep]
    nan = [k for k in IDENTITY_METRICS if not math.isfinite(rep.get(k, float("nan")))]
    assert not missing, f"身份指标键缺失：{missing}（报告键：{sorted(rep)}）"
    assert not nan, f"身份指标为 NaN/非有限：{nan}"


def test_guard_raises_when_snapshot_loses_identity_flag(person_eval):
    """守卫 1：注册表说 identity_labels=True、本次评估用的 spec 是 False ⇒ 直接抛错。"""
    lost = replace(person_eval["snap_spec"], identity_labels=False)
    with pytest.raises(IdentityMetricsUnavailable, match="identity_labels"):
        evaluate_task(person_eval["enc"], person_eval["dec"], person_eval["loader"],
                      torch.device("cpu"), lost)


def test_guard_raises_on_missing_or_nan_identity_metrics():
    """守卫 2：spec 说 True，但报告里键缺失或值为 NaN ⇒ 直接抛错。"""
    spec = all_tasks()["person"].spec
    with pytest.raises(IdentityMetricsUnavailable, match="缺失"):
        _assert_identity_metrics({}, spec)
    broken = {k: 0.5 for k in IDENTITY_METRICS}
    broken["cluster_f1"] = float("nan")
    with pytest.raises(IdentityMetricsUnavailable, match="NaN"):
        _assert_identity_metrics(broken, spec)
    # 反向：全是实数就不许抛
    _assert_identity_metrics({k: 0.0 for k in IDENTITY_METRICS}, spec)
