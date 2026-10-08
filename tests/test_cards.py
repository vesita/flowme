"""cards 框架骨架的四项硬约束守卫（形状契约 / 冻结 / 快照 / MV 变更检测）。

跑法（仓根）：`.venv/bin/python -m pytest tests/test_cards.py -q`
这些用例只用小尺寸（d=16, ff=32, maxlen=64），走真代码路径，不 mock。
"""
from __future__ import annotations

import ast
import dataclasses
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]      # 仓根（tests/ 的上一层）
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cards import (CardPipeline, Contract, InputCard, OutputCard, TargetCard,  # noqa: E402
                   load_snapshot, make_mv, snapshot_is_readonly)
from cards import blackboard as bb_mod  # noqa: E402
from cards import version as ver_mod  # noqa: E402

D, V, FF, NH, MAXLEN = 16, 64, 32, 2, 64


def tiny_pipeline(maxlen: int = MAXLEN, residual: bool = True) -> CardPipeline:
    return CardPipeline(d=D, vocab_size=V, ff=FF, nhead=NH, maxlen=maxlen, pad_id=0,
                        residual=residual)


def tiny_ids(b: int = 3, n: int = 10) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1, V, (b, n), generator=g)
    ids[0, -2:] = 0                                     # 造 padding，走 mask 的 padding 分支
    return ids


# ---------------------------------------------------------------- ① 形状契约
def test_shape_contract_each_card():
    """三类卡各自的形状契约：ids→[n,d] / [n,d]→[n,d]（逐字相等）/ [n,d]→logits。"""
    ids = tiny_ids()
    inp = InputCard(D, V, FF, NH, MAXLEN, pad_id=0)
    tgt = TargetCard(D, FF, NH, MAXLEN)
    out = OutputCard(D, V, MAXLEN)

    h = inp(ids)
    assert h.shape == (3, 10, D)                        # ids (b,n) → (b,n,d)
    h2 = tgt(h)
    assert h2.shape == h.shape                          # 形状保持
    logits = out(h2)
    assert logits.shape == (3, 10, V)

    # d->d 卡无 batch 时 = 逐字断言 x.shape == out.shape == (n, d)
    flat = torch.randn(7, D)
    assert tgt(flat).shape == flat.shape == (7, D)


def test_shape_contract_violation_raises():
    """形状契约是硬约束：违反必须当场 AssertionError（不是等着 torch 报形状错）。"""
    tgt = TargetCard(D, FF, NH, MAXLEN)
    with pytest.raises(AssertionError, match="输入最后两维"):
        tgt(torch.randn(7, D + 1))
    with pytest.raises(AssertionError, match="输入最后两维"):
        OutputCard(D, V, MAXLEN)(torch.randn(7, D - 1))
    with pytest.raises(AssertionError, match="int token ids"):
        InputCard(D, V, FF, NH, MAXLEN, pad_id=0)(torch.randn(3, 10, D))

    class BadTarget(TargetCard):
        """故意不保持形状的实现：契约必须在输出侧接住它。"""

        def apply_card(self, x, mask=None):
            return x[:, :-1]

    with pytest.raises(AssertionError, match=r"形状保持要求 x\.shape == out\.shape == \(n, d\)"):
        BadTarget(D, FF, NH, MAXLEN)(torch.randn(7, D))
    with pytest.raises(AssertionError, match="形状保持要求 out.shape == x.shape"):
        BadTarget(D, FF, NH, MAXLEN)(torch.randn(2, 7, D))


def test_blackboard_records_who_reads_whom():
    """黑板显式记录「谁读了谁」，本轮路由 = input → target → output。"""
    p = tiny_pipeline()
    p(tiny_ids())
    assert p.who_reads_whom() == [("input", bb_mod.EXTERNAL), ("target", "input"),
                                  ("output", "target")]
    keys = [(r.reader, r.key, r.writer) for r in p.blackboard.read_log]
    assert ("target", "meme", "input") in keys
    assert ("output", "meme", "target") in keys
    assert all(r.version for r in p.blackboard.read_log)  # 每次读都带写入方版本


# ------------------------------------------------------------------- ② 冻结
def test_freeze_keeps_frozen_params_bitwise_identical():
    """pipeline.freeze(["input","output"]) 后被冻结卡 max|Δθ| == 0（精确），目标卡在动。"""
    p = tiny_pipeline()
    ids = tiny_ids()
    assert p.freeze(["input", "output"]) == ("input", "output")
    assert p.input.is_frozen and p.output.is_frozen and not p.target.is_frozen
    before = p.theta()

    p.train()                                            # 冻结卡不许被 train() 拉回 dropout
    assert not p.input.training and not p.output.training
    opt = torch.optim.AdamW(p.trainable_parameters(), lr=1e-2)
    for _ in range(3):
        loss = p(ids).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    delta = p.assert_frozen_unchanged(before)            # 内部就是 max|Δθ| == 0.0 的断言
    assert delta["input"] == 0.0 and delta["output"] == 0.0
    assert delta["target"] > 0.0                         # 只训目标卡，且它确实动了


# ------------------------------------------------------------------- ③ 快照
def test_snapshot_readonly_and_reproducible(tmp_path):
    """snapshot() 只读 + sha256；同权重两次快照 sha 相同；换个管线重载权重 sha 仍相同。"""
    p = tiny_pipeline()
    s1 = p.snapshot(tmp_path / "mv_a.json")
    s2 = p.snapshot(tmp_path / "mv_b.json")
    assert s1.sha256 == s2.sha256                        # 同权重两次 ⇒ 同 sha
    assert s1.mv == s2.mv
    assert snapshot_is_readonly(s1.path)                 # 只读文件
    import os
    assert not (os.stat(s1.path).st_mode & 0o222)
    assert load_snapshot(s1.path).sha256 == s1.sha256    # 只有 sha 对得上才读得回来

    q = tiny_pipeline()
    q.load_state_dict(p.state_dict())
    assert q.snapshot(tmp_path / "mv_c.json").sha256 == s1.sha256

    # 篡改内容 ⇒ 校验当场炸
    path = tmp_path / "tampered.json"
    path.write_bytes(pathlib.Path(s1.path).read_bytes().replace(b"placeholder", b"placehold3r"))
    with pytest.raises(AssertionError):
        load_snapshot(path)


# ------------------------------------------------------- ④ MV 变更检测
def test_mv_changes_when_any_contract_field_changes():
    """契约里任一字段变（d / n 语义 / mask / 位置编码 / MAXLEN / 契约版本）⇒ MV 必变。"""
    p = tiny_pipeline()
    base_sha = p.mv().sha256()
    c: Contract = p.contract
    for field, value in (("d", c.d + 1), ("n_semantics", c.n_semantics + "（改）"),
                         ("mask", c.mask + "（改）"), ("pe", c.pe + "（改）"),
                         ("maxlen", c.maxlen * 2), ("version", c.version + 1)):
        c2 = dataclasses.replace(c, **{field: value})
        # 接口快照 sha256 把契约算进去 ⇒ 契约一动，MV 必动
        mv = make_mv(c2, ver_mod.interface_sha256(p.cards(), c2))
        assert mv.sha256() != base_sha, f"改了契约字段 {field} 但 MV 没变"

    # 端到端：只改 MAXLEN ⇒ 管线 MV 变（接口快照 sha 直接变）
    short = tiny_pipeline(maxlen=MAXLEN // 2)
    assert short.interface_sha256() != p.interface_sha256()
    assert short.mv().sha256() != base_sha
    assert short.contract.maxlen == MAXLEN // 2 and p.contract.maxlen == MAXLEN


def test_mv_detects_weight_drift_but_ignores_target_card():
    """接口权重动一位 ⇒ MV 变；只动目标卡（实现）⇒ MV 不变（接口 = 输入卡 + 输出卡）。"""
    p = tiny_pipeline()
    base = p.mv().sha256()
    with torch.no_grad():
        p.input.emb.weight[1, 0] += 1e-6
    assert p.mv().sha256() != base

    q = tiny_pipeline()
    q.load_state_dict(p.state_dict())
    with torch.no_grad():
        q.target.think.linear1.weight[0, 0] += 1e-3
    assert q.mv().sha256() == p.mv().sha256()


# ------------------------------------------------------ 独立性（不 import dtseek）
def test_cards_package_does_not_import_dtseek():
    """cards/ 必须独立：源码里不许出现 dtseek（也不许 import stages/）。"""
    for src in sorted((ROOT / "cards").glob("*.py")):
        tree = ast.parse(src.read_text(encoding="utf-8"))
        mods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods.append(node.module or "")
        assert not any("dtseek" in m or "stages" in m for m in mods), (src.name, mods)
    assert ver_mod.W_VERSION and ver_mod.MV(interface_sha256="x").short()
