"""cards 框架骨架的四项硬约束守卫（形状契约 / 冻结 / 快照 / MV 变更检测）。

跑法（仓根）：`.venv/bin/python -m pytest tests/test_cards.py -q`
这些用例只用小尺寸（d=16, ff=32, maxlen=64），走真代码路径，不 mock。
"""
from __future__ import annotations

import ast
import dataclasses
import math
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
from cards.example_attribution import (IdentityCard, iv_all, iv_ident,  # noqa: E402
                                       top_share, with_intervention)

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


# ================================================ ⑤ 可归因（卖点①：整块可替换/可消融）
def test_identity_card_dtod_and_pipeline_swap_keeps_mv():
    """IdentityCard：io=d->d / 零参数 / 形状保持；换掉目标卡=换实现 ⇒ MV 不变、行为变。"""
    ic = IdentityCard(D, MAXLEN)
    assert ic.io == "d->d" and ic.n_params() == 0
    flat = torch.randn(7, D)
    assert torch.equal(ic(flat), flat)                   # 恒等：逐位相同
    with pytest.raises(AssertionError, match="输入最后两维"):
        ic(torch.randn(7, D + 1))

    p = tiny_pipeline()
    p.eval()
    ids = tiny_ids()
    mv0 = p.mv()
    with torch.no_grad():
        y0 = p(ids).clone()
    p.target = IdentityCard(D, MAXLEN)                   # ★门 B：整块替换目标卡
    assert p.mv() == mv0                                 # 接口 = input + output ⇒ MV 不动
    with torch.no_grad():
        y1 = p(ids)
    assert not torch.equal(y0, y1)                       # 实现换了 ⇒ 行为确实变
    assert p.who_reads_whom() == [("input", bb_mod.EXTERNAL), ("target", "input"),
                                  ("output", "target")]


def test_intervention_hook_is_non_destructive_and_identity_exact():
    """干预钩子挂在目标卡输入侧（S21/S18 side=in）：零干预逐位不变，消融真的改变被测量。"""
    p = tiny_pipeline()
    p.eval()
    ids = tiny_ids()
    with torch.no_grad():
        base = p(ids).clone()
    same = with_intervention(p, iv_ident(), lambda: None) is None      # 注册即摘除
    assert same
    with torch.no_grad():
        got = with_intervention(p, iv_ident(), lambda: p(ids))
    assert torch.equal(got, base)                        # ★零干预恒等：Δ 精确为 0
    with torch.no_grad():
        zeroed = with_intervention(p, iv_all(), lambda: p(ids))
    assert not torch.equal(zeroed, base)                 # ★R29：干预非空、真的改了读数
    with torch.no_grad():
        after = p(ids)                                   # 钩子已摘除
    assert torch.equal(after, base)                      # 非破坏性：不动权重、不留副作用


def test_top_share_ranks_units_by_effect():
    """前 20% 因果占比（S21 curve_stats 口径）：按 |Δ| 降序累积。"""
    assert top_share([10.0, 0.0, 0.0, 0.0, 0.0], 0.2) == 1.0
    assert abs(top_share([1.0] * 5, 0.2) - 0.2) < 1e-12
    assert math.isnan(top_share([0.0] * 4, 0.2))
    assert top_share([5.0, 4.0, 0.0, 0.0], 0.5) == 1.0   # ceil(0.5*4)=2 ⇒ 前两名


# ============================================ ⑥ 硬版本化（卖点②：实现可换、版本可判）
def test_versioning_contract_change_yields_new_mv(tmp_path):
    """同一份接口权重：换 MAXLEN ⇒ 换 MV；换目标卡（实现）⇒ MV 不变；卡快照只读可复核。"""
    from cards.example_versioning import PREF, load_pipeline

    p = tiny_pipeline()
    rev = {v: k for k, v in PREF.items()}
    sd = {}
    for k, t in p.state_dict().items():
        for dst, src in rev.items():
            if k.startswith(dst):
                sd[src + k[len(dst):]] = t
                break
    q = load_pipeline(sd, MAXLEN, ff=FF, nhead=NH, pad_id=0)
    assert q.mv() == p.mv()                              # 同接口权重 + 同契约 ⇒ 同 MV
    qs = load_pipeline(sd, MAXLEN // 2, ff=FF, nhead=NH, pad_id=0)
    assert qs.mv().sha256() != q.mv().sha256()           # ★换契约一项 ⇒ 换 MV
    q.target = IdentityCard(D, MAXLEN)                   # ★换实现 ⇒ MV 不变
    assert q.mv() == p.mv()

    s = ver_mod.snapshot(q.input, tmp_path / "q_input.json")   # 卡级只读快照
    assert snapshot_is_readonly(s.path)
    assert load_snapshot(s.path).sha256 == s.sha256


# ============================================ ⑦ 两个新演示脚本的规格守卫（源文本级）
def test_example_add3d_saturation_default_and_two_stream_rng():
    """★默认 6000 = E6 饱和点口径；★R34 两条随机流结构 + 两栏报告 + E6 锚不许被改掉。"""
    src = (ROOT / "cards" / "example_add3d.py").read_text(encoding="utf-8")
    assert 'os.environ.get("CARDS_STEPS", "6000")' in src
    assert 'os.environ.get("CARDS_RNG", "s19-two-stream")' in src
    assert "DROP_GEN.set_state(torch.random.get_rng_state())" in src
    assert "0.85625" in src and "0.74375" in src         # E6 res 臂锚（seed1234/5678）
    assert "[TABLE]" in src and "是否饱和" in src         # @3000/@6000 两栏 + 饱和标注


def test_example_attribution_retrains_gate_b_under_r16():
    """门 B 必须是**重训**（R16），不是推理期破坏；且要报前 20% 因果占比 + R29/R28。"""
    src = (ROOT / "cards" / "example_attribution.py").read_text(encoding="utf-8")
    assert "IdentityCard" in src and "ex.masked_ce(idpipe" in src
    assert "top_share" in src and "前 10/20/50%" in src
    assert "[R29]" in src and "[R28]" in src
