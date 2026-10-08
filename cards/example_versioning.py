#!/usr/bin/env python3
"""example_versioning —— 卡片化的**第二个真卖点：硬版本化**（接口冻结 ⇒ 实现可换、版本可判）。

背景（E1/S35）：卡边界【不产生能力】，所以卡片化真正的交付是
①可归因（见 `cards/example_attribution.py`）②**可版本化 / 可组合**（本文件）。
E1 的直接推论：**能力住在卡内结构里，可复用的是卡的接口** —— 接口一旦冻结成只读快照，
同一个模因版本（MV）就可以容纳任意多个不同的目标卡实现，而版本号仍可判定。

四步（全程 CPU；除 ③ 的目标卡重训外零训练）：

  ① 把**输入卡 / 输出卡**冻结为【只读快照】（chmod 0o444，追加写入必须 PermissionError），
     记录每个快照文件的 sha256 与卡级 snapshot sha256；
  ② 算 **MV = (接口快照 sha256, 契约版本, W@v0-placeholder)** 并打印，
     校验它 == 只读快照载荷里落盘的 MV（快照被改过 ⇒ `load_snapshot` 当场炸）；
  ③ 用**同一个 MV** 训两张**不同的目标卡**（不同 seed，各自 ≤600 步 = 任务硬约束）
     ⇒ 各报 EM，且**两张卡训完后 MV 都仍等于 base MV**（实现换了、版本没换）；
  ④ **换契约一项（MAXLEN）⇒ 必须得到不同 MV**（并演示「假同版本」被 `Card.load` 拒绝）。

env：CARDS_VER_CKPT / CARDS_VER_STEPS（默认 600，上限 600）/ CARDS_VER_N_EM（默认 200）
     / CARDS_VER_SEEDS（默认 1234,5678）/ CARDS_THREADS
运行：`.venv/bin/python cards/example_versioning.py`（**CPU**）
只写 /tmp（快照目录）；不写 logs/、不 import stages/。
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch                                            # noqa: E402
from torch.optim import AdamW                           # noqa: E402

from cards import (CardPipeline, InputCard, TargetCard, load_snapshot,  # noqa: E402
                   make_mv, snapshot_is_readonly)
from cards import version as ver_mod                    # noqa: E402

PREF = {"emb.": "input.emb.", "in_enc.": "input.enc.",
        "thought.": "target.think.", "head.": "output.head."}
STEPS_CAP = 600          # ★任务硬约束：必须训时 ≤600 步


def load_pipeline(sd: dict, maxlen: int, *, ff: int, nhead: int, pad_id: int) -> CardPipeline:
    """把 S19/S36 容器（emb./in_enc./thought./head.）的权重映射进 CardPipeline。

    maxlen 可指定 ⇒ 用来演示「换契约一项 ⇒ 换 MV」（同一份接口权重、不同契约）。
    """
    mapped = {}
    for k, v in sd.items():
        for src, dst in PREF.items():
            if k.startswith(src):
                mapped[dst + k[len(src):]] = v
                break
    d = int(sd["emb.weight"].shape[1])
    vocab = int(sd["emb.weight"].shape[0])
    pipe = CardPipeline(d=d, vocab_size=vocab, ff=ff, nhead=nhead, maxlen=maxlen,
                        pad_id=pad_id, dropout=0.1, residual=True)
    _missing, unexpected = pipe.load_state_dict(mapped, strict=False)
    assert not unexpected, f"ckpt 有本管线不认的键：{unexpected[:3]}"
    return pipe


def train_target(pipe: CardPipeline, train, steps: int, seed: int, lr: float, batch: int,
                 drop_gen) -> float:
    """只训目标卡（接口已 freeze）；★R34：两条随机流结构与 example_add3d 一致。"""
    import cards.example_add3d as ex
    torch.manual_seed(seed)
    drop_gen.manual_seed(seed)
    before = pipe.theta()
    opt = AdamW(pipe.trainable_parameters(), lr=lr)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    step, pos, t0 = 0, 0, time.time()
    while step < steps:
        idx = []
        for _ in range(batch):
            if pos >= len(order):
                order = torch.randperm(len(train))
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        loss = ex.masked_ce(pipe, [train[i] for i in idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 200 == 0 or step == steps:
            print(f"  [ver-③ train seed={seed}] step={step}/{steps} loss={loss.item():.4f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    delta = pipe.assert_frozen_unchanged(before)          # ★冻结的接口卡必须逐位没动
    assert delta["input"] == 0.0 and delta["output"] == 0.0
    return time.time() - t0


def main() -> int:
    t0 = time.time()
    import cards.example_add3d as ex

    ckpt = Path(os.environ.get("CARDS_VER_CKPT",
                               str(ROOT / "logs" / "36_ckpt_res_s6000_seed1234.pt")))
    steps = min(int(os.environ.get("CARDS_VER_STEPS", "600")), STEPS_CAP)
    n_em = int(os.environ.get("CARDS_VER_N_EM", "200"))
    seeds = tuple(int(s) for s in os.environ.get("CARDS_VER_SEEDS", "1234,5678").split(","))
    tmp = Path(tempfile.mkdtemp(prefix="cards_ver_"))
    kw = dict(ff=ex.FF, nhead=ex.NHEAD, pad_id=ex.PAD_ID)

    print(f"[VER] ===== 卡片化卖点②：硬版本化 | device={ex.DEVICE} ckpt={ckpt.name} | "
          f"MAXLEN={ex.MAXLEN} d={ex.D} | 目标卡重训步数={steps}（上限 {STEPS_CAP}）"
          f"seeds={seeds} | tmp={tmp} =====", flush=True)

    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    train, test = ex.build_data()
    em_recs = test[:n_em]

    # ========================================================================
    # ① 冻结接口为只读快照
    # ========================================================================
    print("\n[VER-①] ===== 冻结输入卡/输出卡为只读快照（chmod 0o444）+ 记录 sha256 =====",
          flush=True)
    pipe = load_pipeline(sd, ex.MAXLEN, **kw)
    pipe.freeze(["input", "output"])
    rows = []
    for name in ("input", "output"):
        card = pipe.card(name)
        s = ver_mod.snapshot(card, tmp / f"{name}.json")     # 卡级只读快照
        ro = snapshot_is_readonly(s.path) and not (os.stat(s.path).st_mode & 0o222)
        reopened = load_snapshot(s.path)
        rows.append(dict(card=name, path=str(s.path), file_sha256=s.sha256,
                         card_sha256=card.snapshot_sha256(),
                         readonly=bool(ro), reopen_ok=bool(reopened.sha256 == s.sha256)))
        print(f"[VER-①] {name}卡快照 {s.path} | 文件sha256={s.sha256[:16]} | "
              f"卡级snapshot_sha256={card.snapshot_sha256()[:16]} | 只读位0o222="
              f"{0 if ro else '★非0'} ⇒ {'只读 ✓' if ro else '★未只读'} | "
              f"load_snapshot 复核sha相同={rows[-1]['reopen_ok']}", flush=True)
    try:                                                 # 硬只读：写它必须失败
        with open(rows[0]["path"], "ab") as f:
            f.write(b"x")
        tamper = "★竟然写进去了（未只读！）"
    except PermissionError as e:
        tamper = f"PermissionError ✓（{e.strerror or 'Permission denied'}）"
    print(f"[VER-①] 只读快照追加写入测试 ⇒ {tamper}", flush=True)

    snap = pipe.snapshot(tmp / "pipeline_mv.json")
    print(f"[VER-①] 管线接口快照 {snap.path} | 文件sha256={snap.sha256[:16]} | "
          f"只读={snapshot_is_readonly(snap.path)}", flush=True)

    # ========================================================================
    # ② MV = (接口快照 sha256, 契约版本, W 版本占位)
    # ========================================================================
    print("\n[VER-②] ===== 算 MV =====", flush=True)
    mv = pipe.mv()
    iface = pipe.interface_sha256()
    from_snap = load_snapshot(snap.path)
    mv_ok = (from_snap.mv == mv) and (snap.payload["interface_sha256"] == iface)
    print(f"[VER-②] 接口快照 sha256 = {iface}", flush=True)
    print(f"[VER-②] 契约版本 = {mv.contract_version} | W 版本 = {mv.w_version} | "
          f"MV = {json.dumps(mv.to_dict(), ensure_ascii=False, sort_keys=True)} | "
          f"MV.sha256 = {mv.sha256()}", flush=True)
    print(f"[VER-②] MV.short = {mv.short()} | 与只读快照载荷逐字段相同={mv_ok} "
          f"⇒ {'快照即版本 ✓' if mv_ok else '★不一致'}", flush=True)

    # ========================================================================
    # ③ 同一个 MV 训两张不同的目标卡（实现可换、版本不变）
    # ========================================================================
    print(f"\n[VER-③] ===== 同一个 MV 下训 {len(seeds)} 张不同目标卡（各 {steps} 步）=====",
          flush=True)
    per_seed, wshas_init = {}, set()
    for seed in seeds:
        torch.manual_seed(seed)
        p = load_pipeline(sd, ex.MAXLEN, **kw)
        p.target = TargetCard(ex.D, ex.FF, ex.NHEAD, ex.MAXLEN, dropout=0.1, residual=True)
        wshas_init.add(p.target.weights_sha256())
        p.freeze(["input", "output"])
        mv_before = p.mv()
        wall = train_target(p, train, steps, seed, ex.LR, ex.BATCH, ex.DROP_GEN)
        mv_after = p.mv()
        r = ex.em_score(p, em_recs)
        d = ex.digit_report(p, test)
        per_seed[seed] = dict(em=r["em"], em_se=r["se"], dig=d["acc"], wall=wall,
                              target_sha16=p.target.weights_sha256()[:16],
                              iface_sha16=p.interface_sha256()[:16])
        same_as_base = (mv_after == mv) and (mv_after == mv_before)
        print(f"[VER-③] target@seed{seed}: EM(batch=1,n={len(em_recs)})={r['em']*100:.2f}%±"
              f"{r['se']*100:.2f} | 数字每步={d['acc']*100:.2f}% | 训 {steps} 步 "
              f"墙钟={wall/60:.2f}min | 目标卡权重sha16={p.target.weights_sha256()[:16]} | "
              f"MV 训前/训后 == base MV：{same_as_base} ⇒ "
              f"{'实现换了、版本没换 ✓' if same_as_base else '★MV 变了'}", flush=True)
    print(f"[VER-③] 两张目标卡初值不同={len(wshas_init) == len(seeds)}"
          f"（{len(wshas_init)}/{len(seeds)} 个不同 weights_sha256）| 各自 EM=" +
          " / ".join(f"seed{s}:{per_seed[s]['em']*100:.2f}%" for s in seeds), flush=True)

    # ========================================================================
    # ④ 换契约一项 ⇒ MV 必变（假同版本防护）
    # ========================================================================
    print("\n[VER-④] ===== 换契约一项（MAXLEN）⇒ 必须得到不同 MV =====", flush=True)
    short = load_pipeline(sd, ex.MAXLEN // 2, **kw)
    short.freeze(["input", "output"])
    mv2 = short.mv()
    print(f"[VER-④] MAXLEN {ex.MAXLEN} → {short.contract.maxlen}：接口 sha256 "
          f"{iface[:16]} → {short.interface_sha256()[:16]} | MV.sha256 {mv.sha256()[:16]} → "
          f"{mv2.sha256()[:16]} ⇒ {'不同 ✓（换契约即换版本）' if mv2.sha256() != mv.sha256() else '★竟然相同'}",
          flush=True)
    c0 = pipe.contract
    changed, unchanged = [], []
    for field, value in (("d", c0.d + 1), ("n_semantics", c0.n_semantics + "（改）"),
                         ("mask", c0.mask + "（改）"), ("pe", c0.pe + "（改）"),
                         ("maxlen", c0.maxlen * 2), ("version", c0.version + 1)):
        c2 = dataclasses.replace(c0, **{field: value})
        mv_f = make_mv(c2, ver_mod.interface_sha256(pipe.cards(), c2))
        (changed if mv_f.sha256() != mv.sha256() else unchanged).append(field)
    print(f"[VER-④] 契约六字段逐个试：MV 变的={changed} | 不变的={unchanged or '无'} ⇒ "
          f"{'六个字段都进 MV ✓' if not unchanged else '★有字段没进 MV'}", flush=True)

    card_path = pipe.input.save(tmp / "input_card.pt")   # 契约写进快照载荷
    wrong = InputCard(ex.D, ex.V, ex.FF, ex.NHEAD, ex.MAXLEN // 2, pad_id=ex.PAD_ID)
    try:
        wrong.load(card_path)
        guard = "★竟然加载成功（假同版本没被挡住！）"
    except AssertionError as e:
        guard = f"AssertionError ✓（{str(e).splitlines()[0].strip()}）"
    print(f"[VER-④] 「假同版本」防护：把 MAXLEN={ex.MAXLEN} 的 input 卡快照灌进 "
          f"MAXLEN={ex.MAXLEN // 2} 的卡 ⇒ {guard}", flush=True)

    out = Path("/tmp") / "cards_versioning_result.json"
    out.write_text(json.dumps(dict(
        ckpt=ckpt.name, maxlen=ex.MAXLEN, steps=steps, seeds=list(seeds),
        interface_sha256=iface, mv=mv.to_dict(), mv_sha256=mv.sha256(),
        snapshots=rows, pipeline_snapshot=snap.path, pipeline_snapshot_sha256=snap.sha256,
        per_seed={str(s): v for s, v in per_seed.items()},
        mv_changed_by_maxlen=bool(mv2.sha256() != mv.sha256()),
        contract_fields_changing_mv=changed,
        wall_min=(time.time() - t0) / 60), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[VER] 结果落 {out} | 总墙钟={(time.time()-t0)/60:.1f}min exit=0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
