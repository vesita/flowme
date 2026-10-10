"""harness.probe —— 推理探针：卡数 N / 接口宽度 w / dtype 扫描 → 驻留·延迟·保真。

对应 L3_PLAN.md §19（R7-b/c/d/e）与 §18.2 方向 III/VI。**只跑前向**：无优化器、无反向。

协议（写死，跨配置恒定，逐条打印 + 指纹）
  · 同一探测批（`--data` 的前 `--batch` 条，ctx-only；跨配置**逐位相同**，报 sha256 指纹）；
  · `--batch 32`、`--maxlen 512`（补 PAD 到定长）；
  · dtype 单列（`--dtype fp32` 主档；fp16/bf16 只作另一档，绝不混在同一对比里）。

两种口径（★别混）
  · **分配（活载荷峰值）**：在每个卡边界遍历 `Packet` 里**实际存在的张量对象**求字节和，取全流程峰值
    （含 mask / h / logits / aux）。确定性：只看 shape/dtype，不看时间与分配器。
    它**不是** `torch.cuda.max_memory_allocated`（CPU 无该口径）⇒ 是下界。
  · **预留**：
      - CPU 档 = 该配置在**独立子进程**里的进程峰值 RSS（`VmHWM`）。分辨率 ~1 MiB，
        比"一张层卡的参数字节（~0.8 MB）"粗 ⇒ 跨 N 的差别**测不出来**（如实报，不写成"没有差别"）；
      - CUDA 档（`--device cuda`）才有真分配器双量：`max_memory_allocated` / `max_memory_reserved`。
  · **驻留预测**（明确标为**推断**，不是实测）：`驻留_顺序 = max(单卡参数) + 接口 + 活载荷`，
    `驻留_全常驻 = Σ卡参数 + 接口 + 活载荷`。本探针**没有实现磁盘 paging**，
    顺序执行是它的**极限情形**（§19 的诚实边界）。

保真（R7-a）：顺序执行（N 张卡逐个跑）vs **整体图**（同一批权重压成一个 `enc_stack`）
  ⇒ 报 `max|Δ|` 与"逐条相同"，这是「动态加载不改语义」的前提。

确定性口径：`--no-measure` 时 stdout 逐字节可复现（Tier 1 全是形状/参数字节算术）；
`--measure`（默认）会追加 Tier 2（墙钟 / RSS），**这些字段逐次会变**，`[DET]` 行会说明。
"""
from __future__ import annotations

import hashlib
import os
import statistics
import subprocess
import sys
import time

import torch

from harness.pipeline import (D, DROPOUT, DATA_TEST_K8, FF, LOG_DIR, MAXLEN, Meter, NHEAD, Pipeline,
                              build_custom, load_jsonl, pack, pad_id, param_bytes, prep_record,
                              tensor_digest, vocab_size)

CKPT_DEFAULT = os.path.join(LOG_DIR, "79_ckpt_B2_k8_s1234.pt")
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


# ---------------------------------------------------------------------------
# 探测批（跨配置逐位相同）
# ---------------------------------------------------------------------------
def probe_batch(data: str = DATA_TEST_K8, batch: int = 32, maxlen: int = MAXLEN):
    """取前 batch 条 ctx-only，右补 PAD 到 maxlen；返回 (ids, ctx_len, fingerprint)。"""
    from harness.pipeline import pad_id
    recs = load_jsonl(data)[:batch]
    for r in recs:
        prep_record(r, with_target=False)
    ids, s, eidx, nc = pack(recs, with_target=False, device="cpu")
    if ids.size(1) < maxlen:
        pad = torch.full((ids.size(0), maxlen - ids.size(1)), pad_id(), dtype=ids.dtype)
        ids = torch.cat([ids, pad], dim=1)
    else:
        ids = ids[:, :maxlen]
    h = hashlib.sha256()
    h.update(bytes(str(tuple(ids.shape)), "utf-8"))
    h.update(ids.numpy().tobytes())
    fp = h.hexdigest()[:16]
    return ids, s, fp, [r["ctx"] for r in recs]


# ---------------------------------------------------------------------------
# 组装：顺序（N 张卡） 与 整体图（1 张 stack 卡）
# ---------------------------------------------------------------------------
def build_seq(n: int, d: int = D, dtype=torch.float32, device: str = "cpu",
              weights: str = "trained", ckpt: str = CKPT_DEFAULT, seed: int = 1234) -> Pipeline:
    layers = [("input", "input")] + [(f"enc{i}", "enc_layer") for i in range(n)] + [("output", "output")]
    pipe = build_custom(layers, seed=seed, device=device, d=d)
    if weights == "trained":
        _load_layers_from_ckpt(pipe, ckpt, n)
    if dtype != torch.float32:
        pipe = pipe.to(dtype=dtype)
    pipe.eval()
    return pipe


def _load_layers_from_ckpt(pipe: Pipeline, ckpt: str, n: int) -> dict:
    """把 79 的 `select.enc.*` / `realign.enc.*` 交替灌进 N 张层卡（我们只有这两种同尺寸层卡）。"""
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "state" in blob and "format" in blob:
        blob = blob["state"]
    srcs = ["select.", "realign."]
    used = {}
    for i in range(n):
        pre = srcs[i % 2]
        tgt = pipe.cards[f"enc{i}"]
        keep = set(tgt.state_dict())
        # 前缀照 registry 的口径剥（`select.enc.*` → `enc.*`）；层卡没有打分向量 q ⇒ 丢掉它
        local = {k[len(pre):]: v for k, v in blob.items() if k.startswith(pre)}
        local = {k: v for k, v in local.items() if k in keep}
        tgt.load_state_dict(local, strict=True)
        used[f"enc{i}"] = pre
    # input / output 卡也从同一 ckpt 灌（保持"同构真实权重"）
    for iid, pre in (("input", "input."), ("output", "output.")):
        local = {k[len(pre):]: v for k, v in blob.items() if k.startswith(pre)}
        if local:
            pipe.cards[iid].load_state_dict(local, strict=True)
    return used


def build_fused(n: int, seq: Pipeline, dtype=torch.float32, device: str = "cpu") -> Pipeline:
    """整体图：input → enc_stack(N 层) → output，权重**逐层照抄**顺序臂（R7-a 的对照）。"""
    from harness.registry import REGISTRY
    torch.manual_seed(1234)
    v = vocab_size()
    inp = REGISTRY["input"].build(d=seq.d, ff=FF, nhead=NHEAD, maxlen=seq.maxlen,
                                  dropout=DROPOUT, vocab_size=v, pad=pad_id())
    stk = REGISTRY["enc_stack"].build(d=seq.d, ff=FF, nhead=NHEAD, dropout=DROPOUT, n_layers=n)
    out = REGISTRY["output"].build(d=seq.d, maxlen=seq.maxlen, vocab_size=v)
    pipe = Pipeline([("input", inp), ("stack", stk), ("output", out)], d=seq.d,
                    maxlen=seq.maxlen, vocab=v, arch="fused", note="整体图对照（等价计算）")
    pipe.cards["input"].load_state_dict(seq.cards["input"].state_dict(), strict=True)
    pipe.cards["output"].load_state_dict(seq.cards["output"].state_dict(), strict=True)
    for i in range(n):
        pipe.cards["stack"].layers[i].load_state_dict(seq.cards[f"enc{i}"].enc.state_dict(),
                                                      strict=True)
    if dtype != torch.float32:
        pipe = pipe.to(dtype=dtype)
    pipe.eval()
    return pipe


# ---------------------------------------------------------------------------
# 单配置测量
# ---------------------------------------------------------------------------
def measure_one(n: int, d: int = D, dtype_name: str = "fp32", device: str = "cpu",
                weights: str = "trained", ckpt: str = CKPT_DEFAULT, data: str = DATA_TEST_K8,
                batch: int = 32, maxlen: int = MAXLEN, repeats: int = 5, do_measure: bool = True,
                ids=None, fp: str = "（复用主进程批）") -> dict:
    """一个配置的 Tier 1（确定性）+ Tier 2（墙钟，可关）+ 保真。"""
    dtype = DTYPES[dtype_name]
    if ids is None:
        ids, _s, fp, _ctx = probe_batch(data, batch, maxlen)
    ids = ids.to(device)
    pipe = build_seq(n, d=d, dtype=dtype, device=device, weights=weights, ckpt=ckpt)
    row = {"n": n, "d": d, "dtype": dtype_name, "device": device,
           "params_total": pipe.n_params(),
           "params_total_bytes": param_bytes(pipe),
           "layer_params": sum(sum(p.numel() for p in pipe.cards[f"enc{i}"].parameters())
                               for i in range(n)),
           "layer_params_bytes": sum(param_bytes(pipe.cards[f"enc{i}"]) for i in range(n)),
           "max_single_layer_bytes": max(param_bytes(pipe.cards[f"enc{i}"]) for i in range(n)),
           "iface_card_bytes": param_bytes(pipe.cards["input"]) + param_bytes(pipe.cards["output"]),
           "interface_bytes": int(ids.size(0) * ids.size(1) * d
                                 * torch.empty(0, dtype=dtype).element_size()),
           "batch": int(ids.size(0)), "seq": int(ids.size(1)), "batch_fp": fp}
    meter = Meter()
    t0 = time.perf_counter()
    pkt = pipe.run(ids, meter=meter)
    row["first_call_ms"] = (time.perf_counter() - t0) * 1000.0
    row["payload_peak_bytes"] = max(s["payload_bytes"] for s in meter.stages)
    row["live_peak_bytes"] = meter.peak_live
    row["stage_bytes"] = {s["stage"]: s["payload_bytes"] for s in meter.stages}
    row["resident_seq_bytes"] = (row["max_single_layer_bytes"] + row["iface_card_bytes"]
                                 + row["payload_peak_bytes"])
    row["resident_all_bytes"] = (row["layer_params_bytes"] + row["iface_card_bytes"]
                                 + row["payload_peak_bytes"])
    row["equiv_over_resident_seq"] = row["params_total_bytes"] / row["resident_seq_bytes"]
    row["equiv_over_resident_all"] = row["params_total_bytes"] / row["resident_all_bytes"]
    row["logits_digest"] = tensor_digest(pkt.logits)
    row["logits_shape"] = list(pkt.logits.shape)
    if do_measure:
        times = []
        with torch.no_grad():
            for _ in range(repeats):
                ta = time.perf_counter()
                pipe.run(ids)
                times.append(time.perf_counter() - ta)
        med = statistics.median(times)
        row["lat_per_batch_ms"] = med * 1000.0
        row["lat_per_token_us"] = med * 1e6 / (row["batch"] * row["seq"])
        row["lat_min_ms"] = min(times) * 1000.0
        row["lat_max_ms"] = max(times) * 1000.0
    row.update(fidelity(pipe, n, dtype=dtype, device=device, ids=ids))
    return row


def fidelity(seq: Pipeline, n: int, dtype=torch.float32, device: str = "cpu", ids=None) -> dict:
    """R7-a：顺序执行 vs 整体图 ⇒ max|Δ| / 逐位相同。

    对照在**同一个 dtype** 下做：fp32 档直接用 seq 的权重建整体图；fp16/bf16 档先建 fp32 孪生臂
    照抄权重、建整体图，再一起 cast 到目标 dtype ⇒ 两边权重逐位相同，差异只可能来自计算图。
    """
    if dtype == torch.float32:
        base = seq
    else:
        base = build_seq(n, d=seq.d, dtype=torch.float32, device="cpu", weights="fresh", ckpt="")
        base.cards["input"].load_state_dict(seq.cards["input"].state_dict(), strict=True)
        base.cards["output"].load_state_dict(seq.cards["output"].state_dict(), strict=True)
        for i in range(n):
            base.cards[f"enc{i}"].load_state_dict(seq.cards[f"enc{i}"].state_dict(), strict=True)
        base.eval()
    fused32 = build_fused(n, base, dtype=torch.float32, device="cpu")
    cmp_seq = seq if dtype != torch.float32 else base
    cmp_fused = fused32 if dtype == torch.float32 else fused32.to(dtype=dtype)
    cmp_fused.eval()
    with torch.no_grad():
        a = cmp_seq.run(ids).logits
        b = cmp_fused.run(ids).logits
    af, bf = a.float(), b.float()
    return {"fidelity_max_abs": float((af - bf).abs().max().item()),
            "fidelity_allclose": bool(torch.equal(a, b)),
            "fidelity_atol1e6": bool(torch.allclose(af, bf, atol=1e-6, rtol=0)),
            "fused_params": fused32.n_params()}


# ---------------------------------------------------------------------------
# 预留（CPU 档）：独立子进程的进程峰值 RSS
# ---------------------------------------------------------------------------
def rss_one(n: int, d: int, dtype_name: str, device: str, weights: str, ckpt: str,
            data: str, batch: int, maxlen: int) -> int:
    """在**独立子进程**里跑一次单配置前向，返回该进程的 VmHWM（KiB）。"""
    code = (
        "import os,sys,resource;"
        f"sys.path.insert(0,{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r});"
        "from harness.probe import _rss_child;"
        f"_rss_child({n!r},{d!r},{dtype_name!r},{device!r},{weights!r},{ckpt!r},{data!r},{batch!r},{maxlen!r})"
    )
    env = dict(os.environ)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HIP_VISIBLE_DEVICES"] = ""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    for ln in out.stdout.splitlines():
        if ln.startswith("[RSS]"):
            return int(ln.split()[1])
    return -1


def _rss_child(n, d, dtype_name, device, weights, ckpt, data, batch, maxlen):
    ids, _s, _fp, _c = probe_batch(data, batch, maxlen)
    pipe = build_seq(n, d=d, dtype=DTYPES[dtype_name], device=device, weights=weights, ckpt=ckpt)
    with torch.no_grad():
        pipe.run(ids.to(device))
    import resource
    print(f"[RSS] {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def human_bytes(x: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(x) < 1024 or unit == "GiB":
            return f"{x:.1f}{unit}" if unit != "B" else f"{int(x)}B"
        x /= 1024.0
    return f"{x:.1f}GiB"


def run(ns: list, ds: list, dtypes: list, device: str = "cpu", weights: str = "trained",
        ckpt: str = CKPT_DEFAULT, data: str = DATA_TEST_K8, batch: int = 32, maxlen: int = MAXLEN,
        repeats: int = 5, do_measure: bool = True, do_rss: bool = True, print_fn=print) -> dict:
    res = {"device": device, "weights": weights, "ckpt": ckpt, "data": data,
           "batch": batch, "maxlen": maxlen, "dtypes": list(dtypes), "n": list(ns), "w": list(ds),
           "torch": torch.__version__, "threads": torch.get_num_threads(),
           "rows": [], "rss": {}, "notes": [], "verdicts": {}}
    ids, _s, fp, ctx = probe_batch(data, batch, maxlen)
    print_fn("=" * 100)
    print_fn("harness probe · 推理探针（只前向；R7-b/c/d/e）")
    print_fn("=" * 100)
    print_fn(f"[T1-a] 探测批 = {data} 前 {len(ctx)} 条（ctx-only，右补 PAD 到 {maxlen}）"
             f" | batch={batch} maxlen={maxlen} | 批指纹 sha256[0:16]={fp}")
    print_fn(f"[T1-a] device={device} dtype 单列扫描={list(dtypes)} weights={weights} "
             f"torch={torch.__version__} threads={torch.get_num_threads()} "
             f"ckpt={os.path.basename(ckpt) if ckpt else '（无）'}")
    print_fn(f"[T1-a] 探针链 = input → N×enc_layer(同构层卡) → output；"
             f"N>2 时层卡权重按 [select.enc, realign.enc] 交替复制（我们只有这两种同尺寸层卡）")
    print_fn("[T1-a] 分配口径 = 逐卡边界遍历 Packet 里实际张量的字节和峰值（确定性）；"
             "预留口径 = 独立子进程 VmHWM（CPU）/ torch.cuda.max_memory_reserved（CUDA）")
    print_fn("")

    for dtype_name in dtypes:
        for d in ds:
            w_eff = weights if d == D else "fresh"
            if w_eff != weights:
                print_fn(f"[T1-a] ★接口宽 d={d} ≠ {D}：79 的 ckpt 形状不符 ⇒ 本档用 **fresh** 权重"
                         f"（只量形状/驻留/延迟，**不量质量**；这正是「接口宽 w 与质量」未测的诚实边界）")
            sub = []
            for n in ns:
                r = measure_one(n, d=d, dtype_name=dtype_name, device=device, weights=w_eff,
                                ckpt=ckpt, batch=batch, maxlen=maxlen, repeats=repeats,
                                do_measure=do_measure, ids=ids, fp=fp)
                sub.append(r)
                res["rows"].append(r)
            # 每 (dtype, d) 一张表
            print_fn(f"[T1-b] dtype={dtype_name} 接口宽 d={d} —— R7-b/R7-c/R7-e"
                     f"（N = 同构层卡数）")
            print_fn(f"  {'N':>2} {'总参数':>9} {'层参数':>9} {'单卡最大':>9} {'接口':>8} "
                     f"{'活载荷峰值':>10} {'驻留·顺序(推断)':>14} {'驻留·全常驻':>12} "
                     f"{'等效/驻留(顺序)':>15} {'等效/驻留(全常驻)':>16} {'保真max|Δ|':>10}")
            for r in sub:
                print_fn(f"  {r['n']:>2} {r['params_total']:>9} {r['layer_params']:>9} "
                         f"{human_bytes(r['max_single_layer_bytes']):>9} "
                         f"{human_bytes(r['interface_bytes']):>8} "
                         f"{human_bytes(r['payload_peak_bytes']):>10} "
                         f"{human_bytes(r['resident_seq_bytes']):>14} "
                         f"{human_bytes(r['resident_all_bytes']):>12} "
                         f"{r['equiv_over_resident_seq']:>15.3f} "
                         f"{r['equiv_over_resident_all']:>16.3f} "
                         f"{r['fidelity_max_abs']:>10.3e}")
            # R7-b 判定
            print_fn(f"[T1-c] R7-b 判定（dtype={dtype_name} d={d}）："
                     f"「驻留 ≈ max(单卡)+接口(+活载荷)」而非「N 卡之和」")
            for r in sub:
                pred_seq = r["max_single_layer_bytes"] + r["iface_card_bytes"]
                pred_all = r["layer_params_bytes"] + r["iface_card_bytes"]
                print_fn(f"        N={r['n']}：参数口径 顺序={human_bytes(pred_seq)} "
                         f"全常驻={human_bytes(pred_all)} 比值={pred_all / pred_seq:.3f}× "
                         f"| 实测（全常驻，含活载荷）={human_bytes(r['live_peak_bytes'])} "
                         f"= 驻留·全常驻={human_bytes(r['resident_all_bytes'])} "
                         f"({r['live_peak_bytes'] / r['resident_all_bytes']:.4f}×)")
            if do_measure:
                print_fn(f"[T2-a] 延迟（R7-d，dtype={dtype_name} d={d}；median/min/max，"
                         f"per-batch 与 per-token）")
                for r in sub:
                    print_fn(f"        N={r['n']}：per-batch={r['lat_per_batch_ms']:.1f}ms "
                             f"(min {r['lat_min_ms']:.1f} / max {r['lat_max_ms']:.1f}) "
                             f"per-token={r['lat_per_token_us']:.1f}us "
                             f"（= per-batch ÷ {r['batch']}×{r['seq']} token；"
                             f"首调 {r['first_call_ms']:.0f}ms 不计入）")
            if do_rss and device == "cpu":
                base = rss_one(0, d, dtype_name, device, "fresh", "", data, batch, maxlen)
                rss = {0: base}
                for n in ns:
                    rss[n] = rss_one(n, d, dtype_name, device, w_eff, ckpt, data, batch, maxlen)
                res["rss"][f"{dtype_name}_d{d}"] = rss
                print_fn(f"[T2-b] 预留（实测，独立子进程 VmHWM，dtype={dtype_name} d={d}）："
                         f"空载(N=0)={base / 1024:.1f}MiB"
                         + " ".join(f" N={n}:{rss[n] / 1024:.1f}MiB(Δ{(rss[n] - base) / 1024:+.1f})"
                                    for n in ns))
                diffs = [rss[n] - rss[m] for m, n in zip(ns[:-1], ns[1:])]
                if len(ns) < 2:
                    verdict = "单档（没有相邻档 ⇒ 逐卡增量无从比）"
                elif all(abs(x) > 2048 for x in diffs):
                    verdict = "可分辨（每档增量 > 2 MiB）"
                else:
                    verdict = "★不可分辨（RSS 分辨率 > 单卡参数差）"
                print_fn(f"[T2-b] 逐卡增量实测 = {[round(x / 1024, 2) for x in diffs]} MiB；"
                         f"一张层卡的**参数字节** = {human_bytes(sub[0]['max_single_layer_bytes'])}"
                         f" ⇒ {verdict}")
            print_fn("")

    # R7-c 曲线
    print_fn("[T1-d] R7-c 曲线：等效参数量 ÷ 驻留（越大 = 同样驻留里塞进的参数越多）")
    print_fn(f"  {'dtype':>5} {'d':>4} {'N':>2} {'等效参数':>10} {'驻留·顺序':>10} "
             f"{'驻留·全常驻':>12} {'等效/驻留顺序':>13} {'等效/驻留全常驻':>15} {'常驻/顺序':>9}")
    for r in res["rows"]:
        print_fn(f"  {r['dtype']:>5} {r['d']:>4} {r['n']:>2} {r['params_total']:>10} "
                 f"{human_bytes(r['resident_seq_bytes']):>10} {human_bytes(r['resident_all_bytes']):>12} "
                 f"{r['equiv_over_resident_seq']:>13.3f} {r['equiv_over_resident_all']:>15.3f} "
                 f"{r['resident_all_bytes'] / r['resident_seq_bytes']:>9.3f}")
    print_fn("")
    # R7-a 汇总
    worst = max(r["fidelity_max_abs"] for r in res["rows"])
    allok = all(r["fidelity_allclose"] for r in res["rows"])
    res["verdicts"]["R7-a"] = bool(allok)
    print_fn(f"[T1-e] R7-a 保真：顺序执行 vs 整体图 —— 全部配置 "
             f"{'逐位相同' if allok else '★不逐位相同'}，max|Δ| 全局最大={worst:.3e} "
             f"（fp32 档应当 0.0；fp16/bf16 档按浮点口径给 atol=1e-6 的 allclose）")
    if device == "cuda":
        res["verdicts"]["cuda_alloc"] = {k: v for k, v in _cuda_readings().items()}
        print_fn(f"[T1-f] CUDA 分配器双量（R7-e）：{_cuda_readings()}")
    if do_measure:
        print_fn("[DET] 读 R7-b 请读**参数口径**那两列：`活载荷峰值` 在本档由 logits"
                 f"（[batch, seq, {8192}]）与 causal mask 主导，不含卡参数的差别")
        print_fn("[DET] ★Tier 2 的墙钟与 RSS **逐次会变**（物理上不可重复）；"
                 "Tier 1（参数字节/接口/活载荷峰值/保真/指纹）与 `--no-measure` 的输出逐字节可复现。")
    else:
        print_fn("[DET] `--no-measure`：本输出不含墙钟与 RSS ⇒ 同命令两次逐字节相同。")
    return res


def _cuda_readings() -> dict:
    if not torch.cuda.is_initialized():
        return {"note": "CUDA 未初始化（本 run 没碰 GPU）"}
    return {"max_memory_allocated": torch.cuda.max_memory_allocated(),
            "max_memory_reserved": torch.cuda.max_memory_reserved()}


# ---------------------------------------------------------------------------
# 一卡定理式的快速自证：把示例卡插进 B2，输出应逐位不变
# ---------------------------------------------------------------------------
def example_card_fidelity(ckpt: str = CKPT_DEFAULT, data: str = DATA_TEST_K8, seed: int = 1234,
                          print_fn=print) -> dict:
    """H-d 的证据：新增一张卡（示例卡）只需接口 + 注册；未训练 ⇒ 输出逐位不变。"""
    from harness.pipeline import greedy_gen
    recs = load_jsonl(data)[:8]
    for r in recs:
        prep_record(r)
    base, _ = _load_b2(ckpt, seed, "B2")
    withex, _ = _load_b2(ckpt, seed, "B2+ex")
    t0, t1 = greedy_gen(base, recs), greedy_gen(withex, recs)
    ids, s, eidx, nc = pack(recs, device="cpu")
    d = (base.run(ids, cand_end=eidx, n_cands=nc, ctx_len=s).logits
         - withex.run(ids, cand_end=eidx, n_cands=nc, ctx_len=s).logits).abs().max().item()
    print_fn(f"[H-d] 示例卡 example_gain 插进 B2（未训练，g 零初始化）：")
    print_fn(f"      logits max|Δ| = {d:.3e} | 贪心文本逐条相同 = {t0 == t1} "
             f"| 参数 2700288 → {withex.n_params()}（+{withex.n_params() - base.n_params()}）"
             f" ⇒ 加卡本身零增量、纯延迟（§18.1）")
    return {"logits_max_abs": d, "greedy_identical": bool(t0 == t1),
            "params_base": base.n_params(), "params_with_example": withex.n_params()}


def _load_b2(ckpt: str, seed: int, arch: str):
    from harness.pipeline import build_pipeline
    pipe = build_pipeline(arch, seed=seed, device="cpu")
    rep = pipe.load_ckpt(ckpt)
    return pipe, rep
