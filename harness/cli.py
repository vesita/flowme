#!/usr/bin/env python3
"""harness.cli —— 子命令：chat / say / selfcheck / probe / list-cards。

    .venv/bin/python -m harness.cli list-cards
    .venv/bin/python -m harness.cli chat
    .venv/bin/python -m harness.cli say --turn "你好，我的售后工单号是AS975482。" \\
        --turn "我希望按返厂检测的方式处理。" --turn "帮我催一下吧。"
    .venv/bin/python -m harness.cli selfcheck --n 800 --json
    .venv/bin/python -m harness.cli probe --n 1,2,3,4 --interface 128

★ 默认 CPU：`--device cpu` 时本文件在 **import torch 之前**清空 `CUDA_VISIBLE_DEVICES` /
`HIP_VISIBLE_DEVICES`，所以连设备枚举都不发生（GPU 上有别的单元在跑）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# ★设备必须在 import torch 之前定下来（照 stages/83_card_demo.py 的机制）
# ---------------------------------------------------------------------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--device", default="cpu")
_pre_args, _ = _pre.parse_known_args()
if _pre_args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from harness import chat as hchat  # noqa: E402
from harness import pipeline as hp  # noqa: E402
from harness import probe as hprobe  # noqa: E402
from harness.registry import CARD_SETS, REGISTRY, card_set_rows, parse_cards  # noqa: E402

KFD_PROC = "/sys/class/kfd/kfd/proc"


def hr(ch="=", n=100):
    return ch * n


def kfd_pids():
    try:
        return set(os.listdir(KFD_PROC))
    except OSError:
        return None


def _ppid_map() -> dict:
    m = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/stat") as f:
                m[int(p)] = int(f.read().rsplit(")", 1)[1].split()[1])
        except Exception:
            pass
    return m


def attribute_pids(new_pids, me: int) -> tuple:
    """把新出现的 KFD PID 归因到**本进程树**：返回 (属于我的, 别人的)。

    别的单元（GPU 上的 84/85）随时可能起停进程 ⇒ 单看"KFD 集合变了"**不能**判定我用了 GPU。
    """
    m = _ppid_map()
    mine, other = [], []
    for p in new_pids:
        try:
            p = int(p)
        except ValueError:
            other.append(p)
            continue
        cur, guard = p, 0
        hit = False
        while cur and guard < 64:
            if cur == me:
                hit = True
                break
            cur, guard = m.get(cur, 0), guard + 1
        (mine if hit else other).append(p)
    return mine, other


def pid_label(p) -> str:
    try:
        with open(f"/proc/{p}/comm") as f:
            return f"{p}({f.read().strip()})"
    except Exception:
        return f"{p}(?)"


# ---------------------------------------------------------------------------
# 组装：--arch / --cards / --ckpt / --seed
# ---------------------------------------------------------------------------
def make_pipe(a, verbose=True):
    """按 (--arch | --cards) + --ckpt 造管线；返回 (pipe, ckpt_report, ckpt_path)。"""
    ckpt = a.ckpt
    if a.cards:
        layers = parse_cards(a.cards)
        pipe = hp.build_custom(layers, seed=a.seed, device=a.device, d=a.d)
        arch = "custom"
    else:
        arch = a.arch
        pipe = hp.build_pipeline(arch=arch, seed=a.seed, device=a.device, d=a.d, verbose=False)
    rep = None
    if ckpt:
        if not os.path.exists(ckpt):
            raise SystemExit(f"[harness] ★ckpt 不存在：{ckpt}")
        rep = pipe.load_ckpt(ckpt)
        if verbose:
            bad = rep["missing_any"] or rep["shape_mismatch_any"]
            print(f"[CKPT] {os.path.relpath(ckpt, ROOT)} sha256[0:16]={rep['sha256'][:16]} "
                  f"| 逐卡报告=" + " ".join(
                      f"{k}:{len(v['loaded'])}" for k, v in rep["per_card"].items())
                  + f" | 缺失={rep['missing_any'] or '无'} | 形状不符={rep['shape_mismatch_any'] or '无'}"
                  + (f" | ★未用的 key={rep['unused_keys']}" if rep["unused_keys"] else ""),
                  file=sys.stderr if verbose else sys.stdout)
            if bad:
                print(f"[CKPT] ★有卡没被完整灌上：{bad}（不许静默丢 key）", file=sys.stderr)
    return pipe, rep, ckpt


def add_model_args(ap):
    ap.add_argument("--arch", default="B2", choices=sorted(CARD_SETS), help="卡组（默认 B2）")
    ap.add_argument("--cards", default=None,
                    help="自定义卡组，如 input:input,select:select,realign:realign,output:output")
    ap.add_argument("--ckpt", default=hp.CKPT_B2_K8_S1234, help="ckpt（默认 B2 s1234）")
    ap.add_argument("--seed", type=int, default=1234, help="★构造用 seed（先播种再构造）")
    ap.add_argument("--d", type=int, default=hp.D, help="接口宽度 w（默认 128；改了要配 d 对应的 ckpt）")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="默认 cpu（别抢卡）")


# ---------------------------------------------------------------------------
# chat / say
# ---------------------------------------------------------------------------
def cmd_chat(a) -> int:
    pipe, rep, ckpt = make_pipe(a)
    sha = rep["sha256"] if rep else None
    if a.k != 8 or a.arch != "B2":
        print(f"[harness] 注意：本次 k={a.k} arch={a.arch}；记录值口径是 B2 + k=8（logs/76_*）")
    return hchat.repl(pipe, k=a.k, seed=a.seed, ckpt=os.path.relpath(ckpt, ROOT) if ckpt else None,
                      ckpt_sha=sha)


def cmd_say(a) -> int:
    pipe, rep, ckpt = make_pipe(a, verbose=False)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(hchat.banner(pipe, k=a.k, ckpt=os.path.relpath(ckpt, ROOT) if ckpt else None,
                       ckpt_sha=rep["sha256"] if rep else None))
    sess = hchat.Session(pipe, k=a.k, seed=a.seed)
    for text in a.turn:
        print(f"你> {text}")
        t = sess.ask(text)
        for o in t.ood:
            print(f"[OOD?] {o}")
        print(t.line())
        if t.source == "model":
            d = t.detail
            print(f"       [选择] 候选层 k={d['n_cands']}，ctx={d['ctx_len']} token，"
                  f"生成 {d['n_steps']} 步；族={d['family']} 实体={d['ent']} 条件={d['cond']}")
            same = d["ref"].strip() == t.reply.strip()
            print(f"       [参考] {d['ref']} ⇒ {'一致' if same else '★不一致'}")
        elif t.detail and t.detail.get("hint"):
            print(f"       [harness] {t.detail['hint']}")
    if a.json:
        print("[JSON] " + json.dumps(
            {"turns": [{"user": x[1], "reply": x[2], "source": x[0], "ood": x[3]}
                       for x in sess.history],
             "session": {"family": sess.family["key"] if sess.family else None,
                         "ent": sess.ent, "cond": sess.cond}},
            ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


# ---------------------------------------------------------------------------
# selfcheck（H-b / H-f）
# ---------------------------------------------------------------------------
def cmd_selfcheck(a) -> int:
    t_start = time.time()
    result = {"subcommand": "selfcheck", "device": a.device, "seed": a.seed, "n_eval": a.n,
              "records": hp.RECORDS, "verdicts": {}, "arms": {}}
    pids0 = kfd_pids()
    print(hr())
    print("flowme harness · selfcheck —— 复现 B2 s1234 记录值（CPU / 贪心 / 同一 ckpt + 同一 jsonl）")
    print(hr())
    print(f"[ENV] device={a.device} seed={a.seed} python={sys.version.split()[0]} "
          f"torch={torch.__version__} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<未设置>')!r} "
          f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES','<未设置>')!r} "
          f"torch.cuda.is_initialized()={torch.cuda.is_initialized()} threads={torch.get_num_threads()}")
    print(f"[ENV] GPU 进程快照：{'无法观测' if pids0 is None else f'{len(pids0)} 个（明细走 stderr）'}"
          f"；本进程 CUDA 未初始化 ⇒ 不占 GPU", file=sys.stderr, flush=True)
    print(f"[ENV] KFD 进程（selfcheck 前，明细，会逐次变 ⇒ 不进 stdout）="
          f"{sorted(pids0, key=int) if pids0 is not None else '无法观测'}", file=sys.stderr, flush=True)

    hp.load_tok()
    test = hp.load_jsonl(hp.DATA_TEST_K8)
    train = hp.load_jsonl(hp.DATA_TRAIN_K8)
    for r in test:
        hp.prep_record(r)
    for r in train:
        hp.prep_record(r)
    fp_test = hp.data_fp(test)
    fp_train = hp.data_fp(train)
    # ★与 stages/72 的定义对拍（同一份数据指纹、两个独立实现）
    d72 = hchat.d72()
    fp_test_ref = d72.data_fp(test)
    print(f"[VERIFY] data test = {os.path.relpath(hp.DATA_TEST_K8, ROOT)} n={len(test)} "
          f"sha256[0:8]={hp.file_sha256(hp.DATA_TEST_K8)[:8]} data_fp={fp_test} "
          f"| train n={len(train)} data_fp={fp_train}")
    print(f"[VERIFY] data_fp 对拍（harness vs stages/72 的独立实现）："
          f"test {'一致' if fp_test == fp_test_ref else '★不一致 ' + fp_test_ref} "
          f"| 引文口径（§11.6）test=1ae76682bd7cc997 train=efeb1ee872f4b6be ⇒ "
          f"{'一致' if fp_test == '1ae76682bd7cc997' else '★与引文不同（如实报）'}")
    result["data"] = {"test_sha256": hp.file_sha256(hp.DATA_TEST_K8), "test_n": len(test),
                      "test_fp": fp_test, "train_fp": fp_train, "fp_matches_d72": fp_test == fp_test_ref}

    # 构造顺序参考（[INIT-CHECK]）
    pipe0 = hp.build_pipeline("B2", seed=a.seed, device=a.device)
    d1 = hp.state_digest(pipe0)
    pipe0b = hp.build_pipeline("B2", seed=a.seed, device=a.device)
    d2 = hp.state_digest(pipe0b)
    del pipe0b
    print(f"[INIT-CHECK] 同 seed 两次构造 state_sha256[0:16]={d1} / {d2} ⇒ "
          f"{'PASS（构造确定，§9.7 的播种在构造之前）' if d1 == d2 else '★FAIL'}"
          f" | 参数={pipe0.n_params()}")
    result["init_check"] = {"digest1": d1, "digest2": d2, "ok": d1 == d2,
                            "params": pipe0.n_params()}
    del pipe0

    # cards/ 正规路径对拍（native3 == cards.CardPipeline(residual=False)）
    nat = hp.cross_check_cards_pipeline(device=a.device)
    print(f"[NATIVE] cards.CardPipeline(residual=False) 对拍：参数 {nat['pipeline_params']} vs "
          f"harness native3 {nat['native_params']} ⇒ {'一致' if nat['params_equal'] else '★不一致'}"
          f" | 同一权重下前向 max|Δ| = {nat['max_abs']:.3e} ⇒ "
          f"{'PASS（同一参数路径、同一数值路径）' if nat['ok'] else '★FAIL'}")
    result["native_check"] = nat

    strict_n = (a.n == 800)
    arms = {"B2": hp.CKPT_B2_K8_S1234, "A": os.path.join(hp.LOG_DIR, "79_ckpt_A_k8_s1234.pt")}
    if a.arm != "both":
        arms = {a.arm: arms[a.arm]}
    all_ok = True
    for arm, ckpt in arms.items():
        if not os.path.exists(ckpt):
            print(f"[ARM {arm}] ★ckpt 不存在：{ckpt} ⇒ 跳过")
            all_ok = False
            continue
        sha = hp.file_sha256(ckpt)
        print(hr("-"))
        print(f"[ARM {arm}] ckpt = {os.path.relpath(ckpt, ROOT)} bytes={os.path.getsize(ckpt)}")
        print(f"[ARM {arm}] sha256 = {sha}（前 16 位 {sha[:16]}）")
        pipe = hp.build_pipeline(arm, seed=a.seed, device=a.device)
        rep = pipe.load_ckpt(ckpt)
        print(f"[ARM {arm}] 载入逐卡 = " + " ".join(
            f"{k}:{len(v['loaded'])}" for k, v in rep["per_card"].items())
            + f" | 缺失={rep['missing_any'] or '无'} 形状不符={rep['shape_mismatch_any'] or '无'}"
            + f" 未用 key={rep['unused_keys'] or '无'}")
        ev = test[:a.n]
        t0 = time.time()
        m = hp.eval_records(pipe, ev)
        wall = time.time() - t0
        print(f"[ARM {arm}] [EM] n={m['n']} T2 完整回复 EM = {m['t2']:.2f}%（原始 {m['t2']:.4f}）"
              f" | T1 字母准确率 = {m['t1']:.2f}%（抽不出 {m['miss']}）"
              f" | 选择准确率 = " + ("不适用（无选择卡）" if m["sel"] is None else f"{m['sel']:.2f}%")
              + f" | 选择 argmax 字母 = " + ("不适用" if m["sel_letter"] is None else f"{m['sel_letter']:.2f}%"))
        print(f"[ARM {arm}] {'指标':<18}{'记录值':>10}{'实测(原始)':>16}{'实测(:.2f)':>14}"
              f"{'Δ(实测−记录)':>16}  判定")
        rows = hp.compare_records(arm, m)
        for r in rows:
            if r["record"] is None:
                got = "nan" if r["measured"] is None else f"{r['measured']:.4f}"
                print(f"[ARM {arm}] {r['label']:<18}{'—':>10}{got:>16}{'—':>14}{'—':>16}  "
                      f"{'PASS' if r['ok'] else '★FAIL'}（{r['note']}）")
            else:
                g = r["measured"]
                print(f"[ARM {arm}] {r['label']:<18}{r['record']:>10.2f}{g:>16.4f}{g:>14.2f}"
                      f"{r['delta']:>+16.4f}  {'PASS' if r['ok'] else '★FAIL'}"
                      + ("" if r["ok"] else f"  ⇒ {r['note']}"))
        ok = all(r["ok"] for r in rows)
        strict = (a.n == 800)
        if not strict:
            print(f"[ARM {arm}] ★--n={a.n} ≠ 800 ⇒ 与记录值的比对**仅供参考**，不计入判词"
                  f"（记录值是 800 条上的）")
        else:
            all_ok &= ok
        print(f"[ARM {arm}] 结论：{'一致 ⇒ 记录值可复现' if ok else '★不一致 ⇒ 见上表 Δ，必须查因（不许改记录值）'}"
              f" | 评测墙钟={wall:.1f}s（走 stderr，不进 stdout）", file=sys.stderr, flush=True)
        result["arms"][arm] = {"ckpt_sha256": sha, "metrics": {
            k: m[k] for k in ("n", "t2", "t1", "sel", "sel_letter", "miss")},
            "record_check": rows, "ok": ok,
            "load_report_missing": rep["missing_any"]}

    # 确定性自查（H-f）
    pids1 = kfd_pids()
    new_pids = None if (pids0 is None or pids1 is None) else sorted(pids1 - pids0, key=int)
    mine, other = (None, None) if new_pids is None else attribute_pids(new_pids, os.getpid())
    print(hr("-"))
    print("[H-f] 确定性：贪心 + 固定 seed + 无随机 + stdout 不含耗时。自查："
          "同命令跑两次后 `sha256sum r1 r2` 两行必须相同")
    print("[H-f] ckpt sha256 与 data_fp 已打印（上面 [ARM]/[VERIFY] 行）；`--json` 供机读")
    hg_ok = (a.device == "cpu") and (not torch.cuda.is_initialized()) and (mine == [] if mine is not None
                                                                           else False)
    print(f"[H-g] 不占 GPU：device={a.device} torch.cuda.is_initialized()={torch.cuda.is_initialized()} "
          f"| 本进程树内的 KFD 进程={len(mine) if mine is not None else '无法观测'} 个"
          f"（别人的进程数会逐次变 ⇒ 只进 stderr）"
          f" ⇒ {'PASS（本进程树内没有 GPU 进程，且 CUDA 从未初始化）' if hg_ok else '★不可判/FAIL（如实报）'}")
    print(f"[H-g] KFD 明细（会逐次变 ⇒ 不进 stdout）：前="
          f"{sorted(pids0, key=int) if pids0 is not None else '无法观测'} 后="
          f"{sorted(pids1, key=int) if pids1 is not None else '无法观测'} 新="
          f"{new_pids if new_pids is not None else '无法观测'} | 本进程树内="
          f"{[pid_label(x) for x in (mine or [])] or '无'}；别人的="
          f"{[pid_label(x) for x in (other or [])] or '无'}", file=sys.stderr, flush=True)
    # JSON 只放**稳定**布尔量（PID 明细走 stderr）⇒ `--json` 的输出也逐字节可复现
    result["kfd"] = {"device": a.device,
                     "torch_cuda_initialized": bool(torch.cuda.is_initialized()),
                     "new_pids_in_my_tree": None if mine is None else len(mine),
                     "ok": bool(hg_ok)}
    result["verdicts"]["H-b"] = bool(all_ok) if strict_n else None
    result["verdicts"]["H-g"] = bool(hg_ok)
    result["ok"] = bool(all_ok and result["init_check"]["ok"] and nat["ok"] and hg_ok)
    print(hr())
    print(f"[总判] H-b {'（--n≠800，未判）' if not strict_n else ('PASS' if all_ok else '★FAIL')} | "
          f"构造确定 {'PASS' if result['init_check']['ok'] else '★FAIL'} | "
          f"cards/ 对拍 {'PASS' if nat['ok'] else '★FAIL'} ⇒ exit={'0' if result['ok'] else '1'}")
    print(hr())
    if a.json:
        print("[JSON] " + json.dumps(result, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")))
    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, sort_keys=True, indent=2)
        print(f"[JSON] 已写入 {a.json_out}", file=sys.stderr, flush=True)
    print(f"[总判] 总墙钟={time.time() - t_start:.0f}s（走 stderr）", file=sys.stderr, flush=True)
    return 0 if result["ok"] else 1


# ---------------------------------------------------------------------------
# probe（H-c）
# ---------------------------------------------------------------------------
def cmd_probe(a) -> int:
    ns = [int(x) for x in a.n.split(",") if x.strip()]
    ds = [int(x) for x in a.interface.split(",") if x.strip()]
    dtypes = [x.strip() for x in a.dtype.split(",") if x.strip()]
    for t in dtypes:
        if t not in hprobe.DTYPES:
            raise SystemExit(f"[harness] 未知 dtype {t!r}（可选 {sorted(hprobe.DTYPES)}）")
    res = hprobe.run(ns=ns, ds=ds, dtypes=dtypes, device=a.device, weights=a.weights,
                     ckpt=a.ckpt, data=a.data, batch=a.batch, maxlen=a.maxlen,
                     repeats=a.repeats, do_measure=not a.no_measure,
                     do_rss=not a.no_measure and not a.no_rss)
    if a.hd:
        res["example_card_fidelity"] = hprobe.example_card_fidelity(
            ckpt=a.ckpt, data=a.data, seed=a.seed)
    if a.json:
        print("[JSON] " + json.dumps(res, ensure_ascii=False, sort_keys=True, default=str,
                                    separators=(",", ":")))
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, sort_keys=True, indent=2, default=str)
        print(f"[JSON] 已写入 {a.json_out}", file=sys.stderr, flush=True)
    return 0


# ---------------------------------------------------------------------------
# list-cards（H-d/H-e）
# ---------------------------------------------------------------------------
def cmd_list_cards(a) -> int:
    print(hr())
    print("harness 注册表：卡片名字 → 构造器 / 接口声明 / 可判目标 / 依赖张量 / ckpt 前缀")
    print(hr())
    print(REGISTRY.table())
    print(hr("-"))
    print("卡组（arch）：")
    for s in card_set_rows():
        print(f"  {s['arch']:<9} " + " → ".join(f"{x['instance']}:{x['spec']}" for x in s["layers"]))
    print(hr("-"))
    print("每张卡的完整声明（接口 / 依赖 / 可判目标 / ckpt）：")
    for r in REGISTRY.rows():
        j = r["judged"]
        print(f"  · {r['name']}（{r['kind']}；构造器来自 {r['source']}）")
        print(f"      接口：d_in={r['interface']['d_in']} d_out={r['interface']['d_out']} "
              f"注入={r['interface']['injection']} 含位置={r['interface']['carries_position']} "
              f"emit={r['interface']['emits']}")
        print(f"      依赖张量：{r['deps']}（供冻结/微调与显存计算）")
        print(f"      ckpt：前缀={r['ckpt_prefix']} 根 key={r['ckpt_root_keys']}")
        print(f"      可判中间目标：{'—（无 ⇒ 按 §18.1 加它只是加延迟）' if j is None else j['metric']}")
        if j is not None:
            print(f"                   目标={j['name']}")
            print(f"                   需要字段={j['needs']} 机会水平={j['chance'] or '—'}")
        print(f"      说明：{r['doc']}")
    if a.json:
        print("[JSON] " + json.dumps({"cards": REGISTRY.rows(), "card_sets": card_set_rows()},
                                    ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="harness", description="flowme harness（推理入口）",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chat", help="交互式 REPL（多轮历史累积）")
    add_model_args(c)
    c.add_argument("--k", type=int, default=8, help="候选层大小（记录值口径 = 8）")
    c.set_defaults(fn=cmd_chat)

    s = sub.add_parser("say", help="一次性：把 --turn 依次喂进同一个会话（非交互）")
    add_model_args(s)
    s.add_argument("--k", type=int, default=8)
    s.add_argument("--turn", action="append", required=True, help="一轮用户输入（可重复）")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_say)

    sc = sub.add_parser("selfcheck", help="在测试集上复现记录值（H-b）+ 构造/正规路径对拍")
    sc.add_argument("--arm", default="both", choices=("B2", "A", "both"))
    sc.add_argument("--n", type=int, default=800, help="评测条数（默认全量 800）")
    sc.add_argument("--seed", type=int, default=1234)
    sc.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    sc.add_argument("--json", action="store_true")
    sc.add_argument("--json-out", default=None)
    sc.set_defaults(fn=cmd_selfcheck)

    p = sub.add_parser("probe", help="推理探针：卡数 / 接口宽度 / dtype → 驻留·延迟·保真（H-c）")
    p.add_argument("--n", default="1,2,3,4", help="同构层卡数（逗号分隔）")
    p.add_argument("--interface", default="128", help="接口宽度 w（逗号分隔；≠128 时无 ckpt ⇒ 只用 fresh）")
    p.add_argument("--dtype", default="fp32", help="fp32 / fp16 / bf16（逗号分隔 = 多档并列，不混算）")
    p.add_argument("--weights", default="trained", choices=("trained", "fresh"))
    p.add_argument("--ckpt", default=hprobe.CKPT_DEFAULT)
    p.add_argument("--data", default=hp.DATA_TEST_K8)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--maxlen", type=int, default=hp.MAXLEN)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--no-measure", action="store_true", help="跳过 Tier 2（墙钟/RSS）⇒ 输出逐字节可复现")
    p.add_argument("--no-rss", action="store_true", help="只跳过 RSS（保留墙钟）")
    p.add_argument("--hd", action="store_true", help="附带跑 H-d 的示例卡保真（插卡后输出逐位不变）")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--json", action="store_true")
    p.add_argument("--json-out", default=None)
    p.set_defaults(fn=cmd_probe)

    lc = sub.add_parser("list-cards", help="打印注册表与卡组（H-d/H-e）")
    lc.add_argument("--json", action="store_true")
    lc.set_defaults(fn=cmd_list_cards)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
