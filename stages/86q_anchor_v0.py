#!/usr/bin/env python3
"""S86q · 锚定 v0「模因版本」的可执行工件（`MEME_FRAMEWORK.md §4.1` 三段式第①段）。

**锚定的准确含义 = 版本化快照 + 冻结开关**（只读快照 + sha256 + 契约版本 + 按卡冻结/解冻），
**不是"永久冻结"** —— 后面仍可解冻 IO 联合微调（§4.1 第③段）。

    --anchor   --ckpts p1,p2,... --out DIR [--note TEXT] [--tokenizer P] [--pairs P] [--cards {input,all}]
        对每个 ckpt 取接口侧（MV 侧）张量，用 `cards.base.state_sha256` 算权重 sha，
        落 `DIR/manifest.json`（只读）+ `DIR/<ckpt名>.anchor_v0.pt`（只读快照）：
          · `--cards input`（**默认**）：只取输入卡 `zh_in.*` + `en_in.*`（行为与字段与旧版逐位一致）；
          · `--cards all`：再取输出卡 `zh_out.*` + `en_out.*`（MV 的接口侧两半都进
            `weights_sha256` ⇒ 都进 `mv.interface_sha256`），逐卡记 `zh_out_sha256` / `en_out_sha256`；
            思维卡 `bridge.*`（**不属于 MV**）单独记 `thinking_card`（sha256 + n_params + 说明），
            **不进 mv**；ckpt 无 bridge ⇒ `thinking_card=null`（输出行显示「无」）。
            manifest 额外落 `cards_scope: "all"`，`--verify` / `--load-check` 据此判定口径。
    --verify   MANIFEST
        重算 契约（分词器/pairs sha + 契约常量）· 每张卡权重 sha · mv · 快照 sha 并逐位比对
        （`cards_scope=all` 时连输出卡两半与 `thinking_card` 一并重算比对）；
        任何不一致 / 字段缺 / 文件缺 ⇒ **非零退出并指出是哪一项**。
    --load-check  MANIFEST --seed S [--steps N] [--device cpu] [--freeze]
        构造 `ZHENModel`，把该 seed 的接口卡权重加载进去，然后
        ① 断言各卡 `state_sha256` 与 manifest 逐位相同（不一致 ⇒ AssertionError）；
          `cards_scope=all` ⇒ 输入卡与输出卡**两半**都加载、都断言（思维卡按 ckpt 张量断言）；
        ② `--freeze` ⇒ 接口卡全部参数 `requires_grad is False` 且再算 sha 仍相同；
        ③ 打印锚定摘要（mv 短哈希 + 卡数 + 契约版本）。

硬纪律（违反即 fail-closed，非零退出）：
  · 权重 sha **只用** `cards.base.state_sha256`（一个事实只有一个实现，本文件不另写哈希）；
  · manifest 无时间戳、无绝对路径 ⇒ **同一输入两次运行逐位相同**（快照 sha 也相同）；
  · 快照与 manifest 落盘后 chmod 0444；缺 ckpt / 缺分词器 / sha 不匹配 / manifest 字段缺 ⇒ 非零退出。

★ 本锚定对应的是 `experiments/meme_anchor` 的 zh-en 模型（`ZHENModel`），
  契约常量取自 `experiments/meme_anchor/model.py:28` 与 `data.py`（家）；
  `cards/base.py` 的 `Contract` 是 `cards/` 那套模型的，此处**只** import
  `state_sha256` 与 `CONTRACT_VERSION`（如实写在 manifest 的 `contract.provenance`）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import NoReturn

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "meme_anchor"))

from cards.base import CONTRACT_VERSION, state_sha256           # noqa: E402  复用，不另写哈希
from data import BOS, EOS, PAD, UNK, VOCAB                      # noqa: E402  家：experiments/meme_anchor/data.py:42-43
from model import D, FF, MAXLEN, NHEAD, SLOTS                   # noqa: E402  家：experiments/meme_anchor/model.py:28
import model_zh_en as MZ                                        # noqa: E402

FORMAT = "flowme-anchor-v0"
SNAPSHOT_FORMAT = "flowme-anchor-v0-snapshot"
W_VERSION_NOTE = "占位，对齐矩阵未实现"      # §4.1：对齐矩阵 W 本轮未实现，位子留在 MV 里
SPLIT_SEED = 20261011                        # 家：stages/86g_zh_en.py:36
DEFAULT_TOKENIZER = ROOT / "logs" / "86_tokenizer" / "char_tokenizer.json"
DEFAULT_PAIRS = ROOT / "logs" / "86_cache_zhen" / "pairs.jsonl"
INPUT_PREFIXES = ("zh_in.", "en_in.")
OUTPUT_PREFIXES = ("zh_out.", "en_out.")          # 输出卡：MV 的接口侧（另一半）
THINKING_PREFIXES = ("bridge.", "bridge_zh.", "bridge_en.")   # 思维卡：**不属于 MV**
INTERFACE_PREFIXES = INPUT_PREFIXES + OUTPUT_PREFIXES         # ★ 接口侧组合顺序**写死在这里**（不依赖 dict 顺序）
SCOPE_INPUT, SCOPE_ALL = "input", "all"
SCOPE_FIELD = "cards_scope"                       # 只在 --cards all 时落盘（input 模式字段集不变）
THINKING_NOTE = "思维卡（bridge.*）单独记录：不属于 MV，不进 mv.interface_sha256，权重留在 ckpt"
SCOPE_NOTE = ("本锚定对应 experiments/meme_anchor 的 zh-en 模型（ZHENModel）；"
              "契约字段取自 experiments/meme_anchor/model.py:28 与 data.py；"
              "cards/base.py 的 Contract 属 cards/ 那套模型，此处只复用 state_sha256 与 CONTRACT_VERSION")

CONTRACT_FIELDS = ("BOS", "D", "EOS", "FF", "MAXLEN", "NHEAD", "PAD", "SLOTS", "UNK", "VOCAB",
                   "pairs", "pairs_count", "pairs_sha256", "provenance", "split_seed",
                   "tokenizer", "tokenizer_sha256")
REQUIRED_TOP = ("format", "contract", "cards", "mv", "recipe", "notes")
REQUIRED_MV = ("interface_sha256", "contract_version", "w_version", "sha256")
REQUIRED_CARD = ("ckpt", "seed", "steps", "maps", "loss_kind", "weights_sha256",
                 "interface_sha256", "zh_in_sha256", "en_in_sha256", "snapshot", "snapshot_sha256")
# --cards all 额外必须有的字段（缺 ⇒ fail-closed）
REQUIRED_CARD_ALL = ("zh_out_sha256", "en_out_sha256", "thinking_card")
RECOMPUTE_CARD_FIELDS = ("seed", "steps", "maps", "loss_kind", "shared_out", "weights_sha256",
                         "interface_sha256", "zh_in_sha256", "en_in_sha256")
# --cards all：**先比逐卡 sha**（哪一半被改就指名哪一半），再比组合/派生 sha
RECOMPUTE_CARD_FIELDS_ALL = ("seed", "steps", "maps", "loss_kind", "shared_out",
                             "zh_in_sha256", "en_in_sha256", "zh_out_sha256", "en_out_sha256",
                             "thinking_card", "weights_sha256", "interface_sha256")
PATH_FLAGS = ("--ckpts", "--tokenizer", "--pairs", "--verify", "--load-check")


# ------------------------------------------------------------------ 基础件
def fail(msg: str, code: int = 1) -> NoReturn:
    """fail-closed：一句话说清是哪一项，然后非零退出。"""
    print(f"[FAIL] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def canonical_sha(obj) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def norm_path(path) -> str:
    """manifest 里绝不落绝对路径：仓库内 → 仓库相对；仓库外 → cwd 相对；已是相对 → 原样。"""
    s = str(path)
    if not os.path.isabs(s):
        return s
    ap = Path(s).resolve()
    cwd = Path.cwd().resolve()
    for base in (ROOT, cwd):
        try:
            return str(ap.relative_to(base))
        except ValueError:
            continue
    return os.path.relpath(str(ap), str(cwd))


def resolve_path(stored: str, bases: tuple[Path, ...] = ()) -> Path | None:
    """按 附加基目录 → cwd → 仓库根 的顺序找文件；找不到返回 None（调用方 fail-closed）。"""
    p = Path(stored)
    cand: list[Path] = []
    if p.is_absolute():
        cand.append(p)
    else:
        cand += [b / p for b in bases]
        cand.append(Path.cwd() / p)
        cand.append(ROOT / p)
    for c in cand:
        if c.is_file():
            return c
    return None


def dump_manifest(manifest: dict) -> bytes:
    return (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_readonly_bytes(path: Path, blob: bytes) -> None:
    if path.exists():
        path.chmod(0o644)                      # 幂等覆盖旧产物
    path.write_bytes(blob)
    path.chmod(0o444)                          # ★ 只读


def write_snapshot(path: Path, payload: dict) -> str:
    if path.exists():
        path.chmod(0o644)
    torch.save(payload, path)                  # 同 payload ⇒ 同字节（无时间戳/路径）
    path.chmod(0o444)
    return file_sha256(path)


# ------------------------------------------------------------------ 契约 / sha
def build_contract(tok_path: Path, pairs_path: Path) -> dict:
    """契约 = 全部契约常量（家：meme_anchor model.py/data.py）+ 分词器与数据版的 sha。"""
    if not tok_path.is_file():
        fail(f"分词器文件缺失：{norm_path(tok_path)}（fail-closed）")
    if not pairs_path.is_file():
        fail(f"pairs 数据缺失：{norm_path(pairs_path)}（fail-closed）")
    return {
        "BOS": BOS, "D": D, "EOS": EOS, "FF": FF, "MAXLEN": MAXLEN, "NHEAD": NHEAD,
        "PAD": PAD, "SLOTS": SLOTS, "UNK": UNK, "VOCAB": VOCAB,
        "pairs": norm_path(pairs_path),
        "pairs_count": count_lines(pairs_path),
        "pairs_sha256": file_sha256(pairs_path),
        "provenance": SCOPE_NOTE,
        "split_seed": SPLIT_SEED,
        "tokenizer": norm_path(tok_path),
        "tokenizer_sha256": file_sha256(tok_path),
    }


def card_interface_sha(*, contract_sha: str, weights_sha: str) -> str:
    h = hashlib.sha256()
    h.update(b"flowme-anchor-v0-card-interface|")
    h.update(contract_sha.encode("utf-8"))
    h.update(b"|")
    h.update(weights_sha.encode("utf-8"))
    return h.hexdigest()


def mv_of(*, contract_sha: str, entries: list[dict]) -> dict:
    """mv = (接口快照 sha256, 契约版本, 对齐矩阵 W 版本占位)（§4.1 只读快照 + sha256 + 契约版本）。

    interface_sha256 = 契约 sha + **各卡权重 sha** 的组合哈希（卡片顺序 = manifest 顺序）。
    `weights_sha256` 按 `--cards` 取的前缀集合算（input = `zh_in.`+`en_in.`；
    all = `INTERFACE_PREFIXES` 四组，**两半都进**）⇒ 输出卡也经此进入本哈希；
    前缀组合顺序写死在 `INTERFACE_PREFIXES`，不依赖 dict 顺序。
    """
    h = hashlib.sha256()
    h.update(b"flowme-anchor-v0-interface|")
    h.update(contract_sha.encode("utf-8"))
    for e in entries:
        h.update(b"|")
        h.update(e["weights_sha256"].encode("utf-8"))
    mv = {
        "contract_version": CONTRACT_VERSION,
        "interface_sha256": h.hexdigest(),
        "w_version": W_VERSION_NOTE,
    }
    mv["sha256"] = canonical_sha(mv)           # hash 三字段本身（"sha256" 不进自己的输入）
    return mv


def card_state(state, *, scope: str, label: str = "--anchor") -> dict:
    """取接口侧（MV 侧）张量：按 `--cards` 的前缀元组逐组取，**顺序写死**，缺任一前缀 ⇒ fail-closed。"""
    prefixes = INTERFACE_PREFIXES if scope == SCOPE_ALL else INPUT_PREFIXES
    if not isinstance(state, dict):
        fail(f"{label}: ckpt 的 state 不是 dict：{type(state).__name__}")
    sub: dict = {}
    for pre in prefixes:                       # ★ 组合顺序 = 前缀元组顺序，不依赖 ckpt/dict 顺序
        for k, v in state.items():
            if isinstance(k, str) and k.startswith(pre):
                sub[k] = v
        if not any(k.startswith(pre) for k in sub):
            fail(f"{label}: ckpt 缺 {pre}* 张量（--cards {scope} 要求的前缀），不是可锚定的 86g ckpt")
    return sub


def prefixed_state(state, prefixes: tuple[str, ...]) -> dict:
    """按给定前缀元组取子 state（**去掉前缀的键名**，供逐卡 load_state_dict / 算逐卡 sha）。"""
    out: dict = {}
    for pre in prefixes:
        for k, v in state.items():
            if isinstance(k, str) and k.startswith(pre):
                out[k[len(pre):]] = v
    return out


def thinking_entry(state) -> dict | None:
    """思维卡（`bridge.*`）单独记录：**不属于 MV**，不进 mv；ckpt 无 bridge ⇒ None（显示「无」）。"""
    if not isinstance(state, dict):
        fail("--anchor: ckpt 的 state 不是 dict（思维卡无法判定）")
    sub: dict = {}
    for pre in THINKING_PREFIXES:
        for k, v in state.items():
            if isinstance(k, str) and k.startswith(pre):
                sub[k] = v
    has_shared = any(k.startswith("bridge.") for k in sub)
    has_dual = any(k.startswith(("bridge_zh.", "bridge_en.")) for k in sub)
    if has_shared and has_dual:
        fail("ckpt 同时含 bridge.* 与 bridge_zh./bridge_en.*（思维卡结构不合法）")
    if has_dual and not (any(k.startswith("bridge_zh.") for k in sub)
                         and any(k.startswith("bridge_en.") for k in sub)):
        fail("ckpt 的 dual 思维卡不完整（bridge_zh.* 与 bridge_en.* 必须同时在）")
    if not sub:
        return None                            # 「没有」：该 ckpt 就是无思维卡的 v0 锚定
    return {
        "kind": "parser" if has_shared else "dual",
        "n_params": int(sum(v.numel() for v in sub.values() if hasattr(v, "numel"))),
        "note": THINKING_NOTE,
        "sha256": state_sha256(sub),
    }


def load_ckpt(path: Path, *, label: str) -> dict:
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:                   # noqa: BLE001 —— fail-closed，原样报错
        fail(f"{label}: ckpt 读取失败 {path}：{exc}")
    if not isinstance(blob, dict) or "state" not in blob:
        fail(f"{label}: {path} 不是 dict(state=...) 格式的 86g ckpt")
    return blob


def card_entry(*, ckpt_norm: str, contract_sha: str, blob: dict, label: str,
               scope: str = SCOPE_INPUT) -> dict:
    for k in ("seed", "steps", "maps", "loss_kind"):
        if k not in blob:
            fail(f"{label}: ckpt 缺字段 {k!r}（86g 格式必须有 seed/steps/maps/loss_kind）")
    sub = card_state(blob["state"], scope=scope, label=label)
    zh = prefixed_state(sub, ("zh_in.",))
    en = prefixed_state(sub, ("en_in.",))
    w_sha = state_sha256(sub)                  # ★ cards.base.state_sha256，全仓唯一实现
    entry = {
        "ckpt": ckpt_norm,
        "en_in_sha256": state_sha256(en),
        "interface_sha256": card_interface_sha(contract_sha=contract_sha, weights_sha=w_sha),
        "loss_kind": str(blob["loss_kind"]),
        "maps": list(blob["maps"]),
        "seed": int(blob["seed"]),
        "shared_out": blob.get("shared_out"),
        "steps": int(blob["steps"]),
        "weights_sha256": w_sha,
        "zh_in_sha256": state_sha256(zh),
    }
    if scope == SCOPE_ALL:
        # 输出卡（MV 接口侧另一半）逐卡记 sha；思维卡单独记，**不进 mv**
        entry["en_out_sha256"] = state_sha256(prefixed_state(sub, ("en_out.",)))
        entry["thinking_card"] = thinking_entry(blob["state"])
        entry["zh_out_sha256"] = state_sha256(prefixed_state(sub, ("zh_out.",)))
    return entry


# ------------------------------------------------------------------ recipe
def hardware() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "cpu_count": os.cpu_count(),
        "cuda_available": bool(cuda),
        "gpu": torch.cuda.get_device_name(0) if cuda else "CUDA 不可见（本进程 CPU-only）",
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }


def normalized_cli(argv: list[str] | None = None) -> str:
    """argv 逐字记录，仅做可复现的归一：脚本路径相对化、路径参数相对化、`--out` 值 → `<out>`。"""
    raw = list(sys.argv if argv is None else argv)
    head = Path(raw[0])
    if head.is_absolute():
        try:
            head_s = str(head.resolve().relative_to(ROOT))
        except ValueError:
            head_s = head.name
    else:
        head_s = str(head)
    toks = [head_s] + list(raw[1:])
    i = 1
    while i < len(toks):
        t = toks[i]
        if t == "--out" and i + 1 < len(toks):
            toks[i + 1] = "<out>"
            i += 2
            continue
        if t.startswith("--out="):
            toks[i] = "--out=<out>"
            i += 1
            continue
        if t in PATH_FLAGS and i + 1 < len(toks):
            if t == "--ckpts":
                toks[i + 1] = ",".join(norm_path(x) for x in toks[i + 1].split(",") if x.strip())
            else:
                toks[i + 1] = norm_path(toks[i + 1])
            i += 2
            continue
        i += 1
    return " ".join(t if (not t or not any(c.isspace() for c in t)) else _shlex_quote(t) for t in toks)


def _shlex_quote(s: str) -> str:
    import shlex
    return shlex.quote(s)


def build_recipe(*, contract: dict) -> dict:
    return {
        "cli": normalized_cli(),
        "cli_note": ("argv 逐字记录，仅归一两处：脚本路径改仓库相对、--out 的值替换为 <out>"
                     "（out 目录不入 manifest，否则两次不同输出目录的运行不可能逐位相同）；"
                     "其余路径一律相对化 ⇒ manifest 里没有绝对路径、没有时间戳"),
        "data": (f"数据版：{contract['pairs']}（{contract['pairs_count']} 对；train 40000 / held-out 2245；"
                 f"SPLIT_SEED={contract['split_seed']}）· 分词器 {contract['tokenizer']}"),
        "hardware": hardware(),
    }


# ------------------------------------------------------------------ manifest 读入（字段缺 ⇒ 非零退出）
def load_manifest(path_str: str) -> tuple[dict, Path]:
    p = resolve_path(path_str)
    if p is None:
        fail(f"manifest 文件不存在：{path_str}（fail-closed）")
    try:
        man = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:                   # noqa: BLE001
        fail(f"manifest 解析失败：{p}：{exc}")
    if not isinstance(man, dict):
        fail(f"manifest 不是 JSON object：{p}")
    for k in REQUIRED_TOP:
        if k not in man:
            fail(f"manifest 字段缺：{k}")
    if man["format"] != FORMAT:
        fail(f"format：期望 {FORMAT!r}，实得 {man.get('format')!r}")
    contract = man["contract"]
    if not isinstance(contract, dict):
        fail("manifest 字段缺/坏：contract 不是 object")
    for k in CONTRACT_FIELDS:
        if k not in contract:
            fail(f"manifest 字段缺：contract.{k}")
    cards = man["cards"]
    if not isinstance(cards, list) or not cards:
        fail("manifest 字段缺/坏：cards 必须是非空数组")
    scope = man.get(SCOPE_FIELD, SCOPE_INPUT)   # 旧 manifest（input）没有该字段 ⇒ input
    if scope not in (SCOPE_INPUT, SCOPE_ALL):
        fail(f"{SCOPE_FIELD}：期望 {SCOPE_INPUT!r}/{SCOPE_ALL!r}，实得 {scope!r}")
    need = REQUIRED_CARD + (REQUIRED_CARD_ALL if scope == SCOPE_ALL else ())
    for i, e in enumerate(cards):
        if not isinstance(e, dict):
            fail(f"cards[{i}] 不是 object")
        for k in need:
            if k not in e:
                fail(f"manifest 字段缺：cards[{i}].{k}（{SCOPE_FIELD}={scope}）")
        if scope == SCOPE_INPUT:               # 口径必须自洽：input 模式不许带输出卡/思维卡字段
            for k in REQUIRED_CARD_ALL:
                if k in e:
                    fail(f"cards[{i}].{k}: {SCOPE_FIELD}=input 却带输出卡字段 —— 口径不符（fail-closed）")
    if not isinstance(man["mv"], dict):
        fail("manifest 字段缺/坏：mv 不是 object")
    for k in REQUIRED_MV:
        if k not in man["mv"]:
            fail(f"manifest 字段缺：mv.{k}")
    if not isinstance(man["recipe"], dict):
        fail("manifest 字段缺/坏：recipe 不是 object")
    return man, p


# ------------------------------------------------------------------ --anchor
def thinking_text(tc: dict | None) -> str:
    """输出行里的思维卡摘要：无 bridge ⇒ 「无」（如实显示，不编造 sha）。"""
    if tc is None:
        return "无（ckpt 无 bridge.* 思维卡；思维卡不属于 MV）"
    return (f"sha256={tc['sha256'][:16]} kind={tc['kind']} n_params={tc['n_params']}"
            f"（思维卡，不属于 MV，不进 mv.interface_sha256）")


def cmd_anchor(args: argparse.Namespace) -> None:
    scope = args.cards                      # 默认值在 main 里归一（参数面只有一个家）
    raw = [s.strip() for s in args.ckpts.split(",") if s.strip()]
    if not raw:
        fail("--anchor 需要 --ckpts p1,p2,...（至少一个）", code=2)
    if not args.out:
        fail("--anchor 需要 --out <目录>", code=2)
    ck_paths: list[Path] = []
    for s in raw:
        p = resolve_path(s)
        if p is None:
            fail(f"ckpt 缺失：{s}（fail-closed）")
        ck_paths.append(p)
    if args.tokenizer:
        tok = resolve_path(args.tokenizer)
        if tok is None:
            fail(f"分词器文件缺失：{args.tokenizer}（fail-closed）")
    else:
        tok = DEFAULT_TOKENIZER
    if args.pairs:
        pairs = resolve_path(args.pairs)
        if pairs is None:
            fail(f"pairs 数据缺失：{args.pairs}（fail-closed）")
    else:
        pairs = DEFAULT_PAIRS

    contract = build_contract(tok_path=tok, pairs_path=pairs)
    contract_sha = canonical_sha(contract)
    entries: list[dict] = []
    blobs: list[dict] = []
    for i, p in enumerate(ck_paths):
        blob = load_ckpt(p, label=f"--anchor ckpt[{i}]")
        blobs.append(blob)
        entries.append(card_entry(ckpt_norm=norm_path(p), contract_sha=contract_sha,
                                  blob=blob, label=f"--anchor ckpt[{i}] {norm_path(p)}",
                                  scope=scope))
    mv = mv_of(contract_sha=contract_sha, entries=entries)

    snap_names = [f"{Path(p).stem}.anchor_v0.pt" for p in ck_paths]
    if len(set(snap_names)) != len(snap_names):
        fail(f"快照重名（ckpt 同名不同目录）：{sorted(snap_names)} —— 请先改名再锚定")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for e, name, blob in zip(entries, snap_names, blobs):
        payload = {
            "format": SNAPSHOT_FORMAT,
            "contract": contract,
            "contract_version": CONTRACT_VERSION,
            "mv_interface_sha256": mv["interface_sha256"],
            "ckpt": e["ckpt"],
            "seed": e["seed"], "steps": e["steps"], "maps": e["maps"],
            "loss_kind": e["loss_kind"], "shared_out": e["shared_out"],
            "weights_sha256": e["weights_sha256"],
            "state": card_state(blob["state"], scope=scope),  # 精简 state：只含接口侧前缀（input: zh_in/en_in；all: +zh_out/en_out）
        }
        e["snapshot"] = name
        e["snapshot_sha256"] = write_snapshot(out / name, payload)

    manifest = {
        "format": FORMAT,
        "contract": contract,
        "cards": entries,
        "mv": mv,
        "recipe": build_recipe(contract=contract),
        "notes": args.note if args.note else "",
    }
    if scope == SCOPE_ALL:
        manifest[SCOPE_FIELD] = SCOPE_ALL       # 只在 all 模式落盘：input 模式的字段集保持不变
    mp = out / "manifest.json"
    write_readonly_bytes(mp, dump_manifest(manifest))
    scope_txt = f" scope={scope}" if scope == SCOPE_ALL else ""    # input 模式输出行保持原样
    print(f"[ANCHOR] manifest={mp} cards={len(entries)}{scope_txt} "
          f"mv=MV({mv['interface_sha256'][:16]}, c{mv['contract_version']}, {mv['w_version']}) "
          f"mv_sha={mv['sha256'][:16]}（只读 0444，无时间戳/绝对路径）")
    for e in entries:
        extra = ""
        if scope == SCOPE_ALL:
            extra = (f" zh_out_sha256={e['zh_out_sha256']} en_out_sha256={e['en_out_sha256']}"
                     f" thinking_card={thinking_text(e['thinking_card'])}")
        print(f"[ANCHOR] card seed={e['seed']} steps={e['steps']} maps={','.join(e['maps'])} "
              f"loss={e['loss_kind']} weights_sha256={e['weights_sha256']} "
              f"snapshot={e['snapshot']} snapshot_sha256={e['snapshot_sha256']}{extra}")


# ------------------------------------------------------------------ --verify
def cmd_verify(args: argparse.Namespace) -> None:
    man, mp = load_manifest(args.verify)
    scope = man.get(SCOPE_FIELD, SCOPE_INPUT)
    contract = man["contract"]

    tok = resolve_path(str(contract["tokenizer"]), bases=(mp.parent,))
    if tok is None:
        fail(f"contract.tokenizer: 文件不存在 {contract['tokenizer']}（fail-closed）")
    pairs = resolve_path(str(contract["pairs"]), bases=(mp.parent,))
    if pairs is None:
        fail(f"contract.pairs: 文件不存在 {contract['pairs']}（fail-closed）")
    got = build_contract(tok_path=tok, pairs_path=pairs)
    for k in sorted(set(got) | set(contract)):
        if k not in contract:
            fail(f"contract.{k}: manifest 缺该字段（重算有值 {got[k]!r}）")
        if k not in got:
            fail(f"contract.{k}: manifest 有 {contract[k]!r} 但重算不产生 —— 字段来源不明")
        if contract[k] != got[k]:
            fail(f"contract.{k}: manifest={contract[k]!r} 重算={got[k]!r}")
    contract_sha = canonical_sha(got)

    entries: list[dict] = []
    for i, e in enumerate(man["cards"]):
        rp = resolve_path(str(e["ckpt"]), bases=(mp.parent,))
        if rp is None:
            fail(f"cards[{i}].ckpt: 文件不存在 {e['ckpt']}（fail-closed）")
        blob = load_ckpt(rp, label=f"cards[{i}].ckpt")
        ent = card_entry(ckpt_norm=norm_path(rp), contract_sha=contract_sha,
                         blob=blob, label=f"cards[{i}].ckpt {e['ckpt']}", scope=scope)
        fields = RECOMPUTE_CARD_FIELDS_ALL if scope == SCOPE_ALL else RECOMPUTE_CARD_FIELDS
        for k in fields:
            if e.get(k) != ent[k]:
                fail(f"cards[{i}].{k}（{e['ckpt']}）: manifest={e.get(k)!r} 重算={ent[k]!r}")
        entries.append(ent)

    got_mv = mv_of(contract_sha=contract_sha, entries=entries)
    for k in REQUIRED_MV:
        if man["mv"].get(k) != got_mv[k]:
            fail(f"mv.{k}: manifest={man['mv'].get(k)!r} 重算={got_mv[k]!r}")

    for i, e in enumerate(man["cards"]):
        sp = resolve_path(str(e["snapshot"]), bases=(mp.parent,))
        if sp is None:
            fail(f"cards[{i}].snapshot: 文件不存在 {e['snapshot']}（fail-closed）")
        got_sha = file_sha256(sp)
        if got_sha != e["snapshot_sha256"]:
            fail(f"cards[{i}].snapshot_sha256（{e['snapshot']}）: "
                 f"manifest={e['snapshot_sha256']} 重算={got_sha}")

    scope_txt = f" scope={scope}" if scope == SCOPE_ALL else ""
    half_txt = "权重(含输出卡)/思维卡" if scope == SCOPE_ALL else "权重"
    print(f"[VERIFY-OK] {mp} cards={len(entries)}{scope_txt} "
          f"mv={got_mv['interface_sha256'][:16]} c{got_mv['contract_version']} "
          f"契约/{half_txt}/mv/快照 全部逐位一致")


# ------------------------------------------------------------------ --load-check
def cmd_load_check(args: argparse.Namespace) -> None:
    man, _mp = load_manifest(args.load_check)
    cands = [e for e in man["cards"] if int(e["seed"]) == int(args.seed)]
    if args.steps is not None:
        cands = [e for e in cands if int(e["steps"]) == int(args.steps)]
    if not cands:
        avail = ", ".join(f"seed={e['seed']}/steps={e['steps']}" for e in man["cards"])
        fail(f"--load-check: seed={args.seed}"
             + (f" steps={args.steps}" if args.steps is not None else "")
             + f" 不在 manifest 里（可用：{avail}）", code=2)
    if len(cands) > 1:
        steps = ", ".join(str(e["steps"]) for e in cands)
        fail(f"--load-check: seed={args.seed} 有 {len(cands)} 张卡（steps={steps}），"
             f"用 --steps 指定其一", code=2)
    e = cands[0]

    scope = man.get(SCOPE_FIELD, SCOPE_INPUT)
    ck = resolve_path(str(e["ckpt"]), bases=(_mp.parent,))
    if ck is None:
        fail(f"cards[].ckpt: 文件不存在 {e['ckpt']}（fail-closed）")
    blob = load_ckpt(ck, label="--load-check")
    sub = card_state(blob["state"], scope=scope, label="--load-check")

    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        fail(f"--device {args.device}: CUDA 不可用（fail-closed）", code=2)
    torch.manual_seed(int(e["seed"]))          # 构造期播种（L3_PLAN §9.7：同 seed 构造可复现）
    model = MZ.build(seed=int(e["seed"])).to(dev)
    model.zh_in.load_state_dict(prefixed_state(sub, ("zh_in.",)), strict=True)
    model.en_in.load_state_dict(prefixed_state(sub, ("en_in.",)), strict=True)
    # all 模式：输出卡（MV 接口侧另一半）一并加载
    if scope == SCOPE_ALL:
        model.zh_out.load_state_dict(prefixed_state(sub, ("zh_out.",)), strict=True)
        model.en_out.load_state_dict(prefixed_state(sub, ("en_out.",)), strict=True)

    # ① 加载后逐位断言（不一致 ⇒ AssertionError，进程非零退出）
    cards = [("zh_in", model.zh_in, e["zh_in_sha256"]), ("en_in", model.en_in, e["en_in_sha256"])]
    if scope == SCOPE_ALL:
        cards += [("zh_out", model.zh_out, e["zh_out_sha256"]), ("en_out", model.en_out, e["en_out_sha256"])]
    for name, mod, want in cards:
        got = state_sha256(mod.state_dict())
        assert got == want, (
            f"[load-check] {name} 权重 sha 与 manifest 不一致：\n"
            f"  manifest={want}\n  加载后={got}")

    def combined_sha() -> str:
        st = {}
        for name, mod, _w in cards:
            st.update({f"{name}.{k}": v for k, v in mod.state_dict().items()})
        return state_sha256(st)

    half = "zh_in+en_in" if scope == SCOPE_INPUT else "zh_in+en_in+zh_out+en_out"
    assert combined_sha() == e["weights_sha256"], (
        f"[load-check] {half} 组合权重 sha 与 manifest 不一致：\n"
        f"  manifest={e['weights_sha256']}\n  加载后={combined_sha()}")

    # 思维卡（不属于 MV）：按 ckpt 张量核对 manifest（bridge 不参与模型加载，也不冻结）
    if scope == SCOPE_ALL:
        tc = thinking_entry(blob["state"])
        assert tc == e["thinking_card"], (
            f"[load-check] thinking_card 与 manifest 不一致：\n"
            f"  manifest={e['thinking_card']}\n  ckpt 重算={tc}")

    # ② 冻结开关：freeze=True ⇒ 接口卡全部 requires_grad is False；freeze=False ⇒ 全部 True
    params = [p for _n, mod, _w in cards for p in mod.parameters()]
    if args.freeze:
        for p in params:
            p.requires_grad_(False)
        for _n, mod, _w in cards:
            mod.eval()
    if args.freeze:
        assert all(not p.requires_grad for p in params), (
            f"[load-check] freeze=True 但仍有 {sum(p.requires_grad for p in params)} 个参数 requires_grad=True")
    else:
        assert all(p.requires_grad for p in params), (
            f"[load-check] freeze=False 但有 {sum(not p.requires_grad for p in params)} 个参数 requires_grad=False")
    # 冻结前后再各算一次 sha：必须仍逐位相同（冻结只改开关，不动权重）
    for name, mod, want in cards:
        assert state_sha256(mod.state_dict()) == want, f"[load-check] 处理后 {name} sha 变了"
    assert combined_sha() == e["weights_sha256"], "[load-check] 处理后组合 sha 变了"

    # ③ 人类可读锚定摘要：mv 短哈希 + 卡数 + 契约版本
    mv = man["mv"]
    freeze = bool(args.freeze)
    grad = "False" if freeze else "True"
    extra = ""
    if scope == SCOPE_ALL:
        extra = (f" zh_out/en_out requires_grad={grad} "
                 f"thinking_card={thinking_text(e['thinking_card'])}")
    print(f"[ANCHOR-LOAD] MV({mv['interface_sha256'][:16]}, c{mv['contract_version']}, {mv['w_version']}) "
          f"mv_sha={mv['sha256'][:16]} cards={len(man['cards'])} "
          f"contract_version={mv['contract_version']} seed={e['seed']} steps={e['steps']} "
          f"device={dev} freeze={freeze} "
          f"zh_in/en_in requires_grad={grad} sha=OK{extra}")


# ------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="stages/86q_anchor_v0.py",
        description="锚定 v0 模因版本：只读 manifest + 只读快照 + 加载/冻结/校验（MEME_FRAMEWORK §4.1）")
    ap.add_argument("--anchor", action="store_true", help="锚定模式（配 --ckpts 与 --out）")
    ap.add_argument("--ckpts", default="", help="逗号分隔的 ckpt 列表（--anchor 用）")
    ap.add_argument("--out", default="", help="锚定输出目录（值不入 manifest，保证逐位可复现）")
    ap.add_argument("--note", default="", help="写进 manifest.notes 的自由文本（允许为空）")
    ap.add_argument("--tokenizer", default="", help="分词器路径（默认 logs/86_tokenizer/char_tokenizer.json）")
    ap.add_argument("--pairs", default="", help="pairs.jsonl 路径（默认 logs/86_cache_zhen/pairs.jsonl）")
    ap.add_argument("--cards", choices=(SCOPE_INPUT, SCOPE_ALL), default=None,
                    help="锚定口径：input=只取输入卡 zh_in./en_in.（默认，与旧版逐位一致）；"
                         "all=再取输出卡 zh_out./en_out.（两半都进 mv），bridge.* 思维卡单独记 thinking_card（不进 mv）")
    ap.add_argument("--verify", default="", metavar="MANIFEST", help="校验模式：重算并逐位比对")
    ap.add_argument("--load-check", default="", metavar="MANIFEST", help="加载 + 冻结 + 校验模式")
    ap.add_argument("--seed", type=int, default=None, help="load-check：要加载的卡 seed")
    ap.add_argument("--steps", type=int, default=None, help="load-check：同 seed 多卡时按步数消歧")
    ap.add_argument("--device", default="cpu", help="load-check 的设备（默认 cpu）")
    ap.add_argument("--freeze", action="store_true",
                    help="load-check：冻结接口卡 requires_grad=False（input=zh_in/en_in；all 另含 zh_out/en_out）")
    ns = ap.parse_args(argv)

    modes = [m for m, v in (("--anchor", ns.anchor), ("--verify", ns.verify),
                            ("--load-check", ns.load_check)) if v]
    if len(modes) != 1:
        fail(f"必须恰好指定一种模式：--anchor | --verify <manifest> | --load-check <manifest>"
             f"（当前指定了 {len(modes)} 个）", code=2)
    mode = modes[0]
    if ns.cards is not None and mode != "--anchor":
        fail("--cards 只能配 --anchor（--verify/--load-check 按 manifest 的 cards_scope 判定口径）", code=2)
    ns.cards = ns.cards or SCOPE_INPUT          # ★ 默认口径只有一个家：input
    if mode == "--load-check":
        if ns.seed is None:
            fail("--load-check 必须配 --seed <S>", code=2)
    elif ns.freeze or ns.seed is not None or ns.steps is not None:
        fail("--freeze/--seed/--steps 只能配 --load-check", code=2)

    if mode == "--anchor":
        cmd_anchor(ns)
    elif mode == "--verify":
        cmd_verify(ns)
    else:
        cmd_load_check(ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
