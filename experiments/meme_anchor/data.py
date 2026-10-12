"""M1 数据层：语料切句 / 抽样切分 / 渲染成图 / 落盘缓存。

判据与规格见 `PREREG.md` §1（数据与切分）与 §2（渲染规格）——本文件是它们的实现，不许在这里发明新规则。

只读 `nanoSeek/data/chinese/clean_v3/*.txt` 与 `logs/86_tokenizer/char_tokenizer.json`；只写 `logs/86_cache/`。
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import numpy as np

ROOT = Path("/home/vesita/coding/my/flowme")
NANOSEEK = Path("/home/vesita/coding/my/nanoSeek/data/chinese")
CORPUS = [NANOSEEK / "clean_v3" / "wikipedia_cn.txt", NANOSEEK / "clean_v3" / "c4_zh.txt"]
# ★ 词表改由本仓库持有（2026-10-12，本机 = noxy/RTX 5060）：nanoSeek 的 `char_tokenizer.json` 不入库、本机没有，
#   而它是 nanoSeek 自己 `train.bin` 的依赖 ⇒ **不在那边重建**。本文件由 nanoSeek 自带实现的同一构建器生成
#   （`data/chinese/train_tokenizer.py` 的 `build_char_wordlevel`，规格 `hanzi_top=7808`；重建命令与 sha256 见
#   `logs/86_tokenizer/`）。默认档 `hanzi_top=4400` 在本机只保留 32026 对中英配对 < 本线规格 `TRAIN_N+TEST_N=42245`，
#   7808 档保留 46636 对 ⇒ 取 7808。★ 特殊符 id 与本文契约逐位一致（128 `<eos>` / 129 `<unk>` / 131 `<pad>` / 132 `<bos>`）。
TOKENIZER = ROOT / "logs" / "86_tokenizer" / "char_tokenizer.json"
CACHE = ROOT / "logs" / "86_cache"

# PREREG §2 渲染规格（写死）
# ★ 订正 4（2026-10-11 01:4x，独立复核 P0-3 实测逼出）：旧版按 FONT_SIZE=24px 排行距，
#   而该字体实测 ascent=28 / descent=7 ⇒ 行盒 35px ⇒ 60 字句第三行被裁 7px（只裁长句）。
#   新版：LINE_ADVANCE=40（≥ 行盒 35），画布高 88 → 128，并加「无裁切」断言。
FONT = "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"
FONT_SIZE, MARGIN, LINE_GAP, MAXCH = 24, 4, 4, 24
N_LINES = 3
LINE_ADVANCE = 40                                                 # 行盒 35 + 余量 5
IMG_H = MARGIN * 2 + N_LINES * LINE_ADVANCE                       # 128
IMG_W = MARGIN * 2 + MAXCH * FONT_SIZE                            # 584
IMG_SHAPE = (IMG_H, IMG_W)
assert IMG_H % 8 == 0 and IMG_W % 8 == 0, "画布必须能被 CNN 的 stride 8 整除"

# 词表契约（实测自 char_tokenizer.json 的 added tokens）
VOCAB = 8192
BOS, EOS, PAD, UNK = 132, 128, 131, 129

SENT_SEED = 20261011                  # PREREG §1 抽样种子（写死）
BYTES_PER_FILE = 40_000_000           # 每个文件读多少字节（够 2.2 万条）
N_TRAIN, N_TEST = 20000, 2000

HAN = re.compile(r"[\u4e00-\u9fff]")
SENT_SPLIT = re.compile(r"(?<=[。！？；])")


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZER))


def _acceptable(s: str, tk) -> bool:
    """PREREG §1 合格句三条件：长度 8..60 · 含 ≥6 汉字 · 逐字符可编码且逐字还原。"""
    if not (8 <= len(s) <= 60):
        return False
    if len(HAN.findall(s)) < 6:
        return False
    return tk.decode(tk.encode(s).ids) == s


def build_sentences(force: bool = False) -> dict:
    """切句 → 过滤 → 去重 → 固定种子 shuffle → 切 train/held-out；结果落盘可复现。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    meta_p = CACHE / "sent_meta.json"
    if meta_p.exists() and not force:
        return json.loads(meta_p.read_text(encoding="utf-8"))

    tk = load_tokenizer()
    raw: list[str] = []
    per_file: dict[str, int] = {}
    for p in CORPUS:
        assert p.exists(), f"语料不存在：{p}"
        with open(p, encoding="utf-8", errors="ignore") as f:
            txt = f.read(BYTES_PER_FILE)
        got = [s.strip() for s in SENT_SPLIT.split(txt)]
        keep = [s for s in got if _acceptable(s, tk)]
        per_file[p.name] = len(keep)
        raw += keep

    # 逐字去重（PREREG §1）
    seen: set[str] = set()
    uniq: list[str] = []
    dup = 0
    for s in raw:
        if s in seen:
            dup += 1
            continue
        seen.add(s)
        uniq.append(s)

    rng = random.Random(SENT_SEED)
    rng.shuffle(uniq)
    if len(uniq) < N_TRAIN + N_TEST:
        raise SystemExit(f"合格句不足：去重后 {len(uniq)} < {N_TRAIN + N_TEST}")
    train = uniq[:N_TRAIN]
    test = uniq[N_TRAIN:N_TRAIN + N_TEST]

    assert not (set(train) & set(test)), "[S3] train/held-out 必须逐字不重叠"
    meta = {
        "seed": SENT_SEED, "bytes_per_file": BYTES_PER_FILE, "n_per_file": per_file,
        "n_raw": len(raw), "n_dup_dropped": dup, "n_uniq": len(uniq),
        "n_train": len(train), "n_test": len(test),
        "font": FONT, "img_shape": list(IMG_SHAPE),
        "train": train, "test": test,
    }
    meta_p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


_FONT_CACHE: dict = {}


def render(text: str) -> np.ndarray:
    """PREREG §2（订正 4）：固定画布 uint8 [128,584]，白底黑字，确定性渲染，**不许裁切**。"""
    from PIL import Image, ImageDraw, ImageFont

    if "font" not in _FONT_CACHE:
        _FONT_CACHE["font"] = ImageFont.truetype(FONT, FONT_SIZE)
    img = Image.new("L", (IMG_W, IMG_H), 255)
    draw = ImageDraw.Draw(img)
    lines = [text[i:i + MAXCH] for i in range(0, len(text), MAXCH)][:N_LINES]
    for i, ln in enumerate(lines):
        draw.text((MARGIN, MARGIN + i * LINE_ADVANCE), ln, fill=0, font=_FONT_CACHE["font"])
    a = np.asarray(img, dtype=np.uint8)
    assert a.shape == IMG_SHAPE, f"[渲染] 形状 {a.shape} != {IMG_SHAPE}"
    # ★ 无裁切断言：最后一行/最后一列必须全白（贴边即说明有内容被切掉）
    assert a[-1].max() == 255 and a[:, -1].max() == 255, \
        f"[渲染] 内容被裁切（长度 {len(text)}）：末行 max={a[-1].max()} 末列 max={a[:, -1].max()}"
    return a


def ink_bbox(text: str) -> tuple[int, int, int, int]:
    """墨迹包围盒（诊断用）：返回 (r0, r1, c0, c1)。"""
    a = render(text)
    rows = np.where((a < 250).any(axis=1))[0]
    cols = np.where((a < 250).any(axis=0))[0]
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def build_images(force: bool = False) -> dict:
    """把 train/held-out 的句子渲染成 uint8 大数组（mmap 读回）。"""
    meta = build_sentences(force=force)
    out = {}
    for split in ("train", "test"):
        p = CACHE / f"img_{split}.npy"
        n = len(meta[split])
        if p.exists() and not force:
            arr = np.load(p, mmap_mode="r")
            assert arr.shape == (n, *IMG_SHAPE), f"[缓存] {p} 形状 {arr.shape} != {(n, *IMG_SHAPE)}"
            out[split] = str(p)
            continue
        buf = np.lib.format.open_memmap(p, mode="w+", dtype=np.uint8, shape=(n, *IMG_SHAPE))
        for i, s in enumerate(meta[split]):
            buf[i] = render(s)
        buf.flush()
        del buf
        out[split] = str(p)
    return out


def load_images(split: str, n: int | None = None) -> np.ndarray:
    p = CACHE / f"img_{split}.npy"
    arr = np.load(p, mmap_mode="r")
    return arr if n is None else arr[:n]


def encode_sentences(sents: list[str]) -> list[list[int]]:
    tk = load_tokenizer()
    return [tk.encode(s).ids for s in sents]


def prepare(smoke: bool = False, force: bool = False) -> dict:
    """跑前把数据与图都备好；返回可直接用的批次。"""
    meta = build_sentences(force=force)
    build_images(force=force)
    n_tr = 2000 if smoke else N_TRAIN
    n_te = 500 if smoke else N_TEST
    train, test = meta["train"][:n_tr], meta["test"][:n_te]
    return {
        "train": train, "test": test,
        "train_ids": encode_sentences(train), "test_ids": encode_sentences(test),
        "train_img": load_images("train", n_tr), "test_img": load_images("test", n_te),
        "meta": {k: v for k, v in meta.items() if k not in ("train", "test")},
    }


if __name__ == "__main__":                                  # 只读自检：python data.py
    m = build_sentences()
    print(json.dumps({k: v for k, v in m.items() if k not in ("train", "test")},
                     ensure_ascii=False, indent=1))
    r1, r2 = render(m["train"][0]), render(m["train"][0])
    print("[S1] 渲染确定性 max|Δ| =", int(np.abs(r1.astype(int) - r2.astype(int)).max()))
    print("样例句：", m["train"][0])
    print("像素统计：白", int((r1 == 255).sum()), "黑", int((r1 == 0).sum()), "唯一值", len(np.unique(r1)))
    assert os.environ.get("M1_SKIP_IMG") == "1" or True
