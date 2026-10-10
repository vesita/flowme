#!/usr/bin/env python3
"""S86d · **还原演示**：teacher forcing ≠ 自由生成 —— 这里做**贪心自回归**还原，直接看句子。

为什么要有这个脚本：跑批里的「字级准确率 / 逐句精确」是 **teacher forcing**（给正确前缀、逐位预测下一个字），
它**不是**"从 <bos> 自己往下写"。本脚本对同一批 held-out 句，从 `<bos>` 起**贪心解码**到 `<eos>`，
打印 原句 / 文本路还原 / 图片路还原。**只作定性演示，不下判据。**

    HSA_OVERRIDE_GFX_VERSION=10.3.0 .venv/bin/python stages/86d_recon_demo.py [ckpt1,ckpt2,...]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path("/home/vesita/coding/my/flowme")
sys.path.insert(0, str(ROOT / "experiments" / "meme_anchor"))
import data as DAT          # noqa: E402
import model as MD          # noqa: E402
from data import BOS, EOS   # noqa: E402

_spec = importlib.util.spec_from_file_location("m1driver", ROOT / "stages" / "86_meme_anchor.py")
DRV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DRV)

DEFAULT = ["86_ckpt_detr_l0.0_s1234.pt",           # 文本/图片路都看（λ=0）
           "86_ckpt_detr_l0.5_s9012.pt"]           # 补救轮 30000 步（λ=0.5）


@torch.no_grad()
def greedy(model, meme: torch.Tensor, tk, maxlen: int = 64) -> str:
    """从 <bos> 起贪心自回归解码到 <eos>（不喂任何目标前缀）。"""
    ids = torch.full((1, 1), BOS, dtype=torch.long, device=meme.device)
    out: list[int] = []
    for _ in range(maxlen):
        logits = model.logits(meme, ids)
        nxt = int(logits[0, -1].argmax())
        if nxt == EOS:
            break
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=meme.device)], dim=1)
    return tk.decode(out)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pack = DRV.get_pack()
    tk = DAT.load_tokenizer()
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else DEFAULT
    n = 6
    print(f"[DEMO] device={device} held-out 前 {n} 句 · 贪心自回归（无 teacher forcing）", flush=True)
    for name in names:
        blob = torch.load(ROOT / "logs" / name, map_location="cpu", weights_only=False)
        model = MD.build(blob["seed"], blob["vision_arch"])
        model.load_state_dict(blob["state"])
        model.to(device).eval()
        ok_t = ok_i = 0
        print(f"\n==== {name}（λ={blob['lam']} seed={blob['seed']} steps={blob['steps']}）====", flush=True)
        for k in range(n):
            gold = pack["test"][k]
            ids = [pack["test_ids"][k]]
            mt = model.meme_text(DRV.pad_ids(ids, device))
            mi = model.meme_image(DRV.img_batch(pack["test_img"], k, device))
            rt, ri = greedy(model, mt, tk), greedy(model, mi, tk)
            ok_t += int(rt == gold)
            ok_i += int(ri == gold)
            print(f"  原句  : {gold}")
            print(f"  文本路: {rt}   {'✓ 整句相同' if rt == gold else '✗'}")
            print(f"  图片路: {ri}   {'✓ 整句相同' if ri == gold else '✗'}")
        print(f"  ⇒ 这 {n} 句里整句全对：文本路 {ok_t}/{n} · 图片路 {ok_i}/{n}", flush=True)


if __name__ == "__main__":
    main()
