"""把 A′ 一体 ckpt 里的基座拆成标准基座产物（只写 experiments/additivity/）。

用途：PREREG 里的 F-A′ / B-A′ 校准档 —— 文档 frozen 基线 0.6383 就是在 A′ 基座上测的，
拿到同一块基座才能把配方复现接上。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from dtseek.tasks.artifacts import save_base  # noqa: E402
from dtseek.encoder.nano_doc_encoder import NanoDocEncoder  # noqa: E402

src = ROOT / "checkpoints" / "arm_aprime_dtseek.pt"
dst = Path(__file__).resolve().parent / "base_aprime.pt"

ap = torch.load(src, map_location="cpu", weights_only=False)
vocab = torch.load(ROOT / "checkpoints" / "base_encoder.pt",
                   map_location="cpu", weights_only=False)["vocab_size"]
kwargs = dict(ap["encoder_kwargs"])
enc = NanoDocEncoder(vocab_size=vocab, hidden_dim=ap["hidden_dim"], dropout=0.0, **kwargs)
missing = enc.load_state_dict(ap["doc_encoder"], strict=True)
save_base(dst, enc, hidden_dim=ap["hidden_dim"], vocab_size=vocab, encoder_kwargs=kwargs,
          meta={"split_from": str(src), "purpose": "E-B bypass 校准档（与文档 frozen 0.6383 同基座）"})
print(f"OK 已写出 {dst}（load_state_dict: {missing}）")
