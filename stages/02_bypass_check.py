"""P26b stage2: sequence-order-reconstruction task (copy cannot win) + V_mid capacity scan.

Task: input card receives a RANDOMLY SHUFFLED token sequence; target = original
(canonical sorted) order.  Copy (output=input) therefore cannot score.
Two arms differ ONLY in the inter-card representation:
  M: [L,128] tensor passed straight through
  T: argmax(Linear(128, V_mid)) -> Emb_mid(V_mid,128)
Run (CPU): CUDA_VISIBLE_DEVICES="" uv run --no-sync python meme/stage2.py
"""
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

V, L, D, HEADS = 64, 16, 128, 4
STEPS, BS, NTR, NTE, LR = 3000, 64, 8000, 2000, 1e-3
SEEDS = [42]
VMIDS = [8, 32, 128, 1024]          # capacity scan (M arm: no capacity cap)
DATA_SEED = 42
GATE_FAIL_CE = 4.0                  # copy below this => task trivialised
IDENT_FAIL_CE = 1.0                 # identity-mid below this => mediator bypassable
LOG = "logs/stage2.jsonl"
torch.set_num_threads(4)
torch.set_num_interop_threads(1)


def log(**kw):
    line = json.dumps(kw, ensure_ascii=False)
    print(line, flush=True)
    os.makedirs("logs", exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def ce(logits, y):
    """CE over token axis: logits [N,L,V], target [N,L]."""
    return F.cross_entropy(logits.transpose(1, 2), y)


def make_data(n, seed):
    """(shuffled input, original target); target is a canonical fn of the multiset."""
    g = torch.Generator().manual_seed(seed)
    orig = torch.randint(0, V, (n, L), generator=g).sort(1).values
    perm = torch.argsort(torch.rand(n, L, generator=g), dim=1)
    return orig.gather(1, perm), orig


class InputCard(nn.Module):
    """Emb(64,128) + position encoding + 1 TransformerEncoder layer -> [L,128]"""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(L, D)
        self.enc = nn.TransformerEncoderLayer(D, HEADS, batch_first=True)

    def forward(self, x):
        return self.enc(self.emb(x) + self.pos(torch.arange(L, device=x.device)))


class ThoughtCard(nn.Module):
    """1 TransformerEncoder layer: [L,128] -> [L,128]"""

    def __init__(self):
        super().__init__()
        self.enc = nn.TransformerEncoderLayer(D, HEADS, batch_first=True)

    def forward(self, h):
        return self.enc(h)


class OutputCard(nn.Module):
    """Linear(128,64) -> logits"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(D, V)

    def forward(self, h):
        return self.fc(h)


class Arm(nn.Module):
    """v_mid=None -> M (tensor direct); v_mid=K -> T (discretise to K symbols).
    identity=True -> thought card replaced by identity (gate / P5)."""

    def __init__(self, v_mid=None, identity=False):
        super().__init__()
        self.in_card = InputCard()
        self.v_mid = v_mid
        if v_mid:
            self.proj = nn.Linear(D, v_mid)
            self.mid_emb = nn.Embedding(v_mid, D)
        self.thought = nn.Identity() if identity else ThoughtCard()
        self.out_card = OutputCard()

    def forward(self, x):
        h = self.in_card(x)
        if self.v_mid:
            h = self.mid_emb(self.proj(h).argmax(-1))
        return self.out_card(self.thought(h))


def train_and_eval(v_mid, identity, seed, tag):
    torch.manual_seed(seed)
    all_x, all_y = make_data(NTR + NTE, DATA_SEED)
    tr_x, tr_y = all_x[:NTR], all_y[:NTR]
    te_x, te_y = all_x[NTR:], all_y[NTR:]
    m = Arm(v_mid, identity)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed)
    ar = torch.arange(NTR)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        idx = ar[torch.randint(NTR, (BS,), generator=g)]
        loss = ce(m(tr_x[idx]), tr_y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            log(tag=tag, kind="train", step=step, loss=round(loss.item(), 4),
                sec=round(time.time() - t0, 1))
    m.eval()
    with torch.no_grad():
        logits = m(te_x)
        tce = ce(logits, te_y).item()
        acc = (logits.argmax(-1) == te_y).float().mean().item()
    r = dict(tag=tag, kind="result", v_mid=v_mid, identity=identity, seed=seed,
             test_ce=round(tce, 4), test_acc=round(acc, 4),
             sec=round(time.time() - t0, 1))
    log(**r)
    return r


def deg_baseline(pred, te_y):
    """Untrained degenerate predictor: one-hot logits with scale 10."""
    logits = 10.0 * F.one_hot(pred, V).float()
    return dict(test_ce=round(ce(logits, te_y).item(), 4),
                test_acc=round((pred == te_y).float().mean().item(), 4))


def main():
    _, all_y = make_data(NTR + NTE, DATA_SEED)
    te_y = all_y[NTR:]
    te_x, _ = make_data(NTR + NTE, DATA_SEED)
    te_x = te_x[NTR:]
    # ---- GATE (must pass before any capacity scan) ----
    log(kind="gate", name="uniform_ref", test_ce=round(math.log(V), 4))
    copy_r = deg_baseline(te_x.clone(), te_y)
    log(kind="gate", name="copy", **copy_r)
    g = torch.Generator().manual_seed(0)
    rnd = te_x.gather(1, torch.argsort(torch.rand(te_x.shape, generator=g), dim=1))
    log(kind="gate", name="random_perm", **deg_baseline(rnd, te_y))
    ident = train_and_eval(None, True, SEEDS[0], "gate_identity_m")
    log(kind="gate", name="identity_mid", **{k: ident[k] for k in ("test_ce", "test_acc")})
    gate_pass = (copy_r["test_ce"] >= GATE_FAIL_CE) and (ident["test_ce"] >= IDENT_FAIL_CE)
    log(kind="gate_verdict", gate_pass=gate_pass,
        rule="copy CE >= 4.0 AND identity_mid CE >= 1.0")
    if not gate_pass:
        log(kind="abort", reason="trivial solution dominates: gate failed")
        return
    # ---- capacity scan ----
    for seed in SEEDS:
        for k in VMIDS:
            train_and_eval(k, False, seed, f"T{k}")
        train_and_eval(None, False, seed, "M")
    log(kind="done")


if __name__ == "__main__":
    main()
