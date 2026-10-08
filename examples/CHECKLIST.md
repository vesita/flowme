# CHECKLIST —— 这个 example 成立的理由（逐条指向已入库证据，无编造）

| # | 结论 | 证据来源（仓内已有）| ckpt sha256[:16] |
|---|---|---|---|
| 1 | 五桶地板锚（S14 单变量训练）| `stages/14_synth_arith.py`、`logs/14_synth.log`、`logs/14_ckpt_*` | `logs/14_ckpt_add_3d_seed1234.pt` |
| 2 | 残差单变量 2 seed（S15）| `stages/15_iteration.py`、`logs/15_results.jsonl`、`logs/15_iter.log` | `logs/15_ckpt_add3d_always_k1_seed1234.pt` = `abdd9318978f59b9` |
| 3 | 五桶 × 2 臂 × 2 seed 对照判决（S19）| `stages/19_residual_scan.py`、`logs/19_results.jsonl`（20 行）| 见 4–8 |
| 4 | add_3d 无残差 EM 2.50% | `logs/19_results.jsonl` | `7b0228af3f7238d0` |
| 5 | add_3d 带残差 EM 37.25%（+34.75pp）| `logs/19_results.jsonl` | `854903674f9b952e` |
| 6 | add_1d 无残差 EM 96.62% | `logs/19_results.jsonl` | `59284de1ca2364a3` |
| 7 | add_1d 带残差 EM 96.88% | `logs/19_results.jsonl` | `6cd0000899dd97f3` |
| 8 | add_3d seed5678：9.62% → 31.13%（两 seed 同号复现）| `logs/19_results.jsonl` | `0fb3d71489c3ff3f` / `e85aa1ff5a1e13b9` |
| 9 | R28 批内仅 1–3/16 一致 ⇒ batch>1 口径作废 | `logs/19_results.jsonl` 的 `r28_same` / `r28_k`；`methodology/RULES.md` R28 | — |
| 10 | 门 A / B / C 依据 | `methodology/RULES.md` R7、R15；`MEME_FRAMEWORK.md` §12、§13.2、§5 L5 | — |

**可核对性**：`logs/19_results.jsonl` 每行含 bucket / arm / seed / em / sha / ckpt，20 行齐全；
本 example 的 CPU 重算值与之逐位相同（add_1d 96.62 / 96.88，add_3d 2.50 / 37.25）。
**不声称**：普遍增益、会对话 / 会推理；GSM8K 仍未过地板。
