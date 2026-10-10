"""harness —— flowme 的**推理入口**（L3_PLAN.md §21）。

不是 demo 脚本，而是"下一阶段的共同测量面"：换卡 / 加卡 / 精度 / 检查点 / 探针都从同一个入口走。

    harness.registry   卡片注册表（名字 → 构造器/接口/可判目标/依赖/ckpt）
    harness.pipeline   按「卡片集合 + 拓扑」装配可推理流水线（顺序执行；卡间 = d 维隐态）
    harness.chat       REPL：读自然语言 → 逐轮生成回复（历史累积）
    harness.probe      推理探针：卡数 N / 接口宽度 w / dtype → 驻留·延迟·保真
    harness.cli        子命令：chat / say / selfcheck / probe / list-cards

★ 诚实边界（README 有完整版，启动横幅也会打印）：
  · 当前唯一的"卡"是**同构 encoder 层卡**（我们没有第二种尺寸/功能的卡）；
  · 对话能力**限于自造合成分布**（5 族 × 20 DET 模板，`logs/76_*`）⇒ 分布内流畅正确、
    分布外会给"看着像但内容错"的回复；
  · **动态加载尚未实现** —— `probe` 的顺序执行是它的**极限情形**（最坏延迟、最小驻留）。
"""
from __future__ import annotations

__version__ = "1.0.0"

from .pipeline import (CKPT_B2_K8_S1234, CKPT_B2_K8_S5678, D, DATA_TEST_K8, DATA_TRAIN_K8,
                       Pipeline, RECORDS, build_custom, build_pipeline, compare_records,
                       data_fp, eval_records, file_sha256, greedy_gen, greedy_trace, load_jsonl,
                       load_model, load_tok, prep_record, select_readout, tensor_digest)
from .registry import CARD_SETS, REGISTRY, CardSpec, parse_cards

__all__ = ["CKPT_B2_K8_S1234", "CKPT_B2_K8_S5678", "CARD_SETS", "D", "DATA_TEST_K8",
           "DATA_TRAIN_K8", "Pipeline", "RECORDS", "REGISTRY", "CardSpec", "build_custom",
           "build_pipeline", "compare_records", "data_fp", "eval_records", "file_sha256",
           "greedy_gen", "greedy_trace", "load_jsonl", "load_model", "load_tok", "parse_cards",
           "prep_record", "select_readout", "tensor_digest", "__version__"]
