"""内置任务包：**import 即注册**（每个包的 `__init__` 末尾调用 `register(CARD)`）。

顺序即注册顺序，保持与历史一致（字典序），这样 `all_tasks()` 的迭代顺序不变。
"""
from dtseek.tasks.builtin import (
    cloze_fill,
    idiom,
    negation,
    ownership,
    person,
    pronoun,
    relation,
    reply_pick,
    sentiment,
)

__all__ = ["cloze_fill", "idiom", "negation", "ownership", "person", "pronoun", "relation",
           "reply_pick", "sentiment"]
