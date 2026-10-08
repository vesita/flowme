"""Blackboard —— 卡间路由 + 显式记录「谁读了谁」。

本轮路由写死为 **input → target → output**（不做路由学习）：

    外部 ids ──▶ [input] ──meme[n,d]──▶ [target] ──meme[n,d]──▶ [output] ──▶ logits[n,V]

黑板的读日志是本轮交付物的一部分：每一次 read 都记下 (reader, key, writer, version)
—— writer 是**写入该 key 的卡**，version 是写入时该卡的接口版本（契约 sha 前 16 位）。
「谁读了谁」= 去重后的 (reader, writer) 对，可直接断言成 input→target→output 这条链。
"""
from __future__ import annotations

from dataclasses import dataclass, field

EXTERNAL = "外部输入"
READ_KEYS = ("ids", "meme", "mask", "logits")


@dataclass(frozen=True)
class Read:
    reader: str        # 谁读
    key: str           # 读了哪个槽
    writer: str        # 谁写的（这张卡上次写这个槽）
    version: str       # 写入时的接口版本（契约 sha 前 16 位）

    def as_dict(self) -> dict:
        return dict(reader=self.reader, key=self.key, writer=self.writer, version=self.version)


@dataclass
class Blackboard:
    """卡间唯一的数据通道；forward 一次 = 清空一次 + 记一遍读日志。"""

    read_log: list[Read] = field(default_factory=list)
    write_log: list[tuple] = field(default_factory=list)
    _slots: dict = field(default_factory=dict, repr=False)
    _producer: dict = field(default_factory=dict, repr=False)
    _version: dict = field(default_factory=dict, repr=False)

    def clear(self) -> None:
        self.read_log.clear()
        self.write_log.clear()
        self._slots.clear()
        self._producer.clear()
        self._version.clear()

    def write(self, key: str, value, writer: str, version: str = "-") -> None:
        self._slots[key] = value
        self._producer[key] = writer
        self._version[key] = version
        self.write_log.append((writer, key, version))

    def read(self, key: str, reader: str):
        assert key in self._slots, f"[黑板] {reader} 想读 {key!r}，但它还没被写过"
        assert reader != self._producer[key], (
            f"[黑板] {reader} 想读自己刚写的 {key!r} —— 卡间路由必须跨卡")
        rec = Read(reader=reader, key=key, writer=self._producer[key], version=self._version[key])
        self.read_log.append(rec)
        return self._slots[key]

    def who_reads_whom(self) -> list:
        """「谁读了谁」= 去重保序的 (reader, writer) 对。"""
        seen, out = set(), []
        for r in self.read_log:
            if (r.reader, r.writer) not in seen:
                seen.add((r.reader, r.writer))
                out.append((r.reader, r.writer))
        return out

    def read_table(self) -> list:
        return [r.as_dict() for r in self.read_log]

    def read_chain(self) -> str:
        return " → ".join(f"{w}[{k}]→{r}" for r, k, w, _ in
                          [(x.reader, x.key, x.writer, x.version) for x in self.read_log])
