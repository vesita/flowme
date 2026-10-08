"""version —— 只读快照 + 模因版本 MV（§7.9 快照协议的可执行实现）。

    MV = (接口快照 sha256, 契约版本, 对齐矩阵 W 版本占位)

  · 接口快照 sha256 = hash(契约, 输入卡权重, 输出卡权重)  ← 接口 = 输入卡 + 输出卡（§7.1）
  · 契约 = d · n 语义 · mask 约定 · 位置编码 · MAXLEN（任一变了就是新 MV，§7.9②）
  · W 版本 = 对齐矩阵占位（本轮不实现路由/对齐，故是常量字符串，但位子留在 MV 里）

  · `freeze()` 只做软冻结（权重不动）；**硬版本化** = `snapshot()` 落只读文件 + sha256（§7.8）。
  · 快照文件 chmod 0o444 只读；同权重两次快照的 sha256 必然相同（载荷里没有时间戳/路径）。
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

from .base import CONTRACT_VERSION, Card, Contract, state_sha256

SNAPSHOT_FORMAT = "cards-snapshot-v1"
W_VERSION = "W@v0-placeholder"        # ★ 对齐矩阵 W 版本占位（本轮 w 未实现）
INTERFACE_CARDS = ("input", "output")  # 接口 = 输入卡 + 输出卡
TARGET_CARDS = ("target",)


@dataclass(frozen=True)
class MV:
    """模因版本：契约版本 + 接口快照 sha256 + 对齐矩阵 W 版本占位。"""

    interface_sha256: str
    contract_version: int = CONTRACT_VERSION
    w_version: str = W_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    def sha256(self) -> str:
        blob = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def short(self) -> str:
        return f"MV({self.interface_sha256[:16]}, c{self.contract_version}, {self.w_version})"


def make_mv(contract: Contract, interface_sha256: str, w_version: str = W_VERSION) -> MV:
    return MV(interface_sha256=interface_sha256, contract_version=contract.version,
              w_version=w_version)


def freeze(card: Card) -> Card:
    """软冻结一张卡（权重不再更新）。"""
    return card.freeze()


# --------------------------------------------------------------------- 快照
@dataclass(frozen=True)
class Snapshot:
    path: str
    sha256: str
    mv: MV
    payload: dict

    def to_dict(self) -> dict:
        return dict(path=self.path, sha256=self.sha256, mv=self.mv.to_dict(), payload=self.payload)


def _snapshot_payload(obj, kind: str, name: str, cards: dict) -> dict:
    contract = obj.contract
    if kind == "pipeline":
        iface_sha = obj.interface_sha256()
    else:
        iface_sha = obj.snapshot_sha256()
    mv = make_mv(contract, iface_sha)
    return {
        "format": SNAPSHOT_FORMAT,                     # 无时间戳/无路径 ⇒ 同权重 ⇒ 同 sha
        "kind": kind,
        "name": name,
        "contract": contract.to_dict(),
        "contract_version": contract.version,
        "interface_sha256": iface_sha,
        "w_version": W_VERSION,
        "mv": mv.to_dict(),
        "mv_sha256": mv.sha256(),
        "cards": {k: {"card_name": c.card_name, "io": c.io, "d": c.d,
                      "maxlen": c.maxlen, "n_params": c.n_params(),
                      "weights_sha256": c.weights_sha256(),
                      "snapshot_sha256": c.snapshot_sha256(),
                      "frozen": c.is_frozen} for k, c in cards.items()},
        "frozen_cards": list(getattr(obj, "frozen_cards", ()) or
                             (() if not getattr(obj, "is_frozen", False) else (name,))),
    }


def snapshot(obj, path: str | Path) -> Snapshot:
    """把卡/管线的接口载荷写成**只读** JSON + sha256（同权重两次 ⇒ sha 相同）。"""
    p = Path(path)
    if isinstance(obj, Card):
        kind, name, cards = "card", obj.card_name, {obj.card_name: obj}
    else:
        kind, name, cards = "pipeline", "CardPipeline", obj.cards()
    payload = _snapshot_payload(obj, kind, name, cards)
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    if p.exists():
        p.chmod(0o644)                                  # 允许覆盖旧快照
    p.write_bytes(blob + b"\n")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    p.chmod(0o444)                                      # ★只读
    return Snapshot(path=str(p), sha256=digest, mv=MV(**payload["mv"]), payload=payload)


def load_snapshot(path: str | Path) -> Snapshot:
    """读只读快照并校验 sha256（内容被动过就当场炸）。"""
    p = Path(path)
    raw = p.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw.decode("utf-8"))
    assert payload.get("format") == SNAPSHOT_FORMAT, f"[快照] 格式不对：{payload.get('format')!r}"
    assert payload["mv_sha256"] == MV(**payload["mv"]).sha256(), "[快照] MV 字段被改过"
    mv = MV(**payload["mv"])
    return Snapshot(path=str(p), sha256=digest, mv=mv, payload=payload)


def snapshot_is_readonly(path: str | Path) -> bool:
    mode = os.stat(Path(path)).st_mode
    return not (mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def mv_of(obj) -> MV:
    """从卡/管线现算 MV（不落盘）。"""
    if isinstance(obj, Card):
        return make_mv(obj.contract, obj.snapshot_sha256())
    return make_mv(obj.contract, obj.interface_sha256())


def interface_sha256(cards: dict, contract: Contract) -> str:
    """接口快照 sha256 = hash(契约, 接口卡权重)（interface cards 名排序）。"""
    h = hashlib.sha256()
    h.update(b"interface-snapshot-v1|")
    h.update(contract.sha256().encode("utf-8"))
    for name in INTERFACE_CARDS:
        card = cards[name]
        h.update(b"|")
        h.update(name.encode("utf-8"))
        h.update(state_sha256(card.state_dict()).encode("utf-8"))
    return h.hexdigest()
