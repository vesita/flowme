"""骨架迁移映射表（P13 §2 的机器可读版；与 `PREREG.md` §4 的跑前裁定逐字对应）。

card_flow 的 25 条 → 唯一家 `src/dtseek/tasks/render.py::SKELETONS`，三种处置：

- **merged**：与官方某条**签名完全相同** ⇒ 并进官方 id（不再另立）；
- **migrated**：进表，id 改到 `CF##` 命名空间（`##` = 原 card_flow 编号）；
- **deprecated**：含 `否` / `数` 槽，不在官方 `POS_TYPES={名,动,形}` ⇒ 过不了
  `slot_schema_problems` 门禁，**不进表**（原因逐条写在 `reason`）。

**签名** = `(pattern, ((pos, theta), ...))` —— V1 的「同 id 不同签名」就用它算。
`v1_report()` 给出 V1 的四个实测数字（家内唯一 / 命名空间相交 / 迁移后签名一致 / 不改名会撞多少）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in ("src", "experiments/card_flow"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

from dtseek.tasks import render as R  # noqa: E402

CF_SLOT_RE = re.compile(r"\[([a-z]\d+)\]")
_TYPE_OF_PREFIX = {"n": "名", "v": "动", "a": "形", "g": "否", "d": "数"}

#: 旧 id → (处置, 新 id 或 None, 旧模板, 原因/说明)
MIGRATION: dict[str, tuple[str, str | None, str, str]] = {
    # —— 并进官方（签名完全相同）——
    "S13": ("merged", "S01", "[n1][v1][n2]。",
            "与官方 S01 签名相同（名·施事/动/名·受事 + `。`）⇒ 并入，不再另立"),
    "S14": ("merged", "S02", "[n1][v1][n2]吗？", "与官方 S02 签名相同 ⇒ 并入"),
    "S12": ("merged", "S04", "[n1][v1]。", "与官方 S04 签名相同（名·施事/动）⇒ 并入"),
    # —— 进表（CF 命名空间）——
    "S01": ("migrated", "CF01", "[n1]。", "单名槽复述句；与官方 S01 **同 id 不同签名** ⇒ 改名"),
    "S02": ("migrated", "CF02", "[n1][a1]。", "题元声明按真卡口径丢弃（A 集 48/48 theme=None）"),
    "S03": ("migrated", "CF03", "[n1][a1]吗？", "与官方 S03 `[1][2][3]！` 同 id 不同签名 ⇒ 改名"),
    "S06": ("migrated", "CF06", "[n1]和[n2]。", "与官方 S06 `[1]们[2]。` 同 id 不同签名 ⇒ 改名"),
    "S07": ("migrated", "CF07", "[n1]与[n2]。", "与官方 S07 `[1]和[2][3]。` 同 id 不同签名 ⇒ 改名"),
    "S08": ("migrated", "CF08", "[n1]和[n2]和[n3]。", "与官方 S08 同 id 不同签名 ⇒ 改名"),
    "S09": ("migrated", "CF09", "[n1][a1]，[n2][a2]。", "与官方 S09 `[1][2]吗？` 同 id 不同签名 ⇒ 改名"),
    "S11": ("migrated", "CF11", "[n1]吗？",
            "与官方 S11 同 id 不同签名 ⇒ 改名（A 集 25/36 条用它）"),
    "S18": ("migrated", "CF18", "[n1][a1]，但[n2][a2]。",
            "逻辑词 `但` 走词面守卫（输入逐字有据 / 意图卡声明）"),
    "S19": ("migrated", "CF19", "[n1][a1]，所以[n2][a2]。", "逻辑词 `所以` 走词面守卫"),
    "S20": ("migrated", "CF20", "因为[n1][a1]，[n2][a2]。", "逻辑词 `因为` 走词面守卫"),
    "S21": ("migrated", "CF21", "[n1]要[v1]。", "逻辑词 `要` 走词面守卫；题元声明丢弃"),
    "S25": ("migrated", "CF25", "[n1][a1]！", "与官方 S03 的 `！` 无关（槽位签名不同）⇒ 独立 id"),
    # —— 废弃（不进表）——
    "S04": ("deprecated", None, "[n1][g1][a1]。",
            "含 `否` 槽 ⊄ POS_TYPES={名,动,形}；与官方 S04 `[1][2]。` **同 id 不同签名** ⇒ 不进表"),
    "S05": ("deprecated", None, "[n1][g1][a1]吗？", "含 `否` 槽 ⊄ POS_TYPES"),
    "S10": ("deprecated", None, "[n1][g1][a1]，[n2][a2]。", "含 `否` 槽 ⊄ POS_TYPES"),
    "S15": ("deprecated", None, "[n1][g1][v1]。", "含 `否` 槽 ⊄ POS_TYPES"),
    "S16": ("deprecated", None, "[n1][g1][v1][n2]。", "含 `否` 槽 ⊄ POS_TYPES"),
    "S22": ("deprecated", None, "[n1][g1][v1]吗？", "含 `否` 槽 ⊄ POS_TYPES"),
    "S26": ("deprecated", None, "[n1][g1][a1]！", "含 `否` 槽 ⊄ POS_TYPES"),
    "S23": ("deprecated", None, "[n1][d1][a1]。", "含 `数` 槽 ⊄ POS_TYPES（现有卡也无数字切片）"),
    "S24": ("deprecated", None, "[n1][d1][a1]吗？", "含 `数` 槽 ⊄ POS_TYPES"),
}

assert len(MIGRATION) == 25, len(MIGRATION)

COUNTS: dict[str, int] = {
    k: sum(1 for v in MIGRATION.values() if v[0] == k)
    for k in ("merged", "migrated", "deprecated")
}


def old_to_new(old: str) -> str | None:
    """card_flow 旧 id → 唯一家里的 id；废弃 ⇒ None。"""
    return MIGRATION[old][1]


def sign(sid: str, table: dict[str, R.Skeleton] | None = None) -> tuple | None:
    """签名 = `(pattern, ((pos, theta), ...))`；id 不在表里 ⇒ None。"""
    table = R.SKELETONS if table is None else table
    sk = table.get(sid)
    if sk is None:
        return None
    return (sk.pattern, tuple((s.pos, s.theta) for s in sk.slots))


def convert(old_sk, *, keep_theta: bool = False) -> tuple[str, tuple[tuple[str, str | None], ...]]:
    """card_flow 骨架 → 官方形状（`[n1]→[1]` 顺序重编号）。

    - `keep_theta=False`（**migrated 族**，PREREG §4）：题元一律丢弃 ⇒ `direction_safe`
      （实测 A 集 48/48 个候选 `theme=None`，真卡不产题元；保留题元会让这些记录全被
      「无角色标签 ⇒ fail-closed」拒掉）；
    - `keep_theta=True`（**merged 族**）：保留 card_flow 声明的题元 —— 它与官方条目逐字相同
      （S13/S14 的 施事·受事、S12 的 施事），这正是"签名完全相同 ⇒ 并入"的核对方式。
    """
    names = CF_SLOT_RE.findall(old_sk.template)
    pat = old_sk.template
    for k, n in enumerate(names, 1):
        pat = pat.replace(f"[{n}]", f"[{k}]")
    slots = tuple(
        (_TYPE_OF_PREFIX[n[0]], old_sk.slots[i].theme if keep_theta else None)
        for i, n in enumerate(names)
    )
    return pat, slots


def v1_report() -> dict:
    """V1 的四个实测数字（全在唯一家上算，不碰 card_flow 的文件）。"""
    import skeletons as cfs  # card_flow 自有表（只读 import）

    old = {s.id: s for s in cfs.SKELETONS}
    home = R.SKELETONS
    official = set(R.SKELETON_OFFICIAL)
    cf = set(R.SKELETON_CF)

    # ① 家内：id 两两不同 ⇒ 「同 id 不同签名」在唯一家里为 0
    ids = [sk.sid for sk in R._SKELETON_LIST]
    home_dup_ids = sorted({i for i in ids if ids.count(i) > 1})
    sig_by_id: dict[str, set] = {}
    for sid in ids:
        sig_by_id.setdefault(sid, set()).add(sign(sid))
    home_conflicts = sorted(s for s, v in sig_by_id.items() if len(v) > 1)

    # ② 命名空间：两族 id 相交
    overlap = sorted(official & cf)

    # ③ 迁移后签名逐字一致（对 migrated / merged 逐条核）
    sig_mismatch = []
    for oid, (kind, new_id, _pat, _r) in sorted(MIGRATION.items()):
        if new_id is None:
            continue
        keep = MIGRATION[oid][0] == "merged"
        want = convert(old[oid], keep_theta=keep)
        got = sign(new_id)
        if got is None or (got[0], tuple(got[1])) != want:
            sig_mismatch.append({"old": oid, "new": new_id,
                                 "want": want, "got": got})

    # ④ 反事实：若不改名，旧 id 直接落在家里会撞多少条（签名是否相同）
    clash = []
    for oid, sk in sorted(old.items()):
        h = sign(oid)
        if h is None:
            continue
        if (h[0], tuple(h[1])) != convert(sk):
            clash.append({"old": oid, "official_sign_same": False,
                          "official_pattern": h[0]})
        else:
            clash.append({"old": oid, "official_sign_same": True,
                          "official_pattern": h[0]})

    # ⑤ 处置完备性：25 条逐条有归宿
    unresolved = [o for o, v in MIGRATION.items() if v[0] not in
                  ("merged", "migrated", "deprecated")]
    missing_in_home = [o for o, (_k, n, _p, _r) in MIGRATION.items()
                       if n is not None and n not in home]
    home_sigs = {sign(s) for s in home}
    deprecated_sneaked = [o for o, (k, _n, _p, _r) in MIGRATION.items()
                          if k == "deprecated" and convert(old[o]) in home_sigs]
    out_of_domain = sorted({s.pos for sk in home.values() for s in sk.slots}
                           - set(R.POS_TYPES))

    return {
        "n_home": len(home),
        "n_official": len(official),
        "n_cf": len(cf),
        "home_dup_ids": home_dup_ids,
        "home_same_id_diff_sign": home_conflicts,
        "namespace_overlap": overlap,
        "migrated_sig_mismatch": sig_mismatch,
        "migration_counts": COUNTS,
        "unresolved": unresolved,
        "missing_in_home": missing_in_home,
        "deprecated_sneaked_in_home": deprecated_sneaked,
        "home_pos_out_of_domain": out_of_domain,
        "naive_clash_ids": [c["old"] for c in clash if not c["official_sign_same"]],
        "naive_clash_same": [c["old"] for c in clash if c["official_sign_same"]],
        "naive_clash_detail": clash,
    }


def record_disposition(records: list[dict]) -> list[dict]:
    """36 条既有成功记录逐条的 `旧 → 新`（或废弃 + 原因）。"""
    out = []
    for i, r in enumerate(records):
        old_id = r["skeleton"]
        kind, new_id, _pat, reason = MIGRATION[old_id]
        out.append({
            "rid": f"A[{i}]", "old": old_id, "kind": kind,
            "new": new_id, "reason": reason if new_id is None else "",
            "same_id_diff_sign_naive": old_id in naive_clash_ids(),
        })
    return out


def naive_clash_ids() -> list[str]:
    return [c["old"] for c in v1_report()["naive_clash_detail"]
            if not c["official_sign_same"]]
