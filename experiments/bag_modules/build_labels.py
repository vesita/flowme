#!/usr/bin/env python3
"""bag_modules 模块标注器：给每个袋项打 {类型, 题元, 槽型} 标签 + 逐模块可靠性。

取值域 = `experiments/card_flow/lexicon.py` 的 SLOT_TYPES/THEMES（= dev-notes/16 §7.7）。
**只读句面 + bag_span**；禁止读 skel_id / skel 模板 / assign —— 模板只在本文件**事后算可靠性金标**。

来源（PREREG §2）：
  M1 类型  = ② 新构造形态规则（本文件 R1）；金标 = ① 现有手写词性闭集 POS_LEXICON（只读 import）
  M2 题元  = ② 新构造句面邻接规则；金标 = 结构金标（gold skel 模板槽位邻接字面 + 声明映射）
  M3 槽型  = ① 现有规则（struct_supervision/build_data.py::slot_class，只读 import）

用法：uv run python experiments/bag_modules/build_labels.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "two_channel_head"))

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SS_DIR = ROOT / "experiments" / "struct_supervision"
TCH_DATA = ROOT / "experiments" / "two_channel_head" / "data"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ss_build = _load(SS_DIR / "build_data.py", "ss_build_ro")      # 只读：拿 slot_class / RE_TIME
slot_class = ss_build.slot_class                                # ① 现有规则，逐字复用

# ① 现有手写词性闭集（人工判定），作 M1 的金标来源
_ds = _load(ROOT / "src" / "dtseek" / "tasks" / "builtin" / "cloze_fill" / "dataset.py",
            "cloze_ds_ro")
POS_LEXICON = _ds.POS_LEXICON

# ---------------------------------------------------------------------------
# 取值域
# ---------------------------------------------------------------------------
TYPES = ("名", "动", "形", "其他")
ROLES = ("施事", "受事", "时", "其他")
CLASSES = ("Q", "N", "M", "T", "O")
T_ID = {v: i for i, v in enumerate(TYPES)}
R_ID = {v: i for i, v in enumerate(ROLES)}
C_ID = {v: i for i, v in enumerate(CLASSES)}

# ---------------------------------------------------------------------------
# M1 类型 R1：形态标记规则（手写闭集，**与 POS_LEXICON 刻意不重合**，跑前写死）
# ---------------------------------------------------------------------------
RE_ASPECT = re.compile(r"了|着|过")                       # 体标记
#: 含 了/着/过 但**不是体标记**的多字词（先摘掉再判，跑前写死的闭集）
ASPECT_STRIP: tuple[str, ...] = (
    "通过", "超过", "经过", "过程", "过滤", "错过", "难过", "过去", "过来", "路过",
    "透过", "了不起", "了解", "明了", "为了", "意味着", "沿着", "顺着", "接着", "朝着",
)


def has_aspect(g: str) -> bool:
    t = g
    for w in ASPECT_STRIP:
        t = t.replace(w, "")
    return bool(RE_ASPECT.search(t))
RE_MODAL = re.compile(r"会|能|要|可以|应该|必须|正在|愿意|敢|需要")         # 能愿/进行（单用"在/得/想"太贪，弃）
RE_CAUS = re.compile(r"把|被|让|使")                      # 使役/被动 ⇒ 该小句必有谓词（"叫/给"太贪，弃）
RE_DEG = re.compile(r"很|太|真|挺|非常|特别|更|最|比较|十分|相当|多么")        # 程度副词
#: 含 会/要/能 但**不是能愿动词**的多字词（先摘掉再判，跑前写死的闭集）
MODAL_STRIP: tuple[str, ...] = (
    "重要", "主要", "要求", "必要", "只要", "要素", "机会", "社会", "会计",
    "开会", "大会", "晚会", "年会", "能量", "能力", "会议",
)

VERB_LEX: tuple[str, ...] = (
    "学习", "工作", "讨论", "研究", "准备", "安排", "检查", "完成", "开始", "结束",
    "参加", "解决", "发现", "记得", "选择", "判断", "比较", "观察", "保持", "改变",
    "提高", "减少", "增加", "支持", "反对", "希望", "觉得", "认为", "打算", "决定",
    "介绍", "说明", "确认", "商量", "建议", "请求", "帮助", "通知", "报告", "等待",
    "出发", "回来", "出去", "进门", "敲门", "开门", "吃饭", "喝水", "睡觉", "休息",
    "锻炼", "逛街", "旅行", "拍照", "打扫", "修理", "种植", "收获", "讲解", "提问",
    "回答", "背诵", "朗读", "抄写", "搬运", "折叠", "挑选", "打印", "复印", "填写",
    "删除", "保存", "下载", "上传",
    "发现", "出现", "增加", "减少", "造成", "导致", "产生", "形成",
    "解决", "处理", "实现", "达到", "取得", "获得", "失去", "造成", "影响", "决定",
    "表示", "认为", "指出", "强调", "发现", "知道", "了解", "明白", "记得", "忘记",
    "喜欢", "讨厌", "害怕", "担心", "希望", "希望", "努力", "坚持", "放弃", "成功",
    "失败", "赢", "输", "比", "赛", "玩", "唱", "跳", "笑", "哭", "睡", "醒",
    "穿", "戴", "洗", "修", "建", "造", "搬", "运", "种", "收", "卖", "借", "还",
    "模仿", "崇拜", "欣赏", "喜欢", "下意识", "分心", "偷偷", "忘记", "吸引",
    "获得", "筹备", "代表", "成为", "剩下", "剩", "看起来", "看起来", "保证",
    "表达", "赚钱", "想象", "保证", "算", "算是", "传输", "造成", "影响", "关注",
    "统一", "赶到", "冥想", "运用", "交流", "存在", "可能", "超过", "受", "欢迎",
    "欢迎", "造成", "记得",
)
# 单字动词一律**不作独立证据**（太贪：在/到/是/有/看/说 会把名词性小句判成动词，
# 实测使 M1 对 POS_LEXICON 金标准确率只有 0.44）；只留体标记与使役/被动标记。
ADJ_LEX: tuple[str, ...] = (
    "重要", "安全", "简单", "复杂", "困难", "容易", "清楚", "干净", "整洁", "热闹",
    "安静", "漂亮", "丰富", "及时", "准确", "合理", "现代", "传统", "常见", "熟悉",
    "陌生", "严格", "轻松", "紧张", "忙碌", "清晰", "模糊", "详细", "具体", "温暖",
    "寒冷", "明亮", "潮湿", "干燥", "宽敞", "拥挤", "新鲜", "陈旧", "诚实", "聪明",
    "顽皮", "温柔", "粗心", "认真", "马虎", "耐心", "着急", "满意", "失望", "兴奋",
    "平静", "感动", "尴尬", "关键", "漫长", "短暂", "稀少", "密集", "柔软", "坚硬",
    "光滑", "粗糙", "香甜", "苦涩", "响亮", "低沉", "鲜艳", "暗淡", "好", "坏", "大",
    "小", "多", "少", "高", "低", "长", "短", "快", "慢", "早", "晚", "远", "近",
    "深", "浅", "宽", "窄", "厚", "薄", "轻", "重", "强", "弱", "新", "旧", "美",
    "丑", "冷", "热", "干", "湿", "亮", "暗", "甜", "苦", "酸", "辣", "香", "臭",
)


VERB2: tuple[str, ...] = tuple(v for v in dict.fromkeys(VERB_LEX) if len(v) >= 2)
ADJ2: tuple[str, ...] = tuple(a for a in dict.fromkeys(ADJ_LEX) if len(a) >= 2)
ADJ1: frozenset = frozenset(a for a in ADJ_LEX if len(a) == 1)
RE_DEG_ADJ = re.compile(r"(?:很|太|真|挺|非常|特别|更|最|比较|十分|相当|多么)([" +
                        "".join(sorted(ADJ1)) + r"])")


def _modal_text(g: str) -> str:
    t = g
    for w in MODAL_STRIP:
        t = t.replace(w, "")
    return t


def type_r1(g: str) -> str:
    """M1 主标注器（输入=袋项文本）。动 > 形 > 名，无证据 ⇒ 名。

    单字动词/单字形容词**不作独立证据**（太贪：在/到/是/有/看/大/好 把名词性小句判成动/形）；
    了/着/过 与 会/要/能 先摘掉非标记多字词再判（见 ASPECT_STRIP / MODAL_STRIP）。
    """
    t = _modal_text(g)
    if has_aspect(t) or RE_CAUS.search(t):
        return "动"
    if any(v in t for v in VERB2):
        return "动"
    if RE_MODAL.search(t):
        return "动"
    if any(a in g for a in ADJ2) or RE_DEG_ADJ.search(g):
        return "形"
    return "名"


def type_r2(g: str) -> str | None:
    """M1 独立对照（① 现有词性闭集查表，多数表决；查不到 = None fail-closed）。"""
    hits = Counter()
    for w, cls in (("名词", "名"), ("动词", "动"), ("形容词", "形")):
        for word in POS_LEXICON.get(w, ()):          # 副词类不进本实验取值域
            if word in g:
                hits[cls] += 1
    if not hits:
        return None
    top = hits.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None                                    # 平票 ⇒ 不判（fail-closed）
    return top[0][0]


# ---------------------------------------------------------------------------
# M2 题元 R1：句面邻接规则（span 前后各看 6 字）
# ---------------------------------------------------------------------------
RE_ROLE_ARG_POST = re.compile(r"^(被)")                 # X 被 … ⇒ X=受事
RE_ROLE_ARG_PRE = re.compile(r"(被)$")                 # … 被 Y ⇒ Y=施事
RE_ROLE_BA_POST = re.compile(r"^(把)")                 # X 把 … ⇒ X=施事
RE_ROLE_BA_PRE = re.compile(r"(把)$")                  # … 把 Y ⇒ Y=受事
RE_ROLE_CAUS_POST = re.compile(r"^(让|叫|给|使)")        # X 让 … ⇒ X=施事（使役）
RE_ROLE_CAUS_PRE = re.compile(r"(让|叫|给)$")           # … 让 Y ⇒ Y=受事（声明映射）
RE_ROLE_TIME_POST = re.compile(r"^(之前|之后|以前|以后|的时候|之时|的话|之前|前|后)")
RE_ROLE_TIME_PRE = re.compile(r"(在|从|自从|每当|到|一)$")
RE_TIME_IN = ss_build.RE_TIME                           # ① 现有时间词规则（与槽型 T 同源）


def role_r1(g: str, pre: str, post: str) -> str:
    """M2 主标注器（输入=袋项文本 + 句面左右邻）。论元标记 > 时间。"""
    pre6, post6 = pre[-6:], post[:6]
    if RE_ROLE_ARG_POST.search(post6):
        return "受事"
    if RE_ROLE_ARG_PRE.search(pre6):
        return "施事"
    if RE_ROLE_BA_POST.search(post6):
        return "施事"
    if RE_ROLE_BA_PRE.search(pre6):
        return "受事"
    if RE_ROLE_CAUS_POST.search(post6):
        return "施事"
    if RE_ROLE_CAUS_PRE.search(pre6):
        return "受事"
    if RE_ROLE_TIME_POST.search(post6) or RE_ROLE_TIME_PRE.search(pre6):
        return "时"
    if RE_TIME_IN.search(g):
        return "时"
    return "其他"


# ---------------------------------------------------------------------------
# 结构金标（gold skel 模板 → 槽位题元）：与句面规则两条代码路径
# ---------------------------------------------------------------------------
RE_SLOT = re.compile(r"\[(\d+)\]")


def template_slot_literals(skel: str) -> dict[int, tuple[str, str]]:
    """槽位号 → (该槽左侧字面, 右侧字面)。"""
    parts = RE_SLOT.split(skel)          # ["", "1", "的时候，", "2", ""]（split 含捕获组）
    lits = parts[::2]                    # 字面段
    sids = [int(x) for x in parts[1::2]]  # 槽号段
    assert len(lits) == len(sids) + 1, (skel, parts)
    return {sid: (lits[i], lits[i + 1]) for i, sid in enumerate(sids)}


def role_gold_slot(skel: str) -> dict[int, str]:
    """声明的「槽位邻接字面 → 题元」映射（PREREG §3；跑前写死）。"""
    lit = template_slot_literals(skel)
    gold: dict[int, str] = {}
    for sid, (pre, post) in lit.items():
        r = "其他"
        if post.startswith("被"):
            r = "受事"
        elif pre.endswith("被"):
            r = "施事"
        elif post.startswith("把"):
            r = "施事"
        elif pre.endswith("把"):
            r = "受事"
        elif post.startswith(("让", "叫", "给", "使")):
            r = "施事"
        elif pre.endswith(("让", "叫", "给")):
            r = "受事"
        elif post.startswith(("之前", "之后", "以前", "以后", "的时候", "之时", "的话")):
            r = "时"
        elif pre.endswith(("在", "从", "自从", "每当")):
            r = "时"
        elif pre.endswith("到") and post.startswith("到"):
            r = "时"
        gold[sid] = r
    return gold


# ---------------------------------------------------------------------------
# 标注一行（按 bag 下标）
# ---------------------------------------------------------------------------
def label_row(row: dict) -> list[dict]:
    s = row["sent"]
    out = []
    for k, (a, b) in enumerate(row["bag_span"]):
        g = s[a:b]
        assert g == row["bag"][k], "span 与 bag 不一致"
        out.append({"t": T_ID[type_r1(g)],
                    "r": R_ID[role_r1(g, s[:a], s[b:])],
                    "c": C_ID[slot_class(g)]})
    return out


def row_key(row: dict) -> str:
    return hashlib.md5((row["sent"] + str(row["bag_span"])).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 可靠性
# ---------------------------------------------------------------------------
def reliability(split: str, rows: list[dict], labels: list[list[dict]]) -> dict:
    n_items = sum(len(l) for l in labels)
    rep: dict = {"split": split, "n_rows": len(rows), "n_items": n_items}

    # ---- M1 类型 ----
    r1 = [TYPES[l["t"]] for lab in labels for l in lab]
    r2 = []
    for row, lab in zip(rows, labels):
        for g in row["bag"]:
            r2.append(type_r2(g))
    cov2 = [i for i, x in enumerate(r2) if x is not None]
    agree = sum(1 for i in cov2 if r2[i] == r1[i])
    gold = r2
    per_cls = {}
    for c in ("名", "动", "形"):
        idx = [i for i in cov2 if gold[i] == c]
        per_cls[c] = {"n_gold": len(idx),
                      "acc": round(sum(1 for i in idx if r1[i] == c) / len(idx), 4)
                      if idx else None}
    rep["M1_type"] = {
        "dist": dict(Counter(r1)),
        "cover_r1": round(sum(1 for x in r1 if x != "其他") / len(r1), 4),
        "lexicon_gold_coverage": round(len(cov2) / max(1, len(r2)), 4),
        "acc_vs_lexicon_gold": round(agree / len(cov2), 4) if cov2 else None,
        "n_lexicon_gold": len(cov2),
        "per_class": per_cls,
        "r2_dist_on_covered": dict(Counter(r2[i] for i in cov2)),
    }

    # ---- M2 题元 ----
    rr = [ROLES[l["r"]] for lab in labels for l in lab]
    gslot_by_row = [role_gold_slot(r["skel"]) for r in rows]
    gold_r, pred_r = [], []
    det = 0
    for row, lab, gs in zip(rows, labels, gslot_by_row):
        inv = {b: s for s, b in enumerate(row["assign"])}   # bag 下标 → 槽位号
        for b, l in enumerate(lab):
            g = gs[inv[b] + 1]
            if g == "其他":
                continue
            det += 1
            gold_r.append(g)
            pred_r.append(ROLES[l["r"]])
    per_role = {}
    for c in ("施事", "受事", "时"):
        idx = [i for i, g in enumerate(gold_r) if g == c]
        per_role[c] = {"n_gold": len(idx),
                       "acc": round(sum(1 for i in idx if pred_r[i] == c) / len(idx), 4)
                       if idx else None}
    rep["M2_role"] = {
        "dist": dict(Counter(rr)),
        "cover_r1_non_other": round(sum(1 for x in rr if x != "其他") / len(rr), 4),
        "struct_gold_determinate": det,
        "struct_gold_coverage": round(det / max(1, n_items), 4),
        "acc_vs_struct_gold": round(sum(1 for a, b in zip(pred_r, gold_r) if a == b)
                                    / len(gold_r), 4) if gold_r else None,
        "n_struct_gold": len(gold_r),
        "per_class": per_role,
        "gold_dist": dict(Counter(gold_r)),
    }

    # ---- M3 槽型 ----
    cc = [CLASSES[l["c"]] for lab in labels for l in lab]
    same = 0
    tot = 0
    for row, lab in zip(rows, labels):
        for g, l in zip(row["bag"], lab):
            tot += 1
            same += int(CLASSES[l["c"]] == slot_class(g))
    rep["M3_cls"] = {
        "dist": dict(Counter(cc)),
        "cover_non_O": round(sum(1 for x in cc if x != "O") / len(cc), 4),
        "agreement_with_adv2_rule": round(same / tot, 6), "n_checked": tot,
    }

    # ---- 模块间冗余 ----
    pair = Counter((ROLES[l["r"]], CLASSES[l["c"]]) for lab in labels for l in lab)
    rep["redundancy"] = {
        "P(cls=T | role=时)": round(pair[("时", "T")] / max(1, sum(v for (r0, c0), v in pair.items() if r0 == "时")), 4),
        "P(role=时 | cls=T)": round(pair[("时", "T")] / max(1, sum(v for (r0, c0), v in pair.items() if c0 == "T")), 4),
        "both_non_other": round(sum(v for (r0, c0), v in pair.items()
                                    if r0 != "其他" and c0 != "O") / max(1, n_items), 4),
    }
    return rep


SPLITS = ("train", "test", "adv1", "adv2")


def load_rows(split: str) -> list[dict]:
    p = TCH_DATA / f"{split}.jsonl" if split in ("train", "test") \
        else SS_DIR / "data" / f"{split}.jsonl"
    if not p.exists():
        raise SystemExit(f"缺数据：{p}")
    with open(p, encoding="utf-8") as fp:
        return [json.loads(x) for x in fp]


def main() -> None:
    DATA.mkdir(exist_ok=True)
    all_rep = {"prereg": "PREREG.md §2/§3",
               "domains": {"type": TYPES, "role": ROLES, "cls": CLASSES},
               "sources": {
                   "M1_type": "② 新构造形态规则（R1）；金标=① POS_LEXICON（cloze_fill/dataset.py，人工判定）",
                   "M2_role": "② 新构造句面邻接规则；金标=gold skel 模板槽位邻接字面+声明映射",
                   "M3_cls": "① 现有规则 struct_supervision/build_data.py::slot_class（只读复用）"}}
    for split in SPLITS:
        rows = load_rows(split)
        labels = [label_row(r) for r in rows]
        # 落盘 + 行对齐指纹
        out = {"fp": [row_key(r) for r in rows],
               "skel_id": [r["skel_id"] for r in rows],
               "labels": labels}
        (DATA / f"labels_{split}.json").write_text(
            json.dumps(out, ensure_ascii=False), encoding="utf-8")
        rep = reliability(split, rows, labels)
        all_rep[split] = rep
        print(f"[{split}] n_rows={rep['n_rows']} n_items={rep['n_items']}")
        print(f"   M1 type dist={rep['M1_type']['dist']} 词典金标覆盖={rep['M1_type']['lexicon_gold_coverage']} "
              f"准确率={rep['M1_type']['acc_vs_lexicon_gold']} (n={rep['M1_type']['n_lexicon_gold']}) "
              f"分类别={rep['M1_type']['per_class']}")
        print(f"   M2 role dist={rep['M2_role']['dist']} 非其他覆盖={rep['M2_role']['cover_r1_non_other']} "
              f"结构金标覆盖={rep['M2_role']['struct_gold_coverage']} "
              f"准确率={rep['M2_role']['acc_vs_struct_gold']} (n={rep['M2_role']['n_struct_gold']}) "
              f"分类别={rep['M2_role']['per_class']}")
        print(f"   M3 cls  dist={rep['M3_cls']['dist']} 非O覆盖={rep['M3_cls']['cover_non_O']} "
              f"与adv2规则一致={rep['M3_cls']['agreement_with_adv2_rule']}")
        print(f"   冗余 {rep['redundancy']}")
    (DATA / "reliability.json").write_text(
        json.dumps(all_rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] → {DATA}/labels_*.json + reliability.json")


if __name__ == "__main__":
    main()
