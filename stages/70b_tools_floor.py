#!/usr/bin/env python3
"""S70b · L3-r0b 真对话路（agent_tools）免费规则地板（CPU、确定性、不训练、只读）。

数据源：nanoSeek/data/chinese/_discarded/agent_tools_dialogue.txt（项目自有合成，MIT）。
块格式（每块恒 4 回合，块间空行）：

    用户：<请求>                         ← 16 条唯一请求
    模型：我…… + ```json {name, arguments} ```   ← 工具调用（工具名只有 3 类）
    用户：[工具返回结果] + ```…```        ← 工具返回
    模型：<最终答复>                      ← 46 条唯一答复

30,000 块。本脚本**只算地板、不训练、不落盘任何数据**，回答 L3_PLAN.md §7 的 P0-b：
该路的免费规则地板会不会把可判信号吞掉。

五条免费规则（P0-b 原文），每条按它**能给出的口径**报 EM（口径不同就分开写，别混）：
  ① 最终答复整体抄工具返回        整段逐字相等；另报「工具返回整段被最终答复包含」的诊断率
  ② 抄用户话                      整段逐字相等；另加 ②b train 全集众数（整段）
  ③ 工具名众数                    两个口径：(i) 工具名预测准确率（槽位级）；
                                  (ii) 工具名条件众数答复（整段 EM）
  ④ 城市槽位从用户话复制          两个口径：(i) 槽位级准确率（与工具调用 JSON 的 city 比）；
                                  (ii) 城市条件众数答复（整段 EM）
  ⑤ 长度桶众数（桶=100 字）        整段 EM
附加两条（判"能不能训"的决定性证据，都是免费且只用可见输入）：
  ⑥ train 请求条件众数答复        整段 EM —— 即「按 16 条题面背 46 条回复」
  ⑦ 配对记忆上界                  test 里 (请求, 最终答复) 对在 train 出现过的比例
  ③c 关键词规则（Python/检索/天气）工具名预测准确率

判决（L3_PLAN.md §7 写死）：max_naive <= 30% ⇒ 该路可判；> 30% ⇒ 该路不训。

用法：python3 stages/70b_tools_floor.py [--seed 20261010] [--train-frac 0.9]
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter, defaultdict

NS_ROOT = "/home/vesita/coding/my/nanoSeek"
SRC = NS_ROOT + "/data/chinese/_discarded/agent_tools_dialogue.txt"

SEED = 20261010
TRAIN_FRAC = 0.9
LEN_BUCKET = 100          # 最终答复长度桶宽（字符）
THRESHOLD = 30.0          # P0-b 门槛（%）

FENCE_RE = re.compile(r"```[a-zA-Z]*\n(.*?)\n```", re.S)
NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
CITY_RE = re.compile(r'"city"\s*:\s*"([^"]+)"')


# ---------------- 解析 ----------------
def parse_blocks(path: str):
    """返回 (rows, n_blocks, n_bad)。rows 每项 = prefix/tc/ret/final。"""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    blocks = [b for b in text.split("\n\n") if b.strip()]
    rows, bad = [], 0
    for b in blocks:
        parts = re.split(r"(?m)^(用户|模型)：", b)
        roles, texts = parts[1::2], parts[2::2]
        if len(roles) != 4 or roles != ["用户", "模型", "用户", "模型"] or len(texts) != 4:
            bad += 1
            continue
        rows.append(dict(prefix=texts[0].strip(), tc=texts[1].strip(),
                         ret=texts[2].strip(), final=texts[3].strip()))
    return rows, len(blocks), bad


def tool_payload(ret: str) -> str:
    """工具返回里的裸载荷（优先取代码围栏内容；没有围栏就去掉 '[工具返回结果]' 前缀）。"""
    m = FENCE_RE.search(ret)
    if m:
        return m.group(1).strip()
    return ret.replace("[工具返回结果]", "").strip()


def enrich(rows):
    for r in rows:
        r["payload"] = tool_payload(r["ret"])
        m = NAME_RE.search(r["tc"])
        r["tname"] = m.group(1) if m else None
        m = CITY_RE.search(r["tc"])
        r["city"] = m.group(1) if m else None
        r["lenb"] = len(r["final"]) // LEN_BUCKET
    return rows


def em_exact(pred, rs):
    """整段逐字相等的 EM（%）。pred 是函数 r -> 预测文本。"""
    hit = sum(1 for r in rs if pred(r) == r["final"])
    return 100.0 * hit / len(rs), hit


def main(argv=None):
    ap = argparse.ArgumentParser(description="S70b agent_tools 免费规则地板（P0-b）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    a = ap.parse_args(argv)

    rows, n_blocks, n_bad = parse_blocks(SRC)
    assert n_bad == 0, f"有 {n_bad} 个块不是标准 4 回合"
    enrich(rows)
    print(f"[SRC] {SRC} 块={n_blocks} 解析出={len(rows)} 格式坏块={n_bad} | "
          f"唯一请求={len({r['prefix'] for r in rows})} 唯一最终答复={len({r['final'] for r in rows})} "
          f"唯一(请求,答复)对={len({(r['prefix'], r['final']) for r in rows})} | "
          f"工具名分布={dict(Counter(r['tname'] for r in rows).most_common())}", flush=True)

    rng = random.Random(a.seed)
    rng.shuffle(rows)
    n_tr = int(len(rows) * a.train_frac)
    train, test = rows[:n_tr], rows[n_tr:]
    print(f"[SPLIT] 逐块 shuffle(seed={a.seed}) 后 90/10：train={len(train)} test={len(test)}；"
          f"★请求集不互斥（全表只有 {len({r['prefix'] for r in rows})} 条题面）"
          f"⇒ test 的每条题面在 train 里都出现过，泄漏是构造性的、无法用切分消除。", flush=True)

    n = len(test)
    # 城市词表由 train 侧 JSON 参数导出（对 test 而言仍是免费可见的闭合词表）
    cities = sorted({r["city"] for r in train if r["city"]})
    print(f"[VOCAB] 城市闭合词表(train 导出)={cities}", flush=True)

    # ---- ① 最终答复整体抄工具返回 ----
    em1, h1 = em_exact(lambda r: r["payload"], test)
    contain = 100.0 * sum(1 for r in test if r["payload"] and r["payload"] in r["final"]) / n
    print(f"[D70B] ① 最终答复整体抄工具返回(整段逐字) EM={em1:.2f}% ({h1}/{n}) | "
          f"诊断：工具返回整段被最终答复包含的比例={contain:.2f}%（不是整段抄，只作参考）", flush=True)

    # ---- ② 抄用户话 + ②b train 全集众数 ----
    em2, h2 = em_exact(lambda r: r["prefix"], test)
    maj_final = Counter(r["final"] for r in train).most_common(1)[0][0]
    em2b, h2b = em_exact(lambda r: maj_final, test)
    print(f"[D70B] ② 抄用户话(整段逐字) EM={em2:.2f}% ({h2}/{n}) | "
          f"②b train 全集众数(整段) EM={em2b:.2f}% ({h2b}/{n})", flush=True)

    # ---- ③ 工具名众数 ----
    maj_tool = Counter(r["tname"] for r in train).most_common(1)[0][0]

    def kw_tool(r):
        """免费关键词规则（顺序有讲究：'请检索…Python…' 既是检索题又含 Python）。
        实测按此顺序 16 条题面全部命中正确工具名。"""
        if "检索" in r["prefix"]:
            return "web_search"
        if "Python" in r["prefix"]:
            return "python_interpreter"
        if "天气" in r["prefix"]:
            return "get_weather"
        return maj_tool

    hit_tool = sum(1 for r in test if r["tname"] == maj_tool)
    hit_kw = sum(1 for r in test if r["tname"] == kw_tool(r))
    per_tool = defaultdict(Counter)
    for r in train:
        per_tool[r["tname"]][r["final"]] += 1
    tool_rep = {k: v.most_common(1)[0][0] for k, v in per_tool.items()}
    em3, h3 = em_exact(lambda r: tool_rep.get(r["tname"], maj_final), test)
    print(f"[D70B] ③ 工具名众数({maj_tool})：工具名预测准确率={100.0 * hit_tool / n:.2f}% ({hit_tool}/{n}) | "
          f"③c 关键词规则(请用Python/请检索/天气)准确率={100.0 * hit_kw / n:.2f}% ({hit_kw}/{n}) | "
          f"③ii 工具名条件众数答复(整段) EM={em3:.2f}% ({h3}/{n})", flush=True)

    def city_copy(r):
        for c in cities:
            if c in r["prefix"]:
                return c
        return None

    wc = [r for r in test if r["city"]]
    copy_hit = sum(1 for r in wc if city_copy(r) == r["city"])
    per_city = defaultdict(Counter)
    for r in train:
        if r["city"]:
            per_city[r["city"]][r["final"]] += 1
    city_rep = {k: v.most_common(1)[0][0] for k, v in per_city.items()}
    em4, h4 = em_exact(lambda r: city_rep.get(r["city"], maj_final), test)
    print(f"[D70B] ④ 城市槽位从用户话复制：槽位准确率={100.0 * copy_hit / len(wc):.2f}% "
          f"({copy_hit}/{len(wc)} 条带 city 的 test 样本) | "
          f"④ii 城市条件众数答复(整段) EM={em4:.2f}% ({h4}/{n})", flush=True)

    # ---- ⑤ 长度桶众数 ----
    per_len = defaultdict(Counter)
    for r in train:
        per_len[r["lenb"]][r["final"]] += 1
    len_rep = {k: v.most_common(1)[0][0] for k, v in per_len.items()}
    em5, h5 = em_exact(lambda r: len_rep.get(r["lenb"], maj_final), test)
    print(f"[D70B] ⑤ 长度桶众数(桶={LEN_BUCKET}字, train 桶数={len(per_len)}) (整段) "
          f"EM={em5:.2f}% ({h5}/{n})", flush=True)

    # ---- ⑥ 请求条件众数（背回复）+ ⑦ 配对记忆上界 ----
    per_prefix = defaultdict(Counter)
    for r in train:
        per_prefix[r["prefix"]][r["final"]] += 1
    prefix_rep = {k: v.most_common(1)[0][0] for k, v in per_prefix.items()}
    em6, h6 = em_exact(lambda r: prefix_rep.get(r["prefix"], maj_final), test)
    seen_pairs = {(r["prefix"], r["final"]) for r in train}
    ub = 100.0 * sum(1 for r in test if (r["prefix"], r["final"]) in seen_pairs) / n
    print(f"[D70B] ⑥ train 请求条件众数答复(整段, 免费且只用可见输入) EM={em6:.2f}% ({h6}/{n}) | "
          f"⑦ (请求,答复) 配对记忆上界={ub:.2f}%", flush=True)

    # ---- max_naive 与判决 ----
    five_slot = {"①抄工具返回(整段)": em1, "②抄用户话(整段)": em2, "②b全集众数(整段)": em2b,
                 "③工具名预测(槽位)": 100.0 * hit_tool / n, "③ii工具名条件众数(整段)": em3,
                 "④城市复制(槽位)": 100.0 * copy_hit / len(wc), "④ii城市条件众数(整段)": em4,
                 "⑤长度桶众数(整段)": em5}
    five_seg = {k: v for k, v in five_slot.items() if "整段" in k or "全集众数" in k}
    mx_all_name = max(five_slot, key=five_slot.get)
    mx_seg_name = max(five_seg, key=five_seg.get)
    mx_all, mx_seg = five_slot[mx_all_name], five_seg[mx_seg_name]
    se2_seg = 2.0 * (mx_seg / 100.0 * (1 - mx_seg / 100.0) / n) ** 0.5 * 100.0
    em_extra = max(em6, ub)
    print(f"[D70B] max_naive(五条按各自口径, 含槽位级)={mx_all:.2f}% 来自={mx_all_name}", flush=True)
    print(f"[D70B] max_naive(五条全折算成整段 EM)={mx_seg:.2f}% 来自={mx_seg_name} | "
          f"2SE={se2_seg:.2f}pp ⇒ 门槛={mx_seg + se2_seg:.2f}%", flush=True)
    print(f"[D70B] max_naive(整段 EM, 含免费捷径⑥/⑦)={max(mx_seg, em_extra):.2f}% 来自="
          f"{'⑥请求条件众数' if em6 >= ub and em6 > mx_seg else ('⑦配对记忆上界' if ub > mx_seg else mx_seg_name)}",
          flush=True)
    verdict = ("不训（免费地板吞掉信号）" if (mx_all > THRESHOLD or max(mx_seg, em_extra) > THRESHOLD)
               else "可判（该路可训）")
    print(f"[D70B-VERDICT] 门槛 max_naive<={THRESHOLD:.0f}% ⇒ 判决={verdict}；"
          f"理由：①/② 免费规则给 0%，但 ④ 城市槽位从用户话复制 = "
          f"{100.0 * copy_hit / len(wc):.2f}%（槽位级），且 ⑥「按题面背回复」= {em6:.2f}%、"
          f"⑦ 配对记忆上界 = {ub:.2f}%（整段 EM，两项都只用可见输入）——"
          f"本数据集只有 {len({r['prefix'] for r in rows})} 条唯一题面 × "
          f"{len({r['final'] for r in rows})} 条唯一回复，是查找表不是对话任务。", flush=True)
    print("[DONE] exit=0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
