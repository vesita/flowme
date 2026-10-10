"""harness.chat —— REPL：读自然语言 → 逐轮生成回复（历史累积）。

★ 这个 REPL 的**诚实形状**（README 与启动横幅同样写着）：

  1. 卡组本身是一个「在候选层里选一个 + 把选中项的字母写进句面」的推理器，
     训练数据里它**只生成最后那句回复**（历史里的 `<resp>…<eos>` 是数据给的、不是它生成的）；
  2. 所以 REPL 的**摄取层（ingestion）**负责把自然语言变成「同分布上下文」：
     识别 (族, 实体, 条件) → 用**数据生成器自己的**模板 / 候选构造器渲染 ctx
     （`stages/72_nl_dialog_gen.py` / `stages/74_nl_dialog_cand_gen.py`，零拷贝、不另写一份）；
  3. 只有**两个槽位都齐了、且本轮是问句**才轮到卡组发言；之前几轮是 harness 的模板应答
     （行尾标 `(模板)`）⇒ 多轮历史真的在累积：实体来自第 1 轮、条件来自第 2 轮、问题来自第 3 轮。

`/cards` 显示当前卡集合、各自角色、拓扑与接口契约；`/reset` 清空历史；`/quit` 退出。
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import random
import sys

from harness.pipeline import (CAND_LETTERS, MAXLEN, STAGES, Pipeline, greedy_trace,
                              label_marker, prep_record)

# ---------------------------------------------------------------------------
# 数据文法（= 合成分布的定义）——直接复用生成器，不另写一份（避免漂移）
# ---------------------------------------------------------------------------
_D72 = None
_D74 = None


def _load_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def d72():
    global _D72
    if _D72 is None:
        _D72 = _load_path("h_d72", os.path.join(STAGES, "72_nl_dialog_gen.py"))
    return _D72


def d74():
    global _D74
    if _D74 is None:
        _D74 = _load_path("h_d74", os.path.join(STAGES, "74_nl_dialog_cand_gen.py"))
    return _D74


# ---------------------------------------------------------------------------
# 示例输入（启动横幅打印；都是**分布内**的）
# ---------------------------------------------------------------------------
EXAMPLES = (
    ("售后（工单号 + 处理方式）",
     ["你好，我的售后工单号是AS975482。", "我希望按返厂检测的方式处理。", "帮我催一下吧。"]),
    ("预约（姓名 + 时间）",
     ["我叫张伟，想预约牙科门诊。", "时间上我希望安排在周四下午。", "那具体几点过去合适？"]),
    ("快递（运单号 + 时间）",
     ["麻烦查一下，运单SF20485716现在是什么状态。", "我希望它在下周一送到。", "帮我看看到底还要多久。"]),
)


class Turn:
    def __init__(self, user: str, reply: str, source: str, ood: list, ref: str | None = None,
                 detail: dict | None = None):
        self.user, self.reply, self.source, self.ood = user, reply, source, ood
        self.ref, self.detail = ref, detail

    def line(self) -> str:
        src = {"model": "(卡组)", "template": "(模板)", "harness": "(harness)"}[self.source]
        return f"flowme> {self.reply} {src}"


class Session:
    """一次对话（历史累积）。`ask(text)` 是唯一入口。"""

    def __init__(self, pipe: Pipeline, k: int = 8, seed: int = 1234, max_new: int = 96):
        self.pipe = pipe
        self.k = k
        self.seed = seed
        self.max_new = max_new
        self.turns: list = []
        self.family: dict | None = None
        self.ent: str | None = None
        self.cond: str | None = None
        self.i1: int | None = None
        self.i2: int | None = None
        self.ood: list = []
        self.history: list = []

    # ------------------------------------------------------------------ 摄取
    def _match_ent(self, text: str):
        """一轮输入里找 (族, 实体, 轮1模板 index, 匹配方式)；模板匹配优先，再退到词表内子串。"""
        for fam in d72().FAMILIES:
            m1 = d72()._match_slot(text, fam["t1_user"], set(fam["ents"]), "ent")
            if m1:
                i1, ent = m1[0]
                return fam, ent, i1, "template"
        for fam in d72().FAMILIES:
            ents = [e for e in fam["ents"] if e in text]
            if len(ents) == 1:
                return fam, ents[0], 0, "vocab"
        return None, None, None, None

    def _match_cond(self, text: str, fam: dict):
        m2 = d72()._match_slot(text, fam["t2_user"], set(fam["conds"]), "cond")
        if m2:
            return m2[0][1], m2[0][0], "template"
        conds = [c for c in fam["conds"] if c in text]
        if len(conds) == 1:
            return conds[0], 0, "vocab"
        return None, None, None

    def _cond_any_family(self, text: str):
        for fam in d72().FAMILIES:
            c, i2, how = self._match_cond(text, fam)
            if c:
                return fam, c, i2, how
        return None, None, None, None

    def _t3_idx(self, text: str, fam: dict) -> int:
        return fam["t3_user"].index(text) if text in fam["t3_user"] else 0

    def ingest(self, text: str) -> tuple:
        """把一句自然语言并进历史，更新 (族, 实体, 条件)。

        返回 (ood 提示, provided 本轮填了哪些槽)。顺序无关地填第一个空槽；
        「先条件后实体」或「一轮给全」都与不变量 D 的顺序不同 ⇒ 记 OOD（照样能答）。
        """
        ood, provided = [], []
        fam_e, ent, i1, how_e = self._match_ent(text)
        if fam_e is not None:
            if self.family is None:
                self.family = fam_e
            if self.family["key"] != fam_e["key"]:
                ood.append(f"本轮像「{fam_e['key']}」族，但会话已锁定「{self.family['key']}」族（混族 = 分布外）")
            else:
                fam = self.family
                if self.ent is None:
                    self.ent, self.i1 = ent, i1
                    provided.append("实体")
                    if how_e == "vocab":
                        ood.append(f"实体「{ent}」在词表内，但措辞不在 {len(fam['t1_user'])} 条轮1模板里"
                                   f" ⇒ 已按词表识别（分布外措辞）")
                c, i2, how_c = self._match_cond(text, fam)
                if c is not None and self.cond is None:
                    self.cond, self.i2 = c, i2
                    provided.append("条件")
                    if how_c == "vocab":
                        ood.append(f"条件「{c}」在词表内，但措辞不在 {len(fam['t2_user'])} 条轮2模板里")
        else:
            fam_c, c, i2, how_c = self._cond_any_family(text)
            if c is not None:
                if self.family is None:
                    self.family = fam_c
                    ood.append("这一轮只给了条件、还没给实体 —— 分布里是先实体后条件（顺序外）")
                if self.family["key"] == fam_c["key"] and self.cond is None:
                    self.cond, self.i2 = c, i2
                    provided.append("条件")
                if self.ent is None:
                    ood.append("还没识别到实体（分布内应先给实体，见不变量 D）")
            elif not (self.ent and self.cond):
                ood.append("这一轮没有识别到实体或条件（两个槽位还没齐 ⇒ 给不了候选层）")
        if self.family is None:
            ood.append("未在合成分布的 5 族词表里识别到实体/条件 ⇒ 无法构造候选层（分布外）")
        self.ood += ood
        return ood, provided

    # ------------------------------------------------------------------ 生成
    def build_record(self, question: str) -> tuple:
        """按数据生成器**自己的**渲染器拼出同分布 ctx；返回 (rec, ood, 参考答复)。"""
        fam = self.family
        d74m, d72m = d74(), d72()
        i1 = self.i1 or 0
        i2 = self.i2 if self.i2 is not None else 0
        i3 = self._t3_idx(question, fam)
        ood = []
        if question not in fam["t3_user"]:
            ood.append(f"问法不在 {len(fam['t3_user'])} 条轮3模板里 ⇒ 已用模板问法替代"
                       f"（回复照旧「看着像」，但它答的不是你的原问题）")
        # ★用 sha256 派生种（不用内置 hash()：它按进程随机化 ⇒ 会破坏确定性）
        key = f"{fam['key']}|{self.ent}|{self.cond}|{self.k}".encode("utf-8")
        stable = int(hashlib.sha256(key).hexdigest()[:8], 16)
        rng = random.Random(self.seed ^ stable)
        pairs, gold_idx = d74m.build_candidates(rng, fam, self.ent, self.cond, self.k)
        cands = d74m._cand_records(pairs)
        cand_line = d74m.render_cand_line(pairs)
        r1u = fam["t1_user"][i1].format(ent=self.ent)
        r1s = fam["t1_sys"][i1].format(ent=self.ent)
        r2u = fam["t2_user"][i2].format(cond=self.cond)
        r2s = fam["t2_sys"][i2].format(cond=self.cond)
        r3u = fam["t3_user"][i3]
        ctx = d74m.render_ctx(r1u, r1s, r2u, r2s, r3u, cand_line)
        # 参考答复：harness 侧按**数据文法**算出（不是数据集标签）—— DET 句面 + 正确位置字母
        label = label_marker(CAND_LETTERS[gold_idx])
        det = d72m.det_reply(fam["key"], i3, self.ent, self.cond)
        ref = det[:-1] + label + "。"
        rec = {"ctx": ctx, "cands": cands, "gold_idx": gold_idx, "n_cands": self.k,
               "family": fam["key"], "ent": self.ent, "cond": self.cond,
               "t1_user": r1u, "t1_sys": r1s, "t2_user": r2u, "t2_sys": r2s, "t3": r3u,
               "n_turns": 3, "cand_line": cand_line}
        return rec, ood, ref

    def ask(self, text: str) -> Turn:
        """历史累积 + 一次生成。"""
        self.turns.append(text)
        ood, provided = self.ingest(text)
        fam = self.family

        have_both = bool(fam is not None and self.ent and self.cond)
        if have_both and not provided:
            rec, ood2, ref = self.build_record(text)
            ood += ood2
            prep_record(rec, with_target=False)
            tr = greedy_trace(self.pipe, rec, maxlen=MAXLEN,
                              show_n=min(self.max_new, MAXLEN - len(rec["p"])))
            detail = {"family": fam["key"], "ent": self.ent, "cond": self.cond,
                      "n_cands": self.k, "ctx_len": len(rec["p"]), "n_steps": tr["n_steps"],
                      "ref": ref, "cands": rec["cands"], "t3_idx": self._t3_idx(text, fam)}
            self.history.append(("model", text, tr["text"], ood))
            return Turn(text, tr["text"], "model", ood, ref=ref, detail=detail)

        if fam is not None and self.ent and not self.cond:
            i1 = self.i1 or 0
            reply = fam["t1_sys"][i1].format(ent=self.ent)
            hint = "条件还没给 ⇒ 试：" + fam["t2_user"][0].format(cond=fam["conds"][0])
            self.history.append(("template", text, reply, ood))
            return Turn(text, reply, "template", ood,
                        detail={"family": fam["key"], "ent": self.ent, "cond": None, "hint": hint})

        if fam is not None and self.cond and not self.ent:
            reply = "请先告诉我" + fam["ent_kind"] + "（分布内例子见启动横幅）。"
            self.history.append(("harness", text, reply, ood))
            return Turn(text, reply, "harness", ood)

        if have_both and provided:
            i2 = self.i2 if self.i2 is not None else 0
            reply = fam["t2_sys"][i2].format(cond=self.cond)
            if len(provided) == 2:
                hint = "两个槽位这次一起给了（顺序外）；再说一句问法（第 3 轮）我就答。"
            elif provided == ["条件"]:
                hint = "条件已记下；再说一句问法（第 3 轮）我就答。"
            else:
                hint = "实体/条件的先后与分布相反（顺序外）；再说一句问法（第 3 轮）我就答。"
            self.history.append(("template", text, reply, ood))
            return Turn(text, reply, "template", ood,
                        detail={"family": fam["key"], "ent": self.ent, "cond": self.cond,
                                "hint": hint})

        reply = "请先给一个我能识别的东西（分布内例子见启动横幅；/examples 再看一遍）。"
        self.history.append(("harness", text, reply, ood))
        return Turn(text, reply, "harness", ood)

    # ------------------------------------------------------------------ 命令
    def cards_text(self) -> str:
        pipe = self.pipe
        lines = [f"当前卡集合：{len(pipe.order)} 张（arch={pipe.arch}；接口 d={pipe.d}；参数={pipe.n_params()}）",
                 f"拓扑：{pipe.topology_line()}"]
        for row in pipe.interface_summary():
            j = "—" if row["judged"] is None else row["judged"]["metric"]
            side = "（侧支）" if row["side_branch"] else ""
            lines.append(f"  · {row['instance']:<9} {row['spec']:<12}{side} "
                         f"参数={row['params']:<10} 注入={row['injection']:<4} 读={','.join(row['deps'])}")
            lines.append(f"      角色：{row['doc']}")
            lines.append(f"      可判中间目标：{j}")
        return "\n".join(lines)

    def reset(self) -> None:
        self.turns, self.history, self.ood = [], [], []
        self.family = self.ent = self.cond = self.i1 = self.i2 = None


# ---------------------------------------------------------------------------
# 横幅
# ---------------------------------------------------------------------------
def banner(pipe: Pipeline, k: int = 8, ckpt: str | None = None, ckpt_sha: str | None = None) -> str:
    fams = d72().FAMILIES
    n_tpl = sum(len(f["t1_user"]) + len(f["t1_sys"]) + len(f["t2_user"]) + len(f["t2_sys"])
                + len(f["t3_user"]) + len(f["reply"]) for f in fams)
    per_fam = n_tpl // len(fams)
    lines = [
        "=" * 92,
        "flowme harness · chat（推理入口 · 默认 CPU · 确定性贪心解码）",
        "=" * 92,
        f"卡组：{' → '.join(pipe.order)}  |  参数={pipe.n_params()}  |  接口 d={pipe.d}  |  候选层 k={k}",
        f"ckpt：{ckpt or '（未载）'}" + (f"  sha256[0:16]={ckpt_sha[:16]}" if ckpt_sha else ""),
        "",
        "★ 诚实边界（先说清做不到什么）：",
        "  1) 当前唯一的「卡」是**同构 encoder 层卡**（我们没有第二种尺寸/功能的卡）；",
        f"  2) 对话能力**限于自造合成分布**：{len(fams)} 族 × {per_fam} 条 DET 模板（共 {n_tpl} 条，logs/76_*，"
        "回复必须含选中候选的字母）⇒ **分布内流畅正确、分布外会给出「看着像但内容错」的回复**；",
        "  3) **动态加载尚未实现**：probe 里的顺序执行是它的**极限情形**（最坏延迟、最小驻留）；",
        "  4) 历史里的应答句是 harness 的模板（尾标 (模板)），只有最后的回复是卡组生成的（尾标 (卡组)）。",
        "",
        "分布内示例输入（照这样连着敲 3 行；换一个族也行）：",
    ]
    for name, ex in EXAMPLES:
        lines.append(f"  [{name}]  " + "  │  ".join(ex))
    lines += [
        "",
        "命令：/cards 看卡集合与角色 · /reset 清空历史 · /examples 示例 · /quit 退出",
        "（分布外输入会打 [OOD?]；标了它也表示「回复可能内容错」，别只看流畅度）",
        "=" * 92,
    ]
    return "\n".join(lines)


def examples_text() -> str:
    return "\n".join(f"[{n}] " + "  │  ".join(e) for n, e in EXAMPLES)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------
def repl(pipe: Pipeline, k: int = 8, seed: int = 1234, ckpt: str | None = None,
         ckpt_sha: str | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")     # 多轮中文走管道也不炸
    except Exception:
        pass
    print(banner(pipe, k=k, ckpt=ckpt, ckpt_sha=ckpt_sha))
    sess = Session(pipe, k=k, seed=seed)
    while True:
        try:
            line = input("你> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        text = line.strip()
        if not text:
            continue
        if text in ("/quit", "/exit", "/q"):
            break
        if text == "/cards":
            print(sess.cards_text())
            continue
        if text == "/reset":
            sess.reset()
            print("[harness] 历史已清空。")
            continue
        if text == "/examples":
            print(examples_text())
            continue
        t = sess.ask(text)
        for o in t.ood:
            print(f"[OOD?] {o}")
        print(t.line())
        if t.source == "model":
            d = t.detail
            print(f"       [选择] 候选层 k={d['n_cands']}，ctx={d['ctx_len']} token，"
                  f"生成 {d['n_steps']} 步；族={d['family']} 实体={d['ent']} 条件={d['cond']}"
                  f"（轮3模板 idx={d['t3_idx']}）")
            same = d["ref"].strip() == t.reply.strip()
            print(f"       [参考] {d['ref']}  ← harness 按数据文法算的答案（非数据集标签）"
                  f" ⇒ {'一致' if same else '★不一致'}")
        elif t.detail and t.detail.get("hint"):
            print(f"       [harness] {t.detail['hint']}")
    print(f"[harness] 会话结束（/reset 之后累计 {len(sess.turns)} 轮用户输入）。")
    return 0
