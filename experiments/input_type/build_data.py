#!/usr/bin/env python3
"""输入类型分类（E-A / P-类型）数据集构建，fail-closed。

产出（全部落 jsonl，逐条可追溯到构造规则）：
  id_train.jsonl      分布内训练集 5 类 × 400
  id_test.jsonl       分布内测试集 5 类 × 120（与训练同构造分布、文本不重合）
  adversarial.jsonl   对抗集 5 规则 × 45 = 225（表面像 A 真值 B，naive != true）
  known_answers.jsonl 已知答案 15 条（5 类 × 3，人工定稿）
  stats.json          规模 / 分布 / 去重校验结果

两条判据函数（详见 README §1）：
  labeler(t)  结构判据（真值口径，fail-closed：任何文本都能给出类型）
  naive(t)    表层判据（有下划线=填空、有 N）=选择、有说话人=多轮）——对抗集要求 naive != true
任一 assert 不满足即抛错，绝不静默降级。
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dtseek.tasks.corpus import CORPUS_GLOB, resolve_corpus_files  # noqa: E402
from dtseek.tasks.plugin import all_tasks  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
SPLIT_SEED = 20240927

LABELS = ["plain", "candidates", "cloze", "multi_turn", "unknown"]
N_TRAIN, N_TEST = 400, 120

# ---------------------------------------------------------------------------
# 判据函数
# ---------------------------------------------------------------------------
CJK_RE = re.compile(r"[一-龥]")
#: 选择族内部的空位：任何 `__`/`＿＿` 都算（卡内容语体门禁已挡住英文单词，
#: 且 119__ / 6x__ 这类“数字或字母紧跟空位”在真卡样本里合法）。
BLANK_ANY = re.compile(r"__|＿＿")
#: 无候选列表时的空位判据：`__` 紧贴 ASCII 字母 ⇒ 是标识符成分（__init__ / user__name），
#: 不算空位；紧贴数字或汉字（119__ 之外的裸空位）算。全角 `＿＿` 恒算空位。
SLOT_BLANK = re.compile(r"(?<![A-Za-z])__(?![A-Za-z])|＿＿")
CAND_MARK = re.compile(r"\|(\d)）")
ANY_NUM_SLOT = re.compile(r"\d）")
EMPTY_ITEM = re.compile(r"\|\d）(?=\||$)")
ANY_UNDERSCORE = re.compile(r"[_＿]")
SPEAKER_LINE = re.compile(r"^(用户|模型|助手|客服|医生|顾客|老师|学生|甲|乙|A|B)[：:]")
SPEAKER_ANY = re.compile(r"(用户|模型|助手|客服|医生|顾客|老师|学生|甲|乙)[：:]")


def cjk_count(s: str) -> int:
    return len(CJK_RE.findall(s))


def is_garbage(s: str) -> bool:
    t = s.strip()
    return len(t) < 4 or cjk_count(t) == 0 or "�" in t


def is_multi_top(t: str) -> bool:
    lines = [l for l in t.split("\n") if l.strip()]
    if len(lines) < 2:
        return False
    if not SPEAKER_LINE.match(lines[0]):
        return False
    return sum(1 for l in lines if SPEAKER_LINE.match(l)) >= 2


def labeler(t: str) -> str:
    """真值结构判据（fail-closed，返回闭集 5 类之一）。"""
    if is_garbage(t):
        return "unknown"
    marks = CAND_MARK.findall(t)
    head, sep, items = t.partition("|1）")
    blank_head = bool(BLANK_ANY.search(head))
    blank_items = bool(sep) and bool(BLANK_ANY.search(items))
    if marks or sep:
        # 选择族：列表必须恰好 |1）|2）|3）|4） 顺序完整、无空条目
        if marks != ["1", "2", "3", "4"] or EMPTY_ITEM.search(t):
            return "unknown"
        if blank_head and blank_items:
            return "unknown"          # 头与条目都有空位：结构错位
        if blank_items:
            return "unknown"          # 空位落在候选条目里：既非择优也非填空
        if blank_head:
            return "cloze"
        if is_multi_top(t):
            return "unknown"
        return "candidates"
    if is_multi_top(t):
        return "multi_turn"
    if SLOT_BLANK.search(t):
        return "cloze"                # 只有空位、无候选列表：仍是填空输入
    return "plain"


def naive(t: str) -> str:
    """表层判据（对抗集的「表面像」定义）：顺序 下划线 > 数字括号 > 说话人。"""
    if ANY_UNDERSCORE.search(t):
        return "cloze"
    if ANY_NUM_SLOT.search(t):
        return "candidates"
    if SPEAKER_ANY.search(t):
        return "multi_turn"
    return "plain"


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"[build_data fail-closed] {msg}")


# ---------------------------------------------------------------------------
# 模板与填充池（A1–A5 = 对抗集；F1 = 训练/分布内 unknown；F2 = 对抗集 unknown）
# ---------------------------------------------------------------------------
OPTS = ["加班", "休息", "早起", "跑步", "读书", "做饭", "散步", "养花", "爬山",
        "钓鱼", "骑车", "听戏", "种菜", "抄经", "练字"]

QS = ["明早的班车几点发", "这道菜要不要焯水", "周末去哪儿玩合适",
      "这份报告什么时候交", "他为什么没来上班"]

PHR = ["六点开门", "七点检票", "八点发车", "九点才有", "坐地铁最快",
       "走路也行", "打车太贵", "明天可以", "后天再说", "周末有空"]

XS = ["你听我说完", "这题我不会", "明天再说吧", "我不想去了", "再给我点时间"]
YS = ["我听得清楚", "慢慢来就好", "那就定下来", "别着急先歇", "行我帮你问"]

# A1：普通句里**提到**候选格式 ⇒ 真值 plain（naive 会判 candidates）
A1_TPL = [
    "他写了 1）{a} 和 2）{b} 两个选项，最后交了白卷。",
    "试卷上 1）{a}、2）{b} 两个选项都被人划掉了。",
    "她在清单上标了 1）{a} 和 2）{b}，让我照着买。",
    "文章里 1）{a} 这个说法比 2）{b} 更常见一些。",
    "他在黑板上写下 1）{a} 和 2）{b}，问全班同学选哪个。",
    "说明书只印了 1）{a}，2）{b} 两项，后面的都被撕掉了。",
    "评论区吵翻了，有人站 1）{a}，有人站 2）{b}。",
    "师傅把 1）{a} 和 2）{b} 念了一遍，说两个都不算错。",
    "备忘录里只留了 1）{a}、2）{b} 两条，其余的划没了。",
]

# A2：普通句里出现下划线但不是空位（标识符/dunder）⇒ 真值 plain（naive 会判 cloze）
A2_ITEMS = [
    "Python 里 __init__ 方法在对象创建时会被自动调用，别写重复的构造逻辑。",
    "这个字段叫 user_name，之前有人写成 username 结果对不上。",
    "数据库里 order_id 这个字段昨天被改名，害得我排查了半天。",
    "配置文件里的 __DEBUG__ 开关一旦打开就会打印全部日志。",
    "文档里说 MAX__COUNT 常量设好之后就不要再动它了。",
    "他把 user__name 和 user_name 两种写法混着用，代码很快就乱了。",
    "这篇文章讲的是 snake_case 命名法，file_name 就是最常见的例子。",
    "测试用例 test__edge_01 每次都会跑，专门覆盖边界情况。",
    "老师在黑板上写下 flag_bit，说这叫下划线命名，别念成下划线位。",
    "模板变量 __title__ 会被渲染成页面标题，改了它整站都会变。",
    "接口文档里 create_time 和 update_time 两个参数都必须传。",
    "他抱怨 __repr__ 没写好，调试的时候打印出来全是一串乱码。",
    "运维把 __version__ 提到了新版，忘了同步改下面的依赖。",
    "变量名 snake_case_convention 在这个仓库里是强制要求。",
    "那份表里 prd__id 列填错了，导出的数据全部错位。",
    "命名规则要求字段用下划线连接，比如 goods__price 这样写才合规。",
    "示例代码里的 __doc__ 被注释掉了，帮助文档因此是空的。",
    "他指着 login__check 说这个函数名写得不够直观。",
    "日志里出现 token__id 说明中间件没有正确替换占位符。",
    "她把字段从 userId 改成 user_id，说是团队统一规范。",
    "论文里用 h__index 表示指标，审稿人建议改成另一种写法。",
    "构建脚本读取 __file__ 拿到当前路径，再拼上子目录名。",
    "那段 SQL 里 table__name 拼错了，查询直接报错找不到表。",
    "配置项 env__name 决定了部署到哪套环境，改前要先申请。",
    "他给宏定义 MAX__DEPTH 设了上限，防止递归爆栈。",
    "代码评审时被指出 __slots__ 用法不对，得按文档来。",
    "这行里的 __weakref__ 是 Python 的内部属性，不该手动赋值。",
    "字段清单里 addr__detail 和 addr__code 挨在一起，容易看混。",
    "他说 __qualname__ 比 __name__ 更适合做序列化的键。",
    "表单里的 captcha__id 刷新后就失效了，需要重新取一次。",
    "脚本按 rule__id 过滤记录，漏了一条就会重复推送。",
    "文档示例把 __mro__ 画成了图，看着比直接读代码清楚。",
    "审计日志按 user__id 归档，超过半年就转冷存储。",
    "配置里 redis__url 写错协议头，连接池一直起不来。",
    "他把 __ne__ 和 __eq__ 一起重载了，比较逻辑才算完整。",
    "报表里 org__code 与 dept__code 必须同时填，缺一个就算无效。",
    "函数签名里 data__path 参数被标成必填，调用时不能省。",
    "这段注释解释了 __all__ 的作用：控制 from 导入的范围。",
    "枚举名按 status__code 的写法统一，别混用驼峰。",
    "他演示时把 __index__ 讲成了下标方法，其实还有别的用途。",
    "字段 phone__no 加了脱敏，接口返回时只露前三位。",
    "示例里 item__id 是主键，删了它整条记录就找不回来。",
    "构建产物按 branch__name 分目录存放，避免互相覆盖。",
    "那段断言检查 cfg__path 是否存在，不存在就直接退出。",
    "他把 __enter__ 和 __exit__ 补齐，with 语句才正常工作。",
]

# A3：引号里转述多轮对话 ⇒ 真值 plain（口径：顶层结构是叙述句；naive 会判 multi_turn）
A3_TPL = [
    "他复述了那场争论：“用户：{x}。模型：{y}。”说完就不再提了。",
    "笔记里抄着一段对话：用户：{x} 模型：{y}",
    "她把早上的聊天记录念给大家听：“用户：{x}，模型：{y}。”",
    "聊天截图上写着：用户：{x} 模型：{y}，看得人一头雾水。",
    "老师在课上举了个例子：用户：{x}，模型：{y}。",
    "文章引用了一句对话：用户：{x}；模型：{y}。",
    "他把对话抄在纸上：用户：{x} 模型：{y}，然后撕掉了。",
    "他在群里贴了一段转述：\n用户：{x}\n模型：{y}\n然后补了一句“就这么回事”。",
    "会议记录里粘了两行原文：\n用户：{x}\n模型：{y}\n后面被他用一句话概括了。",
]

# A4（F2，对抗集）：候选列表被截断/残缺 ⇒ 真值 unknown（naive 会判 candidates）
A4_TPL = [
    "问：{q}|1）{p0}|2）{p1}",
    "问：{q}|1）{p0}|3）{p2}",
    "问：{q}|1）{p0}|2）{p1}|3）",
    "问：{q}|2）{p1}|3）{p2}|4）{p3}",
    "问：{q}|1）{p0}|2）{p1}|3）{p2}|4）{p3}|5）{p4}",
    "问：{q}|1）{p0}|2）",
    "问：{q}|1）{p0}|2）{p1}|4）{p3}",
    "问：{q}|3）{p2}",
    "问：{q}|1）{p0}|2）{p1}|3）{p2}|4）",
]

# A5（F2，对抗集）：既像 candidates 又像 cloze（空位落在候选条目里）⇒ 真值 unknown
A5_TPL = [
    "问：{q}|1）{p0}__{p1}|2）{p2}|3）{p3}|4）{p4}",
    "问：{q}|1）{p0}|2）{p1}__|3）{p2}|4）{p3}",
    "问：{q}|1）{p0}|2）{p1}|3）__{p2}|4）{p3}",
    "问：{q}|1）{p0}|2）{p1}|3）{p2}|4）{p3}__",
    "问：{q}|1）__{p0}|2）__{p1}|3）{p2}|4）{p3}",
    "问：{q}|1）{p0}{p1}__|2）{p2}|3）{p3}|4）{p4}",
    "问：{q}|1）{p0}|2）__{p1}|3）{p2}|4）{p3}",
    "问：{q}|1）{p0}|2）{p1}|3）{p2}|4）__",
    "问：{q}|1）__{p0}|2）{p1}|3）{p2}|4）{p3}",
]

# F1（训练/分布内 unknown）：与 F2 头部措辞不同（对抗集用 `问：`，F1 不用）
F1_TRUNC = [
    "选项|1）{p0}|2）{p1}|3）",
    "回答|1）{p0}|2）{p1}",
    "题目：{q}|1）{p0}|3）{p2}",
    "选一个|2）{p1}|3）{p2}|4）{p3}",
    "问题{q}|1）{p0}|2）{p1}|3）{p2}|4）",
    "候选|1）{p0}|2）",
]
F1_MIX = [
    "选项|1）{p0}__{p1}|2）{p2}|3）{p3}|4）{p4}",
    "题目：{q}|1）{p0}|2）{p1}|3）__{p2}|4）{p3}",
    "选一个|1）__{p0}|2）{p1}|3）{p2}|4）{p3}",
    "回答{q}|1）{p0}|2）{p1}__|3）{p2}|4）{p3}",
    "候选|1）{p0}|2）{p1}|3）{p2}|4）{p3}__",
]

SYMBOLS = "!@#$%^&*()[]{}<>?/\\|~`.,;:'\"-=+_" + "！＠＃￥％…＆＊（）【】〈〉《》、；：“”，。？—～·"
ASCII_GIBBERISH = "xq7v2b9m4k1z8w3f6j5t0y"
#: 空/超短/坏字符 池：长度 <4 或含替换符 U+FFFD ⇒ labeler 判 unknown
STUB_POOL = (
    [c + p for c in "嗯好哦行在无啊对吗呢吧哟嘿哈呀诶噢喂呃哎呗" for p in ("", "。", "！", "？")]
    + [f"乱�码{i}" for i in range(60)]
    + [f"测�试{i}" for i in range(60)]
)


def fills(k: int) -> dict[str, str]:
    """第 k 组填充（k 0..4），保证同模板下文本互不相同。"""
    return {
        "q": QS[k % len(QS)],
        "p0": PHR[k % len(PHR)],
        "p1": PHR[(k + 3) % len(PHR)],
        "p2": PHR[(k + 6) % len(PHR)],
        "p3": PHR[(k + 9) % len(PHR)],
        "p4": PHR[(k + 2) % len(PHR)],
        "a": OPTS[(2 * k) % 15], "b": OPTS[(2 * k + 1) % 15],
        "x": XS[k % 5], "y": YS[k % 5],
    }


def build_rows(category: str, true_label: str, templates: list[str],
               n_per_tpl: int, rows: list[dict]) -> None:
    seen_local: set[str] = set()
    for ti, tpl in enumerate(templates):
        for k in range(n_per_tpl):
            f = fills(k)
            if "{a}" in tpl or "{b}" in tpl:
                # A1 类模板的两个选项槽
                f = dict(f)
                f["a"] = OPTS[(2 * k + ti) % 15]
                f["b"] = OPTS[(2 * k + ti + 1) % 15]
            text = tpl.format(**f)
            check(text not in seen_local, f"{category} 模板{ti} 文本重复：{text}")
            seen_local.add(text)
            n_true = labeler(text)
            n_naive = naive(text)
            check(n_true == true_label,
                  f"{category} labeler={n_true} != 真值 {true_label}：{text}")
            if category.startswith("adv"):
                check(n_naive != true_label,
                      f"{category} 表面判据与真值相同（不具备对抗性）：{text} naive={n_naive}")
            rows.append({"id": f"{category}-{ti:02d}-{k:02d}", "category": category,
                         "rule": category, "true_label": true_label,
                         "naive_label": n_naive, "labeler_label": n_true,
                         "template_id": f"{category}.t{ti:02d}", "text": text})


def build_unknown_f1(n: int, seed: int) -> list[dict]:
    """训练/分布内 unknown：残缺 40% / 混合 25% / 乱码 20% / 空短超短 15%。"""
    rng = random.Random(seed)
    plan = [("unknown_trunc_F1", round(n * 0.40)), ("unknown_mix_F1", round(n * 0.25)),
            ("unknown_garbage", round(n * 0.20))]
    plan.append(("unknown_stub", n - sum(c for _, c in plan)))
    rows, seen = [], set()
    stub_idx = 0
    for rule, cnt in plan:
        made = 0
        guard = 0
        while made < cnt:
            guard += 1
            check(guard < cnt * 60, f"unknown 规则 {rule} 无法凑够 {cnt} 条")
            k = rng.randrange(5)
            f = {"q": rng.choice(QS), "p0": rng.choice(PHR), "p1": rng.choice(PHR),
                 "p2": rng.choice(PHR), "p3": rng.choice(PHR), "p4": rng.choice(PHR),
                 "a": OPTS[(2 * k) % 15], "b": OPTS[(2 * k + 1) % 15],
                 "x": XS[k % 5], "y": YS[k % 5]}
            if rule == "unknown_trunc_F1":
                text = rng.choice(F1_TRUNC).format(**f)
            elif rule == "unknown_mix_F1":
                text = rng.choice(F1_MIX).format(**f)
            elif rule == "unknown_garbage":
                ln = rng.randint(6, 18)
                if rng.random() < 0.5:
                    text = "".join(rng.choice(SYMBOLS) for _ in range(ln))
                else:
                    text = "".join(rng.choice(ASCII_GIBBERISH + "!@#$%^&*<>?/") for _ in range(ln))
            else:  # 空/超短/坏字符
                text = STUB_POOL[stub_idx]
                stub_idx += 1
                check(stub_idx <= len(STUB_POOL), "unknown_stub 池耗尽（fail-closed）")
            if text in seen:
                continue
            seen.add(text)
            check(labeler(text) == "unknown", f"F1 unknown labeler 失败：{text}")
            rows.append({"id": f"unk-{rule}-{made:03d}", "category": rule, "rule": rule,
                         "true_label": "unknown", "naive_label": naive(text),
                         "labeler_label": "unknown", "template_id": rule, "text": text})
            made += 1
    rng.shuffle(rows)
    return rows[:n]


# ---------------------------------------------------------------------------
# ID 数据源
# ---------------------------------------------------------------------------
ROLE_PREFIX = re.compile(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*")
SENT_SPLIT = re.compile(r"[。！？\n；;]+")


def plain_pool(n: int) -> list[str]:
    pool: list[str] = []
    seen: set[str] = set()
    for p in resolve_corpus_files(CORPUS_GLOB):
        with open(p, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                t = ROLE_PREFIX.sub("", line.strip())
                for s in SENT_SPLIT.split(t):
                    s = s.strip()
                    if not (6 <= len(s) <= 40) or s in seen:
                        continue
                    if labeler(s) != "plain" or naive(s) != "plain":
                        continue
                    seen.add(s)
                    pool.append(s)
                    if len(pool) >= n:
                        return pool
    return pool


def multi_pool(n: int, max_len: int = 100) -> list[str]:
    pool: list[str] = []
    seen: set[str] = set()
    for p in resolve_corpus_files(CORPUS_GLOB):
        with open(p, encoding="utf-8", errors="ignore") as fp:
            raw = fp.read()
        for block in re.split(r"\n\s*\n", raw):
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if len(lines) < 2 or len(block) > max_len:
                continue
            text = "\n".join(lines)
            if text in seen:
                continue
            if labeler(text) != "multi_turn" or naive(text) != "multi_turn":
                continue
            seen.add(text)
            pool.append(text)
            if len(pool) >= n:
                return pool
    return pool


def card_pool(name: str, label: str, need: int) -> list[str]:
    reg = all_tasks()
    raw = reg[name].build_dataset(max(need * 2, 800))
    texts = [d["text"] for d in raw]
    seen, uniq = set(), []
    for t in texts:
        if t in seen:
            continue
        seen.add(t)
        if labeler(t) != label or naive(t) != label:
            raise AssertionError(f"[build_data fail-closed] 卡样本判据不符 {name}/{label}: {t}")
        uniq.append(t)
    return uniq


def main() -> None:
    random.seed(SPLIT_SEED)

    # ---------- 对抗集 5 规则 × 45 ----------
    adv: list[dict] = []
    build_rows("adv_plain_mentions_candidates", "plain", A1_TPL, 5, adv)
    build_rows("adv_plain_underscore_id", "plain", A2_ITEMS, 1, adv)
    build_rows("adv_plain_quoted_dialogue", "plain", A3_TPL, 5, adv)
    build_rows("adv_truncated_candidates", "unknown", A4_TPL, 5, adv)
    build_rows("adv_mixed_cand_cloze", "unknown", A5_TPL, 5, adv)
    check(len(adv) == 225, f"对抗集应 225 条，实际 {len(adv)}")
    for rule, cnt in Counter(r["rule"] for r in adv).items():
        check(cnt >= 40, f"对抗规则 {rule} 只有 {cnt} 条（要求 ≥40）")

    # ---------- 已知答案 15 条（5 类 × 3，人工定稿） ----------
    ka_texts = [
        ("plain", "今天下午的会改到三楼会议室开了。"),
        ("plain", "他昨天坐地铁去的图书馆，路上没堵车。"),
        ("plain", "这把伞是上周下雨时在便利店买的。"),
        ("candidates", "问：明早的班车几点发|1）六点开门|2）七点检票|3）八点发车|4）九点才有"),
        ("candidates", "问：周末去哪儿玩合适|1）爬山|2）逛展|3）宅家|4）加班"),
        ("candidates", "问：这份报告什么时候交|1）周三之前|2）下周一|3）再等等|4）不知道"),
        ("cloze", "这份报告写得很__，一眼就能看懂|1）清楚|2）复杂|3）旅行|4）音乐"),
        ("cloze", "请你把＿＿处的错字改掉|1）原文|2）数字|3）段落|4）标题"),
        ("cloze", "这件事我早就跟他__了|1）说明|2）吵架|3）借钱|4）请假"),
        ("multi_turn", "用户：明早几点出发？\n模型：六点半在楼下等你。"),
        ("multi_turn", "用户：这道题怎么列式？\n模型：先把已知条件画成线段图。"),
        ("multi_turn", "用户：晚饭吃什么？\n模型：冰箱里还有昨天的菜，热一下就行。"),
        ("unknown", "问：怎么走最快|1）坐地铁|2）"),
        ("unknown", "问：哪一句合适|1）他__走了|2）她没走|3）都行|4）再看看"),
        ("unknown", "@#$%^&*()<>? 7x2"),
    ]
    ka = []
    for i, (lab, text) in enumerate(ka_texts):
        check(labeler(text) == lab, f"已知答案 labeler 不符 {lab}: {text}")
        ka.append({"id": f"ka-{i:02d}", "category": "known_answer", "rule": "handpicked",
                   "true_label": lab, "naive_label": naive(text), "labeler_label": lab,
                   "template_id": "handpicked", "text": text, "known_answer": True})
    check(len(ka) == 15, "已知答案应 15 条")

    # ---------- ID 数据 ----------
    need = N_TRAIN + N_TEST
    srcs: dict[str, list[dict]] = {}
    for lab, builder in [
        ("plain", lambda: plain_pool(need + 80)),
        ("multi_turn", lambda: multi_pool(need + 80)),
        ("unknown", lambda: build_unknown_f1(need + 80, SPLIT_SEED)),
        ("candidates", lambda: card_pool("reply_pick", "candidates", need)),
        ("cloze", lambda: card_pool("cloze_fill", "cloze", need)),
    ]:
        items = builder()
        check(len(items) >= need, f"{lab} 池不足：{len(items)} < {need}")
        random.Random(SPLIT_SEED + len(lab)).shuffle(items)
        pool_texts = [it if isinstance(it, str) else it["text"] for it in items]
        check(len(set(pool_texts)) == len(pool_texts), f"{lab} 池内有重复文本")
        rows = []
        for i, it in enumerate(items[:need]):
            if isinstance(it, str):
                text = it
                rule = f"corpus_{lab}" if lab in ("plain", "multi_turn") else "unknown_F1"
                tid = "pool"
            else:
                text, rule, tid = it["text"], it["rule"], it["template_id"]
            check(labeler(text) == lab, f"{lab} labeler 失败：{text}")
            rows.append({"id": f"{lab}-{i:04d}", "split": None, "category": lab,
                         "rule": rule, "true_label": lab, "naive_label": naive(text),
                         "labeler_label": lab, "template_id": tid, "text": text})
        srcs[lab] = rows

    id_train, id_test = [], []
    for lab in LABELS:
        rows = srcs[lab]
        id_train.extend([{**r, "split": "train"} for r in rows[:N_TRAIN]])
        id_test.extend([{**r, "split": "test"} for r in rows[N_TRAIN:need]])

    # ---------- 全局 fail-closed 校验 ----------
    check(len(id_train) == 5 * N_TRAIN, f"训练集 {len(id_train)}")
    check(len(id_test) == 5 * N_TEST, f"测试集 {len(id_test)}")
    tr, te = {r["text"] for r in id_train}, {r["text"] for r in id_test}
    check(not (tr & te), f"训练/测试文本重合 {len(tr & te)} 条")
    adv_t = {r["text"] for r in adv}
    ka_t = {r["text"] for r in ka}
    check(not (tr & adv_t), "训练与对抗集重合")
    check(not (te & adv_t), "测试与对抗集重合")
    check(not (adv_t & ka_t), "对抗集与已知答案重合")
    check(len(tr | te | adv_t | ka_t) == len(tr) + len(te) + len(adv_t) + len(ka_t),
          "四份数据之间存在重复文本")
    # 对抗集必须表面判据 ≠ 真值
    for r in adv:
        check(r["naive_label"] != r["true_label"], f"对抗样本不具对抗性：{r['id']}")
    # 分布内测试集必须 naive == true（表面无歧义，对抗性全部集中在对抗集）
    for r in id_test:
        if r["true_label"] != "unknown":
            check(r["naive_label"] == r["true_label"], f"ID 测试样本表层歧义：{r['id']} {r['text']}")

    def dump(name: str, rows: list[dict]) -> None:
        with open(OUT_DIR / name, "w", encoding="utf-8") as fp:
            for r in rows:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump("id_train.jsonl", id_train)
    dump("id_test.jsonl", id_test)
    dump("adversarial.jsonl", adv)
    dump("known_answers.jsonl", ka)

    stats = {
        "split_seed": SPLIT_SEED, "n_train": len(id_train), "n_test": len(id_test),
        "n_adv": len(adv), "n_known": len(ka),
        "train_label_dist": dict(Counter(r["true_label"] for r in id_train)),
        "test_label_dist": dict(Counter(r["true_label"] for r in id_test)),
        "adv_label_dist": dict(Counter(r["true_label"] for r in adv)),
        "adv_rule_dist": dict(Counter(r["rule"] for r in adv)),
        "adv_majority_baseline": max(Counter(r["true_label"] for r in adv).values()) / len(adv),
        "test_majority_baseline": max(Counter(r["true_label"] for r in id_test).values()) / len(id_test),
        "unknown_subrule_dist_train": dict(Counter(r["rule"] for r in id_train if r["true_label"] == "unknown")),
        "dup_check": {"train∩test": 0, "train∩adv": 0, "test∩adv": 0, "adv∩ka": 0},
        "naive_eq_true_in_id_test_nonunknown": True,
        "naive_ne_true_in_adv": True,
    }
    with open(OUT_DIR / "stats.json", "w", encoding="utf-8") as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
