"""否定标记切片数据集：圈出句中的否定标记（不 / 没 / 别 / 未 …），背景类零标记。

任务定义（过 dev-notes/12 §5 的两条筛选判据）：
  1. 否定标记是**闭集小词类** —— 类别语义可泛化（换帧探针能过，不像 idiom 靠背字串）；
  2. 值不是文档局部动态分配 —— 不需要 NDB。

类别只有两个：0 背景 / 1 否定；`span` = 否定标记本身在原文上的 0-based 半开区间。
dev-notes/05 §3.3 实测过情绪卡在否定式上翻车，这张卡把它变成独立可验收的任务。

三条 fail-closed 不变量（破坏任意一条都响亮抛错，绝不静默出数据）：
  1. **正例全标注**：含「同形非否定」（别人 / 未来 …）、「虚化固定式」（不得不 / 不过 …）
     或 A-not-A 疑问（是不是 / 有没有 …）的句子**整句丢弃** —— 这些字不是可标注的否定标记，
     留进正例会漏标（`annotate_all=True` 会炸），留进背景又违反「背景不得含否定词」；
  2. **背景零标记**：背景句一个标记子串都不能含（由提取器判空兜底，字面级干净）；
  3. **载体纯净**：合成载体模板自身不得含标记词；填入标记后必须**恰好**抽出这一个标记、
     且类别与位置正确；探针载体同样逐条校验（`validate_probe_carriers`）。

两条来自实测的载体纪律（dev-notes/05 §3.4 / dev-notes/06 §7）：
  - 训练与探针载体都必须覆盖**无终止标点的裸词形态**（只测带句号的载体时，
    裸词输入的定位会被截断：`开心 → 开`）；
  - 探针句式与训练句式**不得重合**，且类别错与定位错分开报。
"""
from __future__ import annotations

import random
import re
from collections import Counter
from contextlib import ExitStack

from dtseek.tasks.corpus import resolve_corpus_files

LABEL_BG = 0
LABEL_NEG = 1

#: 单样本最多几个标记。必须与 `negation/__init__.py` 里 SPEC.max_steps 一致
#: （测试 test_spec_max_steps_matches_dataset 会钉住这条）。
MAX_STEPS_PER_SAMPLE = 4

#: 正例占目标样本量的比例，其余为背景。
POS_RATIO = 0.65

#: 否定标记闭集。匹配按**长词优先**（不要 / 没有 必须先于 不 / 没 命中），
#: 与情绪卡「不高兴 先于 高兴」同一套机制。
NEG_MARKERS: tuple[str, ...] = (
    "不要", "不用", "没有", "别", "没", "不", "未", "莫", "勿", "难道", "何必",
)

#: 同形异词 / 虚化固定式：这里的「不 没 别 未 莫」不是可标注的否定标记。
#: **命中的句子整句丢弃**（既不进正例也不进背景）。
#: 末条守卫由 `_validate_markers` 强制：每个词条必须至少含一个标记字，
#: 否则就是死条目、一条句子都拦不住，会静默失效。
NEG_EXCEPTIONS: tuple[str, ...] = (
    # 别 = 「另外」义的同形（别人 / 别的 / 别墅…），不是祈使否定
    "别人", "别的", "区别", "分别", "特别", "性别",
    "告别", "离别", "永别", "送别", "辞别", "惜别", "话别", "阔别",
    "临别", "诀别", "识别", "鉴别", "辨别", "个别", "级别", "类别",
    "派别", "错别字", "别有", "别无", "别处", "差别", "之别", "有别",
    "别树", "别来", "别论",
    "别墅", "别针", "别名", "别称", "别离", "别扭", "别致",
    "别具", "别开", "别出",
    # 未 = 将来时前缀；莫 = 音译借词
    "未来", "莫斯科", "淹没", "没落", "沉没", "出没", "隐没", "神出鬼没",
    "莫里斯", "莫文蔚", "莫扎特", "莫奈", "莫言",
    "莫拉塔", "莫泊桑", "莫克", "莫愁", "莫伊",
    "莫里茨", "莫兰", "莫斯", "莫多", "莫达",
    # 否定已语法化的固定式（母语者不会认为句里「有否定」）
    "不得不", "不得已", "不过", "不然", "不管", "不论", "不仅", "不断",
    "不一会儿", "差不多", "不少", "不错", "了不起", "怪不得", "要不",
    "不必要", "不要紧", "不要求", "不要脸", "不要强", "不要好", "不要面子",
    "不用心", "不用功", "不用力",
)

#: A-not-A 疑问（是不是 / 有没有 / 去不去…）里的「不 / 没」是疑问形态，不是否定。
_RE_A_NOT_A = (
    re.compile(r"([一-龥])不\1"),
    re.compile(r"([一-龥])没\1"),
)

_ROLE_PREFIX = re.compile(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*")
_SENT_SPLIT = re.compile(r"[。！？!?；;，,\n]+")

_CJK_RE = re.compile(r"[一-龥]")
_CODE_SYMBOLS = frozenset("{}[]<>;$#\\`|~^=+_*/")
_ENGLISH_WORDS_RE = re.compile(r"[a-zA-Z]{2,}")


def reject_reason(text: str) -> str | None:
    """句子级丢弃原因；返回 None = 可用。同形 / 虚化 / A-not-A 一律整句丢弃。"""
    for exc in NEG_EXCEPTIONS:
        if exc in text:
            return f"同形/虚化固定式 {exc}"
    for pat in _RE_A_NOT_A:
        m = pat.search(text)
        if m:
            return f"A-not-A 疑问 {m.group(0)}"
    return None


def extract_negation_spans(text: str) -> list[dict]:
    """长词优先抽出全部否定标记，返回按 start 升序的 `[{"label", "word", "start", "end"}]`。

    这是「什么算否定标记」的唯一真相源：正例要求抽出的全部标注、背景要求抽出为空。
    不在这里处理例外 —— 例外走句子级丢弃（`reject_reason`），两边都不会拿到脏样本。
    """
    spans: list[dict] = []
    occupied = [False] * len(text)
    for word in sorted(NEG_MARKERS, key=len, reverse=True):
        i = text.find(word)
        while i != -1:
            end = i + len(word)
            if not any(occupied[i:end]):
                for k in range(i, end):
                    occupied[k] = True
                spans.append({"label": LABEL_NEG, "word": word, "start": i, "end": end})
            i = text.find(word, i + 1)
    spans.sort(key=lambda s: s["start"])
    return spans


def style_problems(s: str) -> list[str]:
    """语体门禁，与 relation 卡 `check_neutral_sentence_style` 同口径，返回违规描述。

    中文汉字 >= 70%、ASCII <= 15%、无代码符号、无英文单词 —— 拦截英文代码语料
    混进背景/正例（dev-notes 里 relation 卡踩过的语体污染，此处按同一道门禁拦）。
    """
    problems: list[str] = []
    length = len(s)
    if length == 0:
        return ["句子为空"]
    cjk_ratio = len(_CJK_RE.findall(s)) / length
    if cjk_ratio < 0.70:
        problems.append(f"中文汉字占比过低 ({cjk_ratio * 100:.1f}% < 70%)")
    ascii_ratio = sum(1 for c in s if c.isascii()) / length
    if ascii_ratio > 0.15:
        problems.append(f"ASCII 字符占比过高 ({ascii_ratio * 100:.1f}% > 15%)")
    code_hits = sorted({c for c in s if c in _CODE_SYMBOLS})
    if code_hits:
        problems.append(f"含代码符号: {''.join(code_hits)}")
    en_matches = sorted(set(_ENGLISH_WORDS_RE.findall(s)))
    if en_matches:
        problems.append(f"含英文单词: {en_matches}")
    return problems


# ---------------------------------------------------------------------------
# 合成训练载体：每个标记一套自己的句法框架（功能词没法像情绪形容词那样共用模板）。
# `{e}` 是标记本身；模板静态部分必须零标记、填充后必须恰好抽出这一个标记
# （`_validate_carriers` 强制）。每套都含**无终止标点**的裸词形态 ——
# dev-notes/05 §3.4 实测：训练集全带句号 → 裸词输入定位被截断（`开心 → 开`）。
# ---------------------------------------------------------------------------
CARRIERS_BY_MARKER: dict[str, tuple[str, ...]] = {
    "不要": (
        "{e}插嘴，让他说完。",
        "你{e}再催了。",
        "咱们{e}客气，直接开始吧。",
        "{e}跟我说这些客套话。",
        "我{e}闹了，行了吧？",
        "你{e}这样催我。",
        "{e}这样",
        "真的{e}再等下去了",
        "{e}吓我",
    ),
    "不用": (
        "你{e}解释，我们都懂。",
        "这{e}着急，明天再说。",
        "你{e}谢我，应该的。",
        "我{e}去了，你自己弄吧。",
        "先{e}拆，等评审完再动手。",
        "{e}管我，我有数。",
        "还{e}",
        "真的{e}这么急，明天也行。",
        "我看{e}绕这么大圈子了。",
    ),
    "没有": (
        "我{e}看到那封邮件。",
        "他{e}回消息。",
        "办公室{e}人了。",
        "我{e}听见他说什么。",
        "{e}什么问题吧？",
        "{e}了",
        "我问过了，{e}",
        "并{e}",
        "怎么{e}",
    ),
    "别": (
        "{e}着急，我马上到。",
        "{e}忘了带钥匙。",
        "你{e}这么说。",
        "先{e}急着下结论。",
        "{e}停，继续跑。",
        "{e}管闲事了。",
        "{e}闹",
        "千万{e}",
        "{e}回头再说",
    ),
    "没": (
        "我{e}看见那封邮件。",
        "他{e}回消息。",
        "桌上{e}杯子。",
        "你{e}吃饭就先走。",
        "我{e}反应过来，等下再说。",
        "你说的{e}道理。",
        "到底{e}听见。",
        "说了半天也{e}",
        "压根{e}",
    ),
    "不": (
        "我{e}去了。",
        "他{e}同意这个方案。",
        "这事{e}是他做的。",
        "现在说{e}晚了。",
        "这{e}难，再试一次。",
        "我{e}知道原因",
        "就是{e}",
        "{e}是这样的吗？",
        "先{e}急，我再想想。",
        "真{e}容易，先把需求过一遍。",
    ),
    "未": (
        "{e}提交审核材料。",
        "{e}通过审核的有三份。",
        "此申请{e}获批准。",
        "{e}满十八岁。",
        "{e}曾听说过这件事。",
        "{e}知全貌就下结论",
        "{e}必如此",
        "{e}置可否",
        "至今{e}",
    ),
    "莫": (
        "{e}见怪，他就是这性格。",
        "此事{e}要张扬。",
        "{e}等太久。",
        "{e}怪他。",
        "{e}慌",
        "{e}名其妙",
        "{e}要勉强。",
        "{e}怪自己。",
        "最好{e}",
    ),
    "勿": (
        "{e}忘带工牌。",
        "会场内{e}大声喧哗。",
        "{e}扰他人休息。",
        "{e}信谣传谣。",
        "{e}动",
        "{e}忘",
        "闲人{e}入。",
        "此处{e}停车。",
        "切{e}",
    ),
    "难道": (
        "{e}就这么算了？",
        "{e}是他做的？",
        "这{e}是巧合吗？",
        "{e}连这点小事都计较？",
        "{e}说",
        "{e}是我记错了",
        "他{e}知道这件事？",
        "{e}是你干的？",
        "这{e}",
    ),
    "何必": (
        "{e}呢？",
        "{e}为这点小事生气。",
        "我们{e}亲自跑一趟。",
        "既然已经定了，{e}改来改去。",
        "{e}纠结",
        "{e}绕这么远",
        "{e}为往事纠结。",
        "这{e}呢？",
    ),
}

#: 探针载体：与训练载体**句式不重合**（dev-notes/06 §7 —— 训练句式上考自己没有意义）。
#: `{w}` 是待测标记；每套至少一条无终止标点、且标记顶到句尾的裸词形态
#: （`我偏{w}` → `我偏不` —— 这正是 dev-notes/05 §3.4 定位被截断的那种输入）。
PROBE_CARRIERS_BY_MARKER: dict[str, tuple[str, ...]] = {
    "不要": (
        "{w}打断他，让他说完。",
        "你{w}催得这么紧。",
        "{w}把音量调这么大。",
        "谁{w}跑这一趟。",
        "你{w}",
        "{w}这么大声",
    ),
    "不用": (
        "真的{w}担心，事情会好起来。",
        "{w}绕这么大圈子。",
        "我看{w}特地跑一趟。",
        "大可{w}，我们再等等。",
        "你大可{w}",
        "{w}解释了",
    ),
    "没有": (
        "{w}收到通知就先开始。",
        "他{w}来过这里。",
        "我{w}带现金。",
        "{w}人接应，只能等。",
        "就{w}",
        "怎么{w}人告诉我",
    ),
    "别": (
        "{w}让他进来。",
        "{w}熬夜了，早点睡。",
        "你{w}把门关上。",
        "{w}吵醒他。",
        "劝你还是{w}",
        "{w}介意",
    ),
    "没": (
        "{w}着急，我这就来。",
        "他{w}说话就走了。",
        "到现在{w}回复。",
        "怎么也{w}想起来。",
        "好像{w}",
        "压根{w}注意到",
    ),
    "不": (
        "{w}能就这么走。",
        "大家{w}同意就散会。",
        "你{w}觉得呢",
        "谁都{w}清楚这件事",
        "我偏{w}",
        "万一{w}成功呢",
    ),
    "未": (
        "{w}婚的请举手。",
        "{w}按要求提交材料。",
        "该申请{w}通过初审。",
        "{w}曾收到任何通知。",
        "{w}成年的观众请排队。",
        "尚{w}",
    ),
    "莫": (
        "{w}要声张。",
        "此事{w}急，等他回来再说。",
        "{w}乱猜测。",
        "你{w}走那么快。",
        "{w}骂他",
        "{w}",
    ),
    "勿": (
        "{w}闯红灯。",
        "此处{w}吸烟。",
        "请{w}大声喧哗。",
        "{w}用手触摸展品。",
        "{w}忘带钥匙",
        "{w}",
    ),
    "难道": (
        "{w}是他先挑起的？",
        "他{w}已经知道结果了？",
        "我{w}说错了？",
        "事情{w}就这样算了？",
        "{w}连这点消息都传开了？",
        "{w}",
    ),
    "何必": (
        "{w}追这么远。",
        "你{w}亲自跑一趟。",
        "{w}冒这个险",
        "当初{w}答应下来",
        "又{w}",
        "{w}",
    ),
}

#: 真实背景句池：客观陈述 / 中性提问，一个否定标记字都不能含（构建时 fail-closed 校验）。
BACKGROUND_POOL: tuple[str, ...] = (
    "数据库集群写入延迟保持在五毫秒以内。",
    "系统将于今晚十二点进行常规版本发布与灰度观测。",
    "今天天气晴朗，气温约二十二度。",
    "白日依山尽，黄河入海流。",
    "高铁路网贯通南北，极大缩短了城际通勤时间。",
    "红富士苹果富含维生素，口感清甜多汁。",
    "深度学习模型正在加速推理计算过程。",
    "工业机器人按预设计划完成零件焊接组装。",
    "晨曦初现，山林间弥漫着淡淡的薄雾。",
    "气象台发布大风蓝色预警，请有关单位注意防范。",
    "该方案经过三轮评审，最终确定了实施路径。",
    "缓存命中率提升后，接口平均耗时下降了四成。",
    "新版固件修复了充电协议兼容性问题。",
    "园区绿化改造工程预计在下月完工。",
    "参考文献列出了近五年该领域的主要进展。",
    "请问这个功能具体要怎么使用",
    "今天的评审会议几点开始",
    "这个接口的返回格式是什么样的",
    "麻烦确认一下文档里写的参数含义",
    "这两个接口的超时时间一样吗",
    "现在的进度到哪一步了",
    "这个报错信息对应的原因是什么",
    "配置文件放在哪个目录下",
    "接下来的环节依次是哪些",
    "这个功能预计什么时候上线",
    "环境变量需要配置哪几个",
    "后台任务的执行频率是多少",
    "稍后把测试结果发我看一下",
)


# ---- fail-closed 校验 -------------------------------------------------------

def _validate_markers() -> None:
    """标记闭集与例外表自身的守卫：重复标记、死例外（不含任何标记字）都要响亮失败。"""
    problems: list[str] = []
    if len(set(NEG_MARKERS)) != len(NEG_MARKERS):
        problems.append(f"标记闭集有重复：{NEG_MARKERS}")
    if len(set(NEG_EXCEPTIONS)) != len(NEG_EXCEPTIONS):
        problems.append("例外表有重复词条")
    marker_set = set(NEG_MARKERS)
    for exc in NEG_EXCEPTIONS:
        if not any(m in exc for m in marker_set):
            problems.append(f"例外词条 {exc!r} 不含任何标记字 —— 死条目，一条句子都拦不住")
    if problems:
        raise ValueError("否定卡声明不合法：\n  - " + "\n  - ".join(problems))


def _synthetic_sample(text: str, word: str, *, where: str) -> dict:
    """合成句的 fail-closed 检查：不许带例外 / A-not-A，且必须恰好抽出这一个标记。"""
    reason = reject_reason(text)
    if reason is not None:
        raise ValueError(f"{where}：合成句 {text!r} 触发丢弃规则（{reason}）")
    spans = extract_negation_spans(text)
    if len(spans) != 1 or spans[0]["word"] != word:
        got = [(s["word"], s["start"], s["end"]) for s in spans]
        raise ValueError(
            f"{where}：填入 {word!r} 的合成句 {text!r} 应当恰好抽出 1 个 {word!r}，"
            f"实际 {got} —— 载体模板混入了其它标记或位置错误")
    if style_problems(text):
        raise ValueError(f"{where}：合成句 {text!r} 语体违规：{style_problems(text)}")
    if len(text) > 64:
        raise ValueError(f"{where}：合成句 {text!r} 超过编码窗口 64，span 会随截断漂移")
    return {"text": text, "spans": spans}


def validate_carriers(carriers: dict[str, tuple[str, ...]] | None = None) -> None:
    """载体模板 fail-closed：静态部分零标记；填充后恰好一个目标标记且语体干净。

    一次性报全部问题（参考 sentiment 卡 `_validate_carriers`）。
    """
    table = carriers if carriers is not None else CARRIERS_BY_MARKER
    bad: list[str] = []
    for word, templates in table.items():
        for tpl in templates:
            static = tpl.format(e="", w="")
            if any(m in static for m in NEG_MARKERS):
                bad.append(f"{word}: 模板 {tpl!r} 静态部分含标记（{static!r}）")
                continue
            if reject_reason(static) is not None:
                bad.append(f"{word}: 模板 {tpl!r} 静态部分触发丢弃规则")
                continue
            try:
                if "{e}" in tpl:
                    _synthetic_sample(tpl.format(e=word), word, where=f"训练载体 {tpl!r}")
                else:
                    filled = tpl.format(w=word)
                    if word in tpl:
                        raise ValueError(f"模板字面含待测词 {word!r}")
                    _synthetic_sample(filled, word, where=f"探针载体 {tpl!r}")
            except ValueError as exc:
                bad.append(str(exc))
    if bad:
        raise ValueError("否定卡载体校验失败：\n  - " + "\n  - ".join(bad))


def validate_probe_carriers(probe_table: dict[str, tuple[str, ...]] | None = None) -> None:
    """探针载体 fail-closed：`probe_units()` 每次调用都跑，坏载体绝不带进探针。

    额外两条 dev-notes/06 §7 的纪律在这里钉死：
      1. 每个标记至少一条**无终止标点且标记顶到句尾**的裸词形态；
      2. 模板字面不得含待测词（`probe.build_cases` 会直接抛错）。
    """
    table = probe_table if probe_table is not None else PROBE_CARRIERS_BY_MARKER
    problems: list[str] = []
    terminal = set("。！？!?；;，,")
    for word in NEG_MARKERS:
        templates = table.get(word)
        if not templates:
            problems.append(f"标记 {word!r} 没有探针载体")
            continue
        has_bare_tail = False
        for tpl in templates:
            if word in tpl:
                problems.append(f"{word}: 探针模板 {tpl!r} 字面含待测词 {word!r}")
                continue
            filled = tpl.format(w=word)
            try:
                _synthetic_sample(filled, word, where=f"探针载体 {tpl!r}")
            except ValueError as exc:
                problems.append(str(exc))
                continue
            if not (set(filled) & terminal) and filled.endswith(word):
                has_bare_tail = True
        if not has_bare_tail:
            problems.append(f"{word}: 探针载体缺少「无终止标点且标记顶句尾」的裸词形态")
    if problems:
        raise ValueError("否定卡探针载体校验失败：\n  - " + "\n  - ".join(problems))


def _validate_background_pool() -> None:
    """背景句池 fail-closed：一个标记子串都不能含，语体也必须干净。"""
    bad: list[str] = []
    for s in BACKGROUND_POOL:
        hits = [(x["word"], x["start"]) for x in extract_negation_spans(s)]
        probs = style_problems(s)
        if hits or probs:
            bad.append(f"    {s!r}: 含标记 {hits or ''} 语体 {probs or ''}")
    if bad:
        raise ValueError("否定卡背景句池污染（不得含任何否定标记）：\n" + "\n".join(bad))


# ---- 挖掘与构建 -------------------------------------------------------------

def _mine_real(pos_quota: int, bg_target: int, max_seq_len: int,
               max_lines: int | None) -> tuple[list[dict], list[dict], dict]:
    """一趟跨文件轮转扫描真实语料，同时挖正例与背景（`resolve_corpus_files` 唯一入口）。

    - 正例按**最稀缺标记优先**封顶（`per_marker_cap`）：一句话只在它覆盖的标记里
      当前样本最少的那个上计数，避免高频的「不 / 没」把长尾标记饿死；
    - 背景按**每文件配额**采集，防止字典序第一个文件独占背景池（relation 卡踩过）；
    - 语体门禁与丢弃规则全程 fail-closed：不合格的句子直接不收，并归因统计。
    """
    files = resolve_corpus_files()
    cap = 400
    pos: list[dict] = []
    bg: list[dict] = []
    seen: set[str] = set()
    counts: Counter = Counter()
    order = {w: i for i, w in enumerate(NEG_MARKERS)}
    per_file_bg = [0] * len(files)
    bg_quota_each = (bg_target + len(files) - 1) // len(files) * 3 // 2 + 8
    stats = {"lines": 0, "style": 0, "reject": 0, "too_many": 0, "dup": 0}
    done = False

    with ExitStack() as stack:
        fhs = [stack.enter_context(open(f, encoding="utf-8", errors="ignore")) for f in files]
        active = list(range(len(files)))
        while active and not done:
            nxt = []
            for i in active:
                line = fhs[i].readline()
                if not line:
                    continue
                stats["lines"] += 1
                if max_lines is not None and stats["lines"] > max_lines:
                    done = True
                    break
                text = _ROLE_PREFIX.sub("", line.strip())
                for s in _SENT_SPLIT.split(text):
                    s = s.strip()
                    if not (4 <= len(s) <= max_seq_len):
                        continue
                    if s in seen:
                        stats["dup"] += 1
                        continue
                    if style_problems(s):
                        stats["style"] += 1
                        continue
                    spans = extract_negation_spans(s)
                    if not spans:
                        if len(bg) < bg_target and per_file_bg[i] < bg_quota_each:
                            seen.add(s)
                            bg.append({"text": s, "spans": []})
                            per_file_bg[i] += 1
                        continue
                    if reject_reason(s) is not None:
                        stats["reject"] += 1
                        continue
                    if len(spans) > MAX_STEPS_PER_SAMPLE:
                        stats["too_many"] += 1
                        continue
                    if len(pos) >= pos_quota:
                        continue
                    words = {sp["word"] for sp in spans}
                    head = min(words, key=lambda w: (counts[w], order[w]))
                    if counts[head] >= cap:
                        continue
                    seen.add(s)
                    pos.append({"text": s, "spans": spans})
                    counts.update(words)
                nxt.append(i)
                if len(pos) >= pos_quota and len(bg) >= bg_target:
                    done = True
                    break
            active = nxt
    stats["pos"] = len(pos)
    stats["bg"] = len(bg)
    stats["files"] = len(files)
    return pos, bg, stats


def build_negation_dataset(target_samples: int = 6000, seed: int = 20240927,
                           per_marker_floor: int = 40, max_seq_len: int = 64,
                           max_lines: int | None = None) -> list[dict]:
    """构建否定标记切片数据集（合成保底 + 真实挖掘 + 背景零标记）。

    Args:
        target_samples: 目标总样本数，正例占 65%、背景占 35%，最终**恰好**返回该数量
            （样本量决定每 epoch 步数，与其它卡同为 6000 时步数口径不变）。
        seed: 随机种子；只影响背景裁剪与最终洗牌，改动它会改变数据集内容。
        per_marker_floor: 每个标记的合成保底条数（消灭「从未出现」，情绪卡 Zipf 长尾的教训）。
        max_lines: 调试/测试用的扫描行数上限；None = 扫完整个语料入口。
    """
    _validate_markers()
    validate_carriers()
    _validate_background_pool()
    validate_probe_carriers()

    rng = random.Random(seed)
    n_pos = int(target_samples * POS_RATIO)
    n_bg = target_samples - n_pos
    floor_total = per_marker_floor * len(NEG_MARKERS)
    if n_pos <= floor_total:
        raise ValueError(
            f"target_samples={target_samples} 装不下每标记保底 {floor_total} 条正例"
            f"（正例配额 {n_pos} <= floor_total）—— 加大 target_samples 或调小 per_marker_floor")

    # 1. 合成保底：每个标记恰好 per_marker_floor 条，消灭零覆盖
    pos: list[dict] = []
    for word in NEG_MARKERS:
        templates = CARRIERS_BY_MARKER[word]
        for k in range(per_marker_floor):
            text = templates[k % len(templates)].format(e=word)
            pos.append(_synthetic_sample(text, word, where=f"保底 {word}"))
    floor_n = len(pos)

    # 2. 真实语料：正例补到配额（最稀缺标记优先封顶）+ 背景按文件配额
    mined_pos, mined_bg, stats = _mine_real(
        pos_quota=n_pos - floor_n, bg_target=n_bg,
        max_seq_len=max_seq_len, max_lines=max_lines)
    pos.extend(mined_pos)

    # 3. 正例不足配额时按缺口继续合成（样本最少的标记优先，保持每标记均衡）
    rotate = {w: per_marker_floor for w in NEG_MARKERS}
    counts = Counter(sp["word"] for item in pos for sp in item["spans"])
    guard = 0
    while len(pos) < n_pos and guard < n_pos * 10:
        guard += 1
        word = min(NEG_MARKERS, key=lambda w: (counts[w], NEG_MARKERS.index(w)))
        templates = CARRIERS_BY_MARKER[word]
        tpl = templates[rotate[word] % len(templates)]
        rotate[word] += 1
        sample = _synthetic_sample(tpl.format(e=word), word, where=f"补齐 {word}")
        pos.append(sample)
        counts[word] += 1
    if len(pos) < n_pos:
        raise ValueError(f"正例只凑出 {len(pos)} < {n_pos}，合成补齐守卫触发 —— 数据集不完整")

    # 4. 背景：真实挖掘优先，不足用手写池补齐（同样零标记、语体已校验）
    bg = mined_bg[:n_bg]
    while len(bg) < n_bg:
        bg.append({"text": rng.choice(BACKGROUND_POOL), "spans": []})

    rng.shuffle(pos)
    rng.shuffle(bg)
    dataset = pos + bg
    rng.shuffle(dataset)

    synthetic_n = floor_n + (len(pos) - floor_n - stats["pos"])
    print(f"  否定标记闭集：{len(NEG_MARKERS)} 个 {'/'.join(NEG_MARKERS)}")
    print(f"  正例 {len(pos)}（真实挖掘 {stats['pos']} + 合成 {synthetic_n}，"
          f"保底每标记 {per_marker_floor} 条）| 背景 {len(bg)}（真实 {len(mined_bg[:n_bg])} + 手写补齐）")
    print(f"  语料扫描：{stats['lines']} 行 / {stats['files']} 文件"
          f"（丢弃：语体 {stats['style']} | 例外/A-not-A {stats['reject']} | "
          f"标记超限 {stats['too_many']} | 重复 {stats['dup']}）")
    print(f"  否定切片数据集构建完成：总计 {len(dataset)} 样本 ✅")
    for word in NEG_MARKERS:
        print(f"    标记 {word}: {counts[word]} 条")
    return dataset


def word_coverage_report(dataset: list[dict]) -> dict[str, int]:
    """每个标记在数据集里的样本量 —— 验证「每个标记都有下限」这条不变量。"""
    counter: Counter = Counter()
    for item in dataset:
        for sp in item.get("spans", []):
            counter[sp["word"]] += 1
    return {w: counter[w] for w in NEG_MARKERS}


if __name__ == "__main__":
    ds = build_negation_dataset(target_samples=4000)
    cover = word_coverage_report(ds)
    missing = [w for w, n in cover.items() if n == 0]
    print()
    print(f"标记覆盖：{cover}")
    if missing:
        raise SystemExit(f"零覆盖标记：{missing}")
