#!/usr/bin/env python3
"""构建对抗集：表面特征像卡 A、真正该路由到 B（或都不该用）的 5 类样本。

5 类（每类 45 条，合计 225 条）：
  c1 多代词·主任务情绪        表面 pronoun     → true sentiment
  c2 两成语·非词表关系对      表面 relation    → true neutral
  c3 真实人名·普通陈述        表面 person      → true neutral
  c4 否定词·整体语义积极      表面 negation字面 → true sentiment
  c5 中性句·含情绪词字面      表面 sentiment   → true neutral

fail-closed：任一条不过校验即丢弃并计数；某类不足 --per-cat 直接非 0 退出；
15 条已知答案（每类 3 条）必须全部通过校验，否则退出。
词表现取自 src/dtseek/tasks/builtin/（不在本文件复制词表）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dtseek.tasks.builtin.idiom.lexicon import IDIOMS  # noqa: E402
from dtseek.tasks.builtin.person.dataset import FEMALE_NAMES, MALE_NAMES  # noqa: E402
from dtseek.tasks.builtin.pronoun.dataset import PRONOUN_MAP  # noqa: E402
from dtseek.tasks.builtin.relation.dataset import ALL_PAIRS, VOCAB  # noqa: E402
from dtseek.tasks.builtin.sentiment.dataset import EMOTION_KEYWORDS  # noqa: E402

SEED = 20240927
PER_CAT = 45

CATS = ("c1", "c2", "c3", "c4", "c5")
CAT_META = {
    "c1": ("多代词·主任务情绪", "pronoun", "sentiment"),
    "c2": ("两成语·非词表关系对", "relation", "neutral"),
    "c3": ("真实人名·普通陈述", "person", "neutral"),
    "c4": ("否定词·整体语义积极", "negation", "sentiment"),
    "c5": ("中性句·含情绪词字面", "sentiment", "neutral"),
}

PRONOUNS = sorted({p for _, lst in PRONOUN_MAP for p in lst}, key=len, reverse=True)
EMO_BY_CAT = {cat: list(ws) for cat, ws in EMOTION_KEYWORDS}
ALL_EMO = [w for _, ws in EMOTION_KEYWORDS for w in ws]
CAT1_EMO = EMO_BY_CAT[1]
CAT23_EMO = sorted(EMO_BY_CAT[2] + EMO_BY_CAT[3], key=len, reverse=True)
NAMES = list(MALE_NAMES) + list(FEMALE_NAMES)

#: 公认成语补充：idiom 卡词表按「语料出现 >= 50 次」过滤，这批成语频次不够但确为成语。
HAND_IDIOMS = frozenset(
    "守株待兔 亡羊补牢 对牛弹琴 掩耳盗铃 刻舟求剑 井底之蛙 狐假虎威 "
    "南辕北辙 叶公好龙 画饼充饥 缘木求鱼".split()
)
IDIOM_OK = frozenset(IDIOMS) | HAND_IDIOMS
IDIOM_POOL = sorted(set(IDIOMS) - VOCAB)  # 生成用：成语且不在 relation 词表
PAIR_SET = {frozenset((a, b)) for a, b in ALL_PAIRS}

NEG_HARD = set("不没未非")  # C4 必须含（类1情绪词之外）/ C1·C3 必须不含
NEG_ADJ = set("不没未无")  # 类1情绪词左侧紧邻禁止（不高兴 / 不满意 之类）


def pronoun_count(text: str) -> int:
    occ = [False] * len(text)
    n = 0
    for p in PRONOUNS:
        start = 0
        while True:
            i = text.find(p, start)
            if i < 0:
                break
            if not any(occ[i:i + len(p)]):
                for k in range(i, i + len(p)):
                    occ[k] = True
                n += 1
            start = i + 1
    return n


def find_all(text: str, word: str) -> list[int]:
    out, start = [], 0
    while True:
        i = text.find(word, start)
        if i < 0:
            return out
        out.append(i)
        start = i + 1


def find_words(text: str, words: list[str]) -> list[str]:
    return [w for w in words if w in text]


def pair_cooccurs(text: str) -> tuple[str, str] | None:
    for a, b in ALL_PAIRS:
        if a in text and b in text:
            return (a, b)
    return None


def validate(cat: str, text: str) -> tuple[str | None, dict]:
    """返回 (reject_reason | None, evidence)。"""
    ev: dict = {"chars": len(text)}
    if not (6 <= len(text) <= 64):
        return "length", ev
    if pair_cooccurs(text) is not None:
        return "relation_pair_cooccur", ev

    if cat == "c1":
        pc = pronoun_count(text)
        emo = find_words(text, ALL_EMO)
        ev.update(pronouns=pc, emotion_words=emo)
        if pc < 3:
            return "pronouns_lt3", ev
        if not emo:
            return "no_emotion_word", ev
        if any(ch in text for ch in NEG_HARD):
            return "has_negation", ev

    elif cat == "c2":
        found = sorted({i for i in IDIOM_OK if i in text})
        ev.update(idioms=found)
        if len(found) < 2:
            return "idioms_lt2", ev
        for x in range(len(found)):
            for y in range(x + 1, len(found)):
                if frozenset((found[x], found[y])) in PAIR_SET:
                    return "idiom_pair_in_vocab", ev

    elif cat == "c3":
        hit = sorted({n for n in NAMES if n in text})
        pc = pronoun_count(text)
        ev.update(names=hit, pronouns=pc)
        if len(hit) != 1:
            return "names_not_exactly_one", ev
        if pc > 0:
            return "has_pronoun", ev
        if find_words(text, ALL_EMO):
            return "has_emotion_word", ev
        if any(ch in text for ch in NEG_HARD | {"别"}):
            return "has_negation", ev

    elif cat == "c4":
        pos = find_words(text, CAT1_EMO)
        other = find_words(text, CAT23_EMO)
        spans = [(i, i + len(w)) for w in set(pos) for i in find_all(text, w)]
        ev.update(pos_words=sorted(set(pos)), nonpos_words=other)
        if not pos:
            return "no_pos_emotion_word", ev
        if other:
            return "has_nonpositive_emotion", ev
        for i, _j in spans:
            if i > 0 and text[i - 1] in NEG_ADJ:
                return "negation_adjacent", ev
        covered = [False] * len(text)
        for i, j in spans:
            for k in range(i, j):
                covered[k] = True
        if not any(ch in NEG_HARD and not covered[k] for k, ch in enumerate(text)):
            return "no_free_negation", ev

    elif cat == "c5":
        emo = find_words(text, ALL_EMO)
        pc = pronoun_count(text)
        ev.update(emotion_words=emo, pronouns=pc)
        if not emo:
            return "no_emotion_word", ev
        if pc > 2:
            return "pronouns_gt2", ev

    return None, ev


# ── 手写池（known_answer 只从这里挑，构造期人工复核） ────────────────────────
HAND: dict[str, list[str]] = {
    "c1": [
        "我今天真的特别高兴，谢谢你一直陪着我。",
        "你要是看到我这么难过，肯定也会心疼我的。",
        "他一直说她太敏感了，可我觉得她只是委屈。",
        "我们都很担心你，你却笑着说一切正常。",
        "我们三个都很佩服你，你能扛下这么多压力真厉害。",
        "她今天特别开心，我看她确实很愉快。",
        "你猜我为什么这么开心？因为我终于升职了！",
        "你一脸难过，我看着心疼，他也跟着叹气。",
        "她们俩都替你高兴，你可得请客。",
        "我妈总说我太较真，结果我越想越生气。",
        "他和她为了谁去开会争了半天，我在旁边看得特别开心。",
        "你再这样下去，我可真要生气了，他也拿你无可奈何。",
        "老师看我一脸难过，就过来安慰我，我的眼泪一下就下来了。",
        "你和我都觉得他太过分了，他却还生着气。",
        "她和她同桌都笑得特别开心，我也跟着高兴。",
        "你带来的消息让我们全家都很兴奋，我爸妈高兴坏了。",
        "他总是担心自己发挥失常，其实我们都很佩服他。",
        "我跟你说，他今天在台上特别紧张，我看着都跟着紧张。",
        "你、我、他都看见她为了这次演出练了多久，我们都特别欣慰。",
        "你冲我发脾气的时候，我心里特别难受，他也被吓到了。",
        "我们都替她担心，她自己反倒很平静。",
        "他俩越说越激动，我、她、你听着也很开心。",
        "你和我都为他高兴，他也特别自豪。",
        "她总觉得自己受了冷落，我看了心里特别难受。",
        "领导夸了他，他高兴得一晚上都在笑，我们都替他开心。",
        "我们问他为什么这么兴奋，他只是笑个不停。",
        "大家都说他满面春风，她看着也紧张，我更紧张。",
        "她说她今天心情很好，我看她确实特别开心。",
        "我和你都佩服她的勇气，他也一样。",
        "你一出现，我们都特别兴奋，她也跟着欢呼。",
        "她们讨论得特别热烈，我和你也听得特别开心。",
        "我问你，他为什么这么沮丧。",
        "你把我惹得特别生气，她也在一旁跟着着急。",
        "我们都在为你高兴，他也盼着你开心呢。",
        "她埋怨我来得太晚，我心里也特别委屈。",
    ],
    "c2": [
        "守株待兔和画蛇添足这两个词要一起记。",
        "老师让我们区分掩耳盗铃和刻舟求剑。",
        "请在亡羊补牢和对牛弹琴下面画横线。",
        "他把井底之蛙和狐假虎威记在了同一个本子上。",
        "早读的时候，南辕北辙和叶公好龙被画了重点。",
        "作业要求给画饼充饥和缘木求鱼各造一个句子。",
    ],
    "c3": [
        "张伟今天早上七点到了公司。",
        "王芳在超市买了两斤苹果。",
        "李军已经搬去南京定居了。",
        "刘洋把自行车停在了地铁口。",
        "陈杰上周五去参加了招聘会。",
        "赵鹏的工位换到了三楼。",
        "黄涛在阳台上种了两盆多肉。",
        "周斌昨天下午三点出发去了合肥。",
        "徐峰把会议纪要发到了群里。",
        "孙浩在球场上跑了十圈。",
        "马超的座位靠窗。",
        "朱亮在厨房里煮了一锅汤。",
        "胡兵参加了市里的摄影展。",
        "郭勇把车开去了修理厂。",
        "林伟在书店里翻了半小时书。",
        "何军已经交完了房租。",
        "高翔在阳台上晒了被子。",
        "罗刚在门口等了五分钟。",
        "郑凯把钥匙忘在了办公室。",
        "梁辉在超市排队结账。",
        "谢东在公园里散步。",
        "宋健把快递放到了门卫处。",
        "有人在门口喊张敏的名字。",
        "前台给刘婷留了一个座位。",
        "陈静的电话号码换了。",
        "杨丽昨晚坐火车去了济南。",
        "赵敏把旧书捐给了社区。",
        "黄燕在花坛边浇了水。",
        "周雪报名参加了周末的义工活动。",
        "吴倩把窗户擦得很干净。",
        "徐丽在河边钓了一下午鱼。",
        "孙悦的作业本落在了教室。",
        "马丽在楼下便利店买了瓶水。",
        "朱琳把手机落在了出租车上。",
        "镇上的人都认识胡蝶。",
        "照片里郭静站在最后一排。",
    ],
    "c4": [
        "这个办法不算复杂，做完之后大家都很满意。",
        "这不失为一种好办法，大家都很满意。",
        "这结果不算差，评委们相当满意。",
        "他从不夸人，这次却对结果十分满意。",
        "这个项目没出岔子，进度比预期顺利，大家都很满意。",
        "虽然准备时间不长，但我们对这次表现相当满意。",
        "他没有放弃，最后交出的作品让所有人都满意。",
        "这次活动没出任何问题，家长们非常满意。",
        "别看它价格不高，用起来却让人很满意。",
        "这种做法未必最好，但足够让人满意。",
        "他说这不算什么，可大家都看出了他的高兴。",
        "这项服务没挑出毛病，顾客十分满意。",
        "这方案没花什么钱，效果却让经理很满意。",
        "难得他这次没皱眉，反而很满意地笑了。",
        "时间未到，大家却已经满意地开始鼓掌了。",
        "他并非第一次获奖，但这次格外满意。",
        "这家店没让我们多等，菜一上桌大家就都很满意。",
        "不得不说，这个方案确实让客户很满意。",
        "任务没拖到最后一刻，主管对进度很满意。",
        "这本书不算厚，内容却让人很满意。",
        "你没想到吧，这次比赛我们都没输，大家都很满意。",
        "没什么能拦住他们，他们对结果非常满意。",
        "准备工作没白做，客户对结果十分放心。",
        "会议未到一半，领导已经对方案相当满意。",
        "他没急着表态，神色里却透着满意。",
        "虽然山路未修平整，游客们对这趟行程很满意。",
        "这份答卷没挑出毛病，老师非常满意。",
        "雨未停，演员们却已满意地谢了幕。",
        "预算没超支，甲方对成品十分满意。",
        "这道菜我们都没尝过，评价却都很满意。",
    ],
    "c5": [
        "老师在黑板上写了「高兴」两个字。",
        "布告栏上写着「祝大家天天高兴」。",
        "他高不高兴都一样，会议照常进行。",
        "「不高兴」是这本绘本的书名。",
        "会议记录里「难过」这个词被划掉了。",
        "学生们把「开心」抄到了生词本上。",
        "这篇课文的生词表里有「愤怒」和「悲伤」。",
        "词典里「满意」的例句一共列了三条。",
        "他把「高兴」这个词念了三遍。",
        "新闻标题里「高兴」两个字被打上了引号。",
        "说明书上写着「如不满意请联系我们」。",
        "妈妈让他把「天天高兴」抄三遍。",
        "黑板上的「生气」和「难过」两个词还没擦掉。",
        "课文标题里有「愉快」这个词。",
        "他写作业时总把「高兴」写成「高与」。",
        "店名就叫「大家高兴」，其实卖的是文具。",
        "这句话里的「舒服」其实是个中性词。",
        "报纸把「焦虑」列入了本期生词表。",
        "他到底开不开心，得等他说了才算。",
        "名字叫「高兴」的同学今天没来。",
        "「满意」这两个字在作文里出现了四次。",
        "牌子上写着「高高兴兴上班」。",
        "他把「愉快的一天」写进了日记题目。",
        "「高兴」在汉语里是形容词，这一点大家都清楚。",
        "邻居给布娃娃取名叫「小欢喜」。",
        "贴在门上的「开心每一天」已经褪色了。",
        "这首歌的歌名里有「快乐」两个字。",
        "名单上写着「真高兴认识你」这句欢迎语。",
        "词典把「悲伤」标成了形容词。",
        "课本第十八页的插图说明里有「愉快」。",
        "他抽到的签上写着「天天开心」。",
        "横幅上「高高兴兴过周末」几个字很醒目。",
        "语文作业是抄写「难过」这个词各三遍。",
        "他把「生气」的表情画在了纸飞机上。",
        "词典第两百页能查到「忧虑」。",
        "小孩的衣服上印着「我很快乐」。",
        "广播稿的第一句是「祝大家愉快」。",
        "他把「高兴」两个字贴在了墙上。",
        "照片背面写着「那天真开心」。",
        "这首儿歌里反复出现「快乐」。",
        "标题「愤怒的葡萄」是一本书的名字。",
        "老师让用「愉快」这个词造句。",
        "他把「悲伤」写成了拼音。",
        "便签上画了个笑脸，写着「开心」。",
        "课文里「忐忑」出现了两次，老师让大家画出来。",
        "「高兴」和「难过」这两个词的部首不一样。",
        "广播里念到了「祝大家天天高兴」。",
        "墙上贴着的标语是「天天快乐」。",
        "他把「忧虑」念成了「忧滤」。",
        "这一页的例词是「满足」和「失望」。",
        "卡片背面印着「祝你愉快」。",
        "课桌侧面刻着「天天开心」四个字。",
    ],
}

# 已知答案对照：构造期逐条人工复核，先于探针出数（每类 3 条）
KNOWN_ANSWERS: dict[str, list[str]] = {
    "c1": [
        "我今天真的特别高兴，谢谢你一直陪着我。",
        "我们都很担心你，你却笑着说一切正常。",
        "你猜我为什么这么开心？因为我终于升职了！",
    ],
    "c2": [
        "守株待兔和画蛇添足这两个词要一起记。",
        "老师让我们区分掩耳盗铃和刻舟求剑。",
        "请在亡羊补牢和对牛弹琴下面画横线。",
    ],
    "c3": [
        "张伟今天早上七点到了公司。",
        "王芳在超市买了两斤苹果。",
        "李军已经搬去南京定居了。",
    ],
    "c4": [
        "这个办法不算复杂，做完之后大家都很满意。",
        "这结果不算差，评委们相当满意。",
        "他从不夸人，这次却对结果十分满意。",
    ],
    "c5": [
        "老师在黑板上写了「高兴」两个字。",
        "他高不高兴都一样，会议照常进行。",
        "会议记录里「难过」这个词被划掉了。",
    ],
}


# ── 生成池 ────────────────────────────────────────────────────────────────
C1_WORDS = [
    "开心", "高兴", "难过", "生气", "紧张", "担心", "兴奋", "激动", "满足",
    "安心", "佩服", "欣慰", "焦虑", "害怕", "委屈", "失望", "沮丧", "郁闷",
    "快乐", "愉快", "难受", "心痛",
]
C1_PRONOUNS = ["我", "你", "他", "她", "我们", "他们", "她们", "俺"]
C1_FRAMES = [
    "{p1}今天真的特别{e}，{p2}快来看看，{p3}也在等。",
    "{p1}和{p2}都替{p3}感到{e}。",
    "{p1}觉得这件事让{p2}很{e}，{p3}来评评理。",
    "{p1}、{p2}和{p3}都特别{e}。",
    "{p1}说{p2}今天特别{e}，{p3}快来看呢。",
    "{p1}问{p2}为什么这么{e}，{p3}也想知道。",
    "{p1}、{p2}都替{p3}高兴，你说呢？",
    "大家都说{p1}最近特别{e}，{p2}、{p3}都看出来了。",
    "{p1}和{p2}讨论了半天，{p3}听得特别{e}。",
    "{p1}发现{p2}心情低落就去安慰，{p3}也跟着去了。",
]

C2_FRAMES = [
    "{i1}和{i2}。", "{i1}、{i2}。", "{i1}和{i2}这两个词要一起记。",
    "老师让我们区分{i1}和{i2}。", "老师把{i1}和{i2}写在黑板上。",
    "老师让他用{i1}和{i2}各说一句话。", "昨天的听写里，{i1}和{i2}他都写对了。",
    "请在{i1}和{i2}下面画横线。", "作业要求给{i1}和{i2}各造一个句子。",
    "我们一起来读一读{i1}和{i2}。", "{i1}和{i2}，老师讲过好几遍了。",
    "这次考试的题目里出现了{i1}和{i2}。", "{i1}和{i2}的区别，你能说清楚吗？",
    "请说说{i1}和{i2}的区别。", "课文里出现了{i1}和{i2}。",
    "这本书里{i1}和{i2}都出现过。", "读课文的时候遇到{i1}和{i2}，要圈出来。",
    "这段话里的{i1}和{i2}用得都很恰当。", "作文里同时用了{i1}和{i2}，老师夸了他。",
    "这段文章用了{i1}和{i2}来对比。", "看到{i1}和{i2}，他就想起了那篇课文。",
    "读到{i1}和{i2}时，他停下来做了批注。", "他查了字典，弄明白了{i1}和{i2}。",
    "他把{i1}写成了{i2}。", "小明分不清{i1}和{i2}。",
    "他分不清{i1}和{i2}，就来问我。", "他把{i1}和{i2}记在了同一个本子上。",
    "他把{i1}、{i2}两个词抄了五遍。", "{i1}、{i2}这两个词他都会写。",
    "{i1}和{i2}，这两个词你认识吗？", "妈妈问我{i1}和{i2}的意思有什么差别。",
    "妹妹不认识{i1}，也不认识{i2}。", "{i1}和{i2}之间到底有什么不同？",
    "他第一次听到{i1}和{i2}是在同一篇课文里。", "{i1}和{i2}这两个词经常被放在一起讲。",
    "路上他一直在琢磨{i1}和{i2}。", "读书笔记里同时抄下了{i1}和{i2}。",
    "考试要求用{i1}和{i2}各写一句话。", "词典里{i1}和{i2}分在不同的页码。",
    "黑板角落里写着{i1}和{i2}。", "同桌问我{i1}和{i2}哪个更常见。",
    "预习的时候，{i1}和{i2}被画了重点。", "他把{i1}和{i2}的意思抄反了。",
    "晚饭后他还在复习{i1}和{i2}。", "试卷最后一题考了{i1}和{i2}。",
    "早读时全班一起念了{i1}和{i2}。", "他给{i1}和{i2}分别标了序号。",
]

C3_FRAMES = [
    "{n}今天早上七点到了公司。", "{n}在超市买了两斤苹果。",
    "{n}已经搬去南京定居了。", "{n}把自行车停在了地铁口。",
    "{n}上周五去参加了招聘会。", "图书馆门口有人在等{n}。",
    "{n}的工位换到了三楼。", "{n}在阳台上种了两盆多肉。",
    "{n}昨天下午三点出发去了合肥。", "{n}把会议纪要发到了群里。",
    "{n}在球场上跑了十圈。", "{n}的座位靠窗。",
    "{n}在厨房里煮了一锅汤。", "{n}参加了市里的摄影展。",
    "{n}把车开去了修理厂。", "{n}在书店里翻了半小时书。",
    "{n}已经交完了房租。", "{n}在门口等了五分钟。",
    "{n}把钥匙忘在了办公室。", "{n}在超市排队结账。",
    "{n}在公园里散步。", "{n}把快递放到了门卫处。",
    "快递员把包裹交给了{n}。", "前台给{n}留了一个座位。",
    "{n}的电话号码换了。", "有人在门口喊{n}的名字。",
    "{n}昨晚坐火车去了济南。", "{n}把旧书捐给了社区。",
    "社区的公告栏贴着{n}的名字。", "{n}在河边钓了一下午鱼。",
    "{n}报名参加了周末的义工活动。", "{n}把窗户擦得很干净。",
    "{n}在花坛边浇了水。", "照片里{n}站在最后一排。",
    "{n}的快递到了驿站。", "{n}替同事代了一节课。",
    "{n}从书架上取下了那本词典。", "班车在路口接上了{n}。",
    "{n}把奖金存进了银行。", "{n}今天休息待在家里。",
    "{n}在操场上走了两圈。", "隔壁师傅在修{n}家的水管。",
    "{n}的作业本落在了教室。", "{n}在楼下便利店买了瓶水。",
    "篮球场上{n}和队友在训练。", "{n}把手机落在了出租车上。",
    "镇上的人都认识{n}。", "{n}给邻居捎了一袋水果。",
]

C4_WORDS = ["满意", "放心", "安心", "愉快", "开心", "高兴", "佩服", "欣慰", "满足"]
C4_WHO = ["客户", "评委", "大家", "家长", "经理", "主管", "观众", "读者", "用户", "学员"]
C4_THING = ["方案", "服务", "安排", "产品", "演出", "课程", "展会", "改版"]
C4_FRAMES = [
    "准备工作没白做，{who}对结果十分{e}。",
    "这{thing}不算完美，不过{who}非常{e}。",
    "{who}没有立刻回答，但从神色看他们很{e}。",
    "会议未开完，{who}就已经对{thing}相当{e}。",
]


def gen_candidates(seed: int) -> dict[str, list[tuple[str, str]]]:
    """{cat: [(text, rule), ...]}，顺序确定。"""
    rng = random.Random(seed)
    out: dict[str, list[tuple[str, str]]] = {c: [] for c in CATS}

    for _ in range(150):
        p1, p2, p3 = rng.sample(C1_PRONOUNS, 3)
        e = rng.choice(C1_WORDS)
        fr = rng.choice(C1_FRAMES)
        out["c1"].append((fr.format(p1=p1, p2=p2, p3=p3, e=e), f"模板 {fr}"))

    used: set[tuple[str, str]] = set()
    made, tries = 0, 0
    while made < 80 and tries < 8000:
        tries += 1
        i1, i2 = rng.sample(IDIOM_POOL, 2)
        if (i1, i2) in used or (i2, i1) in used:
            continue
        used.add((i1, i2))
        fr = C2_FRAMES[made % len(C2_FRAMES)]
        out["c2"].append((fr.format(i1=i1, i2=i2), f"关系中立帧 + 词表外成语对 {i1}/{i2}"))
        made += 1

    for fi, fr in enumerate(C3_FRAMES):
        for shift in (0, 1, 2):
            n = NAMES[(fi + shift * 17) % len(NAMES)]
            out["c3"].append((fr.format(n=n), f"专名 {n} 单次出现 · 普通陈述"))

    for _ in range(150):
        e, who, thing = rng.choice(C4_WORDS), rng.choice(C4_WHO), rng.choice(C4_THING)
        fr = rng.choice(C4_FRAMES)
        out["c4"].append((fr.format(who=who, e=e, thing=thing), f"否定字面 + 积极词 {e}"))

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "adversarial.jsonl"))
    ap.add_argument("--per-cat", type=int, default=PER_CAT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    # 已知答案必须先过校验（fail-closed）
    for cat in CATS:
        for t in KNOWN_ANSWERS[cat]:
            reason, _ = validate(cat, t)
            if reason:
                print(f"[FATAL] 已知答案未过校验: {cat} {t!r} -> {reason}")
                return 1

    gen = gen_candidates(args.seed)
    rejects: Counter = Counter()
    samples: dict[str, list[dict]] = {c: [] for c in CATS}
    seen: set[str] = set()

    for cat in CATS:
        label, surface, true_label = CAT_META[cat]
        ka_set = set(KNOWN_ANSWERS[cat])
        ordered: list[tuple[str, str, bool]] = []
        for t in HAND[cat]:
            ordered.append((t, "手写", t in ka_set))
        for t, rule in gen[cat]:
            ordered.append((t, rule, False))
        ordered.sort(key=lambda x: not x[2])  # 已知答案排最前（稳定排序保持相对顺序）

        for text, rule, known in ordered:
            if len(samples[cat]) >= args.per_cat:
                break
            if text in seen:
                continue
            reason, ev = validate(cat, text)
            if reason:
                rejects[reason] += 1
                continue
            seen.add(text)
            samples[cat].append({
                "id": f"{cat}-{len(samples[cat]) + 1:02d}",
                "category": cat,
                "category_label": label,
                "surface_card": surface,
                "true_label": true_label,
                "text": text,
                "rule": rule,
                "known_answer": known,
                "evidence": ev,
            })

        if len(samples[cat]) < args.per_cat:
            print(f"[FATAL] {cat} 只有 {len(samples[cat])} 条 < {args.per_cat}；"
                  f"拒绝原因: {dict(rejects)}")
            return 1
        if sum(r["known_answer"] for r in samples[cat]) < 3:
            print(f"[FATAL] {cat} 已知答案不足 3 条进入最终集")
            return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [s for cat in CATS for s in samples[cat]]
    with open(out_path, "w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[build] 写出 {len(rows)} 条 -> {out_path}")
    for cat in CATS:
        rs = samples[cat]
        label, surface, true = CAT_META[cat]
        print(f"  {cat} {label:12s} {len(rs):3d} 条 | surface={surface:9s} "
              f"true={true:9s} | known_answer={sum(r['known_answer'] for r in rs)}")
    if rejects:
        print(f"[build] 校验淘汰 {sum(rejects.values())} 条: {dict(rejects)}")
    labels = Counter(r["true_label"] for r in rows)
    print(f"[build] 标签分布 {dict(labels)} | 经验多数类基线 "
          f"{max(labels.values()) / len(rows) * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
