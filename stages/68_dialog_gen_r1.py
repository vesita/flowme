#!/usr/bin/env python3
"""S68 · L2-r1 数据生成器：把 r0 的「跨轮引用」升级成「跨轮消解 + 干扰项排除」。

设计见 L2_PLAN.md §7（H2）。相对 r0（stages/66_dialog_gen.py）加两条难度：

  1. **干扰实体**：每一位说话人各引一个**同类目**物品（快递/预约/订座/售后/会员），
     但归属不同且**带角色线索**——一个是「我自己的」，另一个是「我朋友的」；
  2. **指代/省略**：轮3 用代词「它」或省略式指代（「帮我把它退了」），
     且轮3 **一个实体都不出现**（机器可查断言）。

目标回复必须点名**正确的那一个**实体 = 角色为「我自己的」那一个。
关键结构（不变量 D 的构造保证）：**哪个先说由记录序号奇偶严格 50/50 决定**：

    偶数条：轮1 = 我自己的{a}；轮2 = 我朋友的{b}
    奇数条：轮1 = 我朋友的{b}；轮2 = 我自己的{a}

两种顺序里正确实体恒为「我自己的」那个，而它有时先说、有时后说
⇒ 「取第一个出现的实体」这条**位置规则只能拿 50%**（逐条报出）；
⇒ 判定必须靠**角色线索**（我自己的 / 我朋友的），不是靠位置。

四条机器可查不变量（逐条断言 + 打印通过/违例计数；违例必须 0，否则退出码非 0）：
  A  轮3 用户话里**任何实体都不出现**（正确实体与干扰实体都查）+ 静态词表级检查；
  B  目标回复里**出现正确实体**且**不出现干扰实体**；
  C  ctx → target 是函数（ctx 唯一；同 ctx 不同 target 记违例）；
  D  判定不能只靠位置：位置规则 EM≈50%、角色规则 EM=100%，并逐条作为地板报出。

产物：logs/68_dialog_train.jsonl / logs/68_dialog_test.jsonl。
每条字段（沿用 r0 + r1 新增）：
  ctx / target / gold / ent（正确实体）/ ent_other（干扰实体）/ role
  / self_first（哪一方先被提到）/ t3 / t1_user / t1_sys / t2_user / t2_sys
  / family / v3 / n_turns / ent_count（恒=2）/ ent_pos / ent_pos_rank / t3_chars / t3_tok
  （计数类地板特征 + 供 69 的 --swap 用）

实体词表 40 个人名，train/test **各 20、互不相交**：这样"把 train 里最常见的实体填进去"
这条捷径在 test 上恰好归零，max_naive 由"模板填空"决定（贴地），任何涨分都只能来自
"从历史里挑对实体"。

★ `--shared-vocab`（L2-r1b 的**唯一变量**）：测试人名改从**与训练同一份** 20 人名里取
  （其余构造逐字不变：同一 RNG 消耗序、同一模板/填充/顺序位），落盘改写到
  logs/68b_dialog_{train,test}.jsonl（**绝不覆盖 r1 的 logs/68_dialog_* 与 69 的 ckpt**），
  并打印 [D68B]（规模 + 不变量计数 + 本臂自己的指纹 + 地板）。不带该开关时行为与 r1 逐字一致。

★ `--det-reply`（L2-r2 的**唯一变量**，隐含 `--shared-vocab`）：**回复句面不再随机抽**，
  改由**输入里可见的线索**唯一确定 ——
      规则 DET：句面 index := 该族「轮3 用户话（省略式）类型」的 index xt；
                族内 t3_user_forms 与 t3_sys_forms 长度恒等（=4，import 时断言）⇒ 一一对应，
                取模只是防御性写法。
      ⇒ (family, 省略式类型) → 句面 是一个函数；同一 (family, 省略式) **永远**同一句面；
      ⇒ 该线索逐字写在输入里（轮3 文本可见）⇒ 句面可学，§8.5 的"句面不可学"缺陷被修掉。
  其余构造**逐字不变**：仍然消耗那次 goldi 抽样（只是丢弃其值）⇒ **RNG 消耗序与 68b 完全一致**
  ⇒ 68d 的 ctx 与 68b **逐条相同**，唯一差别是 target（[D68D-CMP] 会实测这一点）。
  落盘 logs/68d_dialog_{train,test}.jsonl（**绝不覆盖 68/68b**），打印 [D68D]（规模 + 全部
  不变量计数 + DET 检查 + 指纹 + 地板）与新增的 [CEIL]（本档"完美实体+完美 v3"下的 EM 上限，
  **实算**，不写断言）。
  ★ 唯一没修、也不在本轮范围内的量：回复模板里的 {v3} 槽在轮3 文本里**只有第 1 种省略式可见**
    （每族 4 种里 1 种）⇒ 其余记录的 v3 在输入里不可见。这正是 [CEIL] 口径写"完美 v3"的原因，
    并由 [CEIL] 第二行显式报出"完美实体+完美句面、v3 只能均匀蒙"的期望可达上限。

指纹口径与 stages/65_textgen_cards.py::data_fp 逐字一致：逐条 len|ids 拼接后 sha256[:16]。
只写 logs/ 与本文件；不 import torch（纯 CPU、秒级、确定性）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import sys
from collections import Counter

NS_ROOT = "/home/vesita/coding/my/nanoSeek"
if NS_ROOT not in sys.path:
    sys.path.insert(0, NS_ROOT)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
TOK_PATH = NS_ROOT + "/data/chinese/char_tokenizer.json"
OUT_TMPL = os.path.join(LOG_DIR, "68_dialog_{split}.jsonl")

N_TRAIN = 7000
N_TEST = 800
SEED = 20261010
EOS = "<eos>"

# ---------------- 闭合实体词表（40 个人名；train/test 各 20，互不相交） ----------------
# ★互不相交是**故意的**：r0 的 train/test 共享词表，47 个计数类"多数词"规则的 EM 靠
#   "记住 train 里那个实体"就能拿到分；r1 要判的是**消解**，所以让「把 train 多数实体
#   填进去」这条捷径在 test 上恰好归零，max_naive 由 ④模板填空（实体位填 train 多数实体）
#   决定，门槛因此贴地；任何涨分都只能来自"从历史里挑对实体"。
NAMES_TRAIN = (
    "张伟", "李娜", "王芳", "刘洋", "陈静", "杨帆", "赵磊", "黄敏", "周涛", "吴倩",
    "徐强", "孙丽", "马超", "朱婷", "胡军", "郭涛", "林晓", "何静", "高翔", "罗敏",
)
NAMES_TEST = (
    "郑爽", "梁静", "谢军", "宋佳", "唐磊", "许峰", "韩雪", "冯刚", "曹阳", "邓超",
    "彭飞", "曾毅", "萧然", "田甜", "董浩", "潘玮", "蒋雯", "蔡明", "余音", "杜鹃",
)
NAMES = NAMES_TRAIN + NAMES_TEST
TRAIN_SET, TEST_SET = set(NAMES_TRAIN), set(NAMES_TEST)

# ---------------- 角色词（与位置无关的线索；正确实体恒为「我的」） ----------------
ROLE_SELF = "我的"
ROLE_OTHER = "朋友的"
# 干扰实体的引入句式（3 种，都点明「朋友的」+ 同类目物品；轮2 必须含错实体）
T2U_VARIANTS = (
    "对了，我朋友用{other}的账号也{act}，你顺便记一下。",
    "我朋友说{other}名下也{act}，跟我的不是同一个。",
    "顺便说一句，我朋友拿{other}的账号{act}过。",
)
T2S_VARIANTS = (
    "好的，你朋友的{other}我也记下了，需要我处理的时候说一声。",
    "明白，{other}那边是你朋友的，我先只记着。",
    "收到，你朋友的{other}和你自己的我分开了。",
)

# ---------------- 轮2 其他填充（与任务无关；只提供 ctx 唯一性） ----------------
# ★这些句子不得含任何实体名、也不得含「我的 / 朋友的」（静态检查会断言）。
T2_TOPICS = (
    "今天天气不错，适合出门散步。",
    "最近公司的事情有点多。",
    "我昨天看了一部电影，挺好看的。",
    "周末打算在家休息一下。",
    "早上出门的时候有点堵车。",
    "楼下新开了一家面包店。",
    "我最近在学做菜，还在摸索。",
    "这几天降温挺明显的。",
    "中午和同事一起吃了碗面。",
    "我养的那盆绿萝长出新叶子了。",
    "昨天睡得有点晚，今天没什么精神。",
    "最近在追一部纪录片。",
    "明天要去机场接个朋友。",
    "我家猫又把杯子推下桌子了。",
    "这段时间在整理以前的照片。",
    "晚上准备去楼下跑跑步。",
    "刚泡了杯茶，准备看会儿书。",
    "上个月买的书还没拆封。",
    "隔壁在装修，白天有点吵。",
    "今天地铁上人特别多。",
    "打算下周请两天假出去走走。",
    "我最近在练习早睡早起。",
    "昨天试着做了一道新菜。",
    "阳台上晒的被子收进来了。",
    "手机换了新电池，续航好多了。",
    "楼下的桂花开了，味道很好闻。",
    "周末想去逛逛旧书店。",
    "今天的工作总算告一段落。",
    "刚把房间彻底打扫了一遍。",
    "朋友推荐了一家不错的咖啡店。",
    "我最近迷上了拼图。",
    "早上煮了个鸡蛋当早餐。",
    "这两天在收拾换季的衣服。",
    "同事送了我一盆小多肉。",
    "昨天散步走了很远。",
    "新买的台灯亮度挺合适。",
    "傍晚的天空很好看。",
    "我打算把作息调回正常。",
    "今天路过公园，花都开了。",
    "晚上约了朋友一起吃火锅。",
)

# 轮1 系统话的收尾（与任务无关；提供 ctx 唯一性 —— 7,800 条里组合会撞车）
R1S_TAILS = (
    "稍后可以在我的订单里看到。",
    "如果需要修改随时说一声。",
    "记录编号已经生成。",
    "有变化我会第一时间通知您。",
    "详细信息也在站内信里。",
    "到期前会有提醒。",
)

# ---------------- 5 个「族」= 5 套（轮1/轮2/轮3/目标）模板 ----------------
# ★ r1 的关键结构（不变量 D 的构造保证）：
#   同一族有**两种顺序**，由记录序号奇偶**严格 50/50** 决定：
#     顺序 0（自己先说）：轮1 = 我自己的东西{a}；轮2 = 「我朋友的{other}」{b}；
#     顺序 1（朋友先说）：轮1 = 「我朋友的{other}」{b}；轮2 = 我自己的东西{a}；
#   两种顺序里**正确实体永远是「我自己的」那个**，而它有时先说、有时后说
#   ⇒ 「取第一个出现的实体」这条位置规则只能拿到 ≈50%（实测见 [INV] D 行）；
#   ⇒ 判定必须靠**角色线索**（我自己的 / 我朋友的），不是靠位置。
#
# 轮3 一律是代词「它」或省略式，且**一个实体都不出现**（构造 + 静态检查双保险）。
FAMILIES = (
    {
        "key": "快递",
        "item": "快递",                       # 类目词（只说类目、不指名）
        "t1_act": "寄了一个包裹",
        "t2_act": "寄了个包裹",
        # 轮1 = 我自己的东西（顺序 0 用）：三种句面，都带「我自己」的自指
        "t1_self_forms": (
            "我叫{a}，我自己的{item}上礼拜就寄了。",
            "我是{a}，我自己的那个{item}麻烦你看一下。",
            "{a}你好，我自己的{item}还在你那边吧？",
        ),
        # 轮1 = 朋友的东西（顺序 1 用）：朋友先说，替他转达
        "t1_friend_forms": (
            "我朋友的{other}也{act2}，我让我朋友帮忙问一声。",
            "我朋友用{other}的账号{act2}，我朋友说这边能查。",
            "我朋友的{other}{act2}，麻烦你也记一下。",
        ),
        # 轮2 = 我自己的东西（顺序 1 用）
        "t2_self_forms": (
            "我自己的{item}也{act2}，我叫{a}，麻烦你记一下。",
            "我自己的{item}是我的，{a}，刚才忘说了。",
            "对了我自己的{item}也{act2}，{a}，一起记着吧。",
        ),
        "t2_friend_forms": (
            "我朋友用{other}的账号也{act2}，你先记着。",
            "我朋友的{other}也{act2}，跟我的不是同一个。",
            "我朋友拿{other}的账号{act2}过。",
        ),
        "t1_sys_forms": (
            "好的{a}，您自己的{item}已经登记在您名下。",
            "收到{a}。",
            "嗯，{a}，我记下了。",
        ),
        "t2_sys_forms": (
            "好的，您朋友的{other}我也记下了，需要时告诉我。",
            "明白，{other}那边的我不动。",
            "收到，我先记着。",
        ),
        "t3_user_forms": (
            "帮我把它退了吧，我不想要了。",
            "那件还是帮我退了吧，{v3}方便吗？",
            "那个帮我退一下，谢谢。",
            "还是退了吧，麻烦你了。",
        ),
        "t3_sys_forms": (
            "{a}，您这个包裹已按退回处理，{v3}会有快递员联系您。",
            "{a}，退回申请已提交，您的包裹预计{v3}上门取件。",
            "{a}，已为您办理退回，{v3}快递员会上门取件。",
            "{a}，退回已受理，{v3}前快递员会来取件。",
        ),
        "v3": ("明天下午", "后天早上", "这周五", "下周一"),
    },
    {
        "key": "预约",
        "item": "牙科门诊",
        "t1_act": "约了牙科门诊",
        "t2_act": "约了牙科门诊",
        "t1_self_forms": (
            "我叫{a}，我自己的牙科门诊是约在这周的。",
            "我是{a}，我自己那个牙科号还在吧？",
            "{a}，我自己的牙科预约你帮我记着了吧。",
        ),
        "t1_friend_forms": (
            "我朋友的{other}也{act2}，我让我朋友帮忙问一声。",
            "我朋友用{other}的账号{act2}，我朋友说这边能查。",
            "我朋友的{other}{act2}，麻烦你也记一下。",
        ),
        "t2_self_forms": (
            "我自己的{item}也{act2}，我叫{a}，麻烦你记一下。",
            "我自己的牙科号是我的，{a}，刚才忘说了。",
            "对了我自己的{item}也{act2}，{a}，一起记着吧。",
        ),
        "t2_friend_forms": (
            "我朋友用{other}的账号也{act2}，你先记着。",
            "我朋友的{other}也{act2}，跟我的不是同一个。",
            "我朋友拿{other}的账号{act2}过。",
        ),
        "t1_sys_forms": (
            "好的{a}，您自己的牙科门诊已经登记在您名下。",
            "收到{a}。",
            "嗯，{a}，我记下了。",
        ),
        "t2_sys_forms": (
            "好的，您朋友的{other}我也记下了，需要时告诉我。",
            "明白，{other}那边的我不动。",
            "收到，我先记着。",
        ),
        "t3_user_forms": (
            "帮我把它取消掉吧。",
            "把它退了吧，{v3}我有空再约。",
            "那个号帮我退掉，谢谢。",
            "先取消吧，我改天再去。",
        ),
        "t3_sys_forms": (
            "{a}，您这个门诊已取消，想再约的时候{v3}还有号。",
            "{a}，预约已取消，{v3}之后可以重新安排您的门诊。",
            "{a}，已为您取消预约，{v3}起可以再约。",
            "{a}，您的预约已取消，{v3}还有空号。",
        ),
        "v3": ("周三上午", "周四下午", "周六上午", "周日下午"),
    },
    {
        "key": "订座",
        "item": "靠窗的位子",
        "t1_act": "订了个靠窗的位子",
        "t2_act": "订了个靠窗的位子",
        "t1_self_forms": (
            "我叫{a}，我自己的位子是订在靠窗那边的。",
            "我是{a}，我自己的靠窗那桌你留着了吧？",
            "{a}，我自己的订的那桌还在吧？",
        ),
        "t1_friend_forms": (
            "我朋友的{other}也{act2}，我让我朋友帮忙问一声。",
            "我朋友用{other}的账号{act2}，我朋友说这边能查。",
            "我朋友的{other}{act2}，麻烦你也记一下。",
        ),
        "t2_self_forms": (
            "我自己的{item}也{act2}，我叫{a}，麻烦你记一下。",
            "我自己的靠窗那桌是我的，{a}，刚才忘说了。",
            "对了我自己的{item}也{act2}，{a}，一起记着吧。",
        ),
        "t2_friend_forms": (
            "我朋友用{other}的账号也{act2}，你先记着。",
            "我朋友的{other}也{act2}，跟我的不是同一个。",
            "我朋友拿{other}的账号{act2}过。",
        ),
        "t1_sys_forms": (
            "好的{a}，您自己那桌已经登记在您名下。",
            "收到{a}。",
            "嗯，{a}，我记下了。",
        ),
        "t2_sys_forms": (
            "好的，您朋友的{other}我也记下了，需要时告诉我。",
            "明白，{other}那边的我不动。",
            "收到，我先记着。",
        ),
        "t3_user_forms": (
            "帮我把它改成三个人吧。",
            "那桌先按{v3}来改吧，谢谢。",
            "那桌帮我加个位子，谢谢。",
            "改成四个人吧，麻烦你了。",
        ),
        "t3_sys_forms": (
            "{a}，您这个位子已改成{v3}的桌子，靠窗保留。",
            "{a}，已为您把桌位改成{v3}，靠窗那桌还是留着的。",
            "{a}，{v3}的位子已经排好，您靠窗桌位不变。",
            "{a}，您那桌已按{v3}调整，靠窗还是您的。",
        ),
        "v3": ("三个人", "四个人", "五个人", "六个人"),
    },
    {
        "key": "售后",
        "item": "咖啡机",
        "t1_act": "买了台咖啡机",
        "t2_act": "买了台咖啡机",
        "t1_self_forms": (
            "我叫{a}，我自己的那台咖啡机有点问题。",
            "我是{a}，我自己的咖啡机昨天开始漏水了。",
            "{a}你好，我自己的咖啡机得找人来看看。",
        ),
        "t1_friend_forms": (
            "我朋友的{other}也{act2}，我让我朋友帮忙问一声。",
            "我朋友用{other}的账号{act2}，我朋友说这边能查。",
            "我朋友的{other}{act2}，麻烦你也记一下。",
        ),
        "t2_self_forms": (
            "我自己的{item}也{act2}，我叫{a}，麻烦你记一下。",
            "我自己的咖啡机是我的，{a}，刚才忘说了。",
            "对了我自己的{item}也{act2}，{a}，一起记着吧。",
        ),
        "t2_friend_forms": (
            "我朋友用{other}的账号也{act2}，你先记着。",
            "我朋友的{other}也{act2}，跟我的不是同一个。",
            "我朋友拿{other}的账号{act2}过。",
        ),
        "t1_sys_forms": (
            "好的{a}，您自己那台已经登记在您名下。",
            "收到{a}。",
            "嗯，{a}，我记下了。",
        ),
        "t2_sys_forms": (
            "好的，您朋友的{other}我也记下了，需要时告诉我。",
            "明白，{other}那边的我不动。",
            "收到，我先记着。",
        ),
        "t3_user_forms": (
            "帮我把它送去修吧。",
            "那台咖啡机{v3}能安排维修吗？",
            "那台帮我报个修，谢谢。",
            "直接安排维修吧，麻烦你了。",
        ),
        "t3_sys_forms": (
            "{a}，您这台咖啡机已安排维修，师傅{v3}上门。",
            "{a}，维修单已提交，{v3}师傅会上门看您的咖啡机。",
            "{a}，已为您报修，{v3}前会有师傅联系您。",
            "{a}，您的咖啡机已报修，师傅{v3}到。",
        ),
        "v3": ("明天上午", "明天下午", "周六上午", "周日下午"),
    },
    {
        "key": "会员",
        "item": "会员积分",
        "t1_act": "开了会员",
        "t2_act": "开了会员",
        "t1_self_forms": (
            "我叫{a}，我自己的会员积分攒了不少了。",
            "我是{a}，我自己的会员卡还能用吧？",
            "{a}，我自己的会员你帮我查一下积分。",
        ),
        "t1_friend_forms": (
            "我朋友的{other}也{act2}，我让我朋友帮忙问一声。",
            "我朋友用{other}的账号{act2}，我朋友说这边能查。",
            "我朋友的{other}{act2}，麻烦你也记一下。",
        ),
        "t2_self_forms": (
            "我自己的{item}也{act2}，我叫{a}，麻烦你记一下。",
            "我自己的积分是我的，{a}，刚才忘说了。",
            "对了我自己的{item}也{act2}，{a}，一起记着吧。",
        ),
        "t2_friend_forms": (
            "我朋友用{other}的账号也{act2}，你先记着。",
            "我朋友的{other}也{act2}，跟我的不是同一个。",
            "我朋友拿{other}的账号{act2}过。",
        ),
        "t1_sys_forms": (
            "好的{a}，您自己的会员已经登记在您名下。",
            "收到{a}。",
            "嗯，{a}，我记下了。",
        ),
        "t2_sys_forms": (
            "好的，您朋友的{other}我也记下了，需要时告诉我。",
            "明白，{other}那边的我不动。",
            "收到，我先记着。",
        ),
        "t3_user_forms": (
            "帮我把积分换成十元券吧。",
            "那些积分按{v3}帮我兑了吧。",
            "那些积分帮我兑了吧，谢谢。",
            "帮我把积分兑成二十元券。",
        ),
        "t3_sys_forms": (
            "{a}，您的积分已兑换成{v3}，消费时可用。",
            "{a}，{v3}已发到您的账户，下次消费直接抵。",
            "{a}，已为您兑换{v3}，您的积分余额已清零。",
            "{a}，您的积分已换成{v3}，请查收。",
        ),
        "v3": ("十元券", "二十元券", "三十元券", "五十元券"),
    },
)

FAM_BY_KEY = {f["key"]: f for f in FAMILIES}
N_TEMPLATE_FAMILIES = len(FAMILIES)
N_FAMILY_FORMS = sum(len(f["t1_self_forms"]) + len(f["t1_friend_forms"])
                     + len(f["t2_self_forms"]) + len(f["t2_friend_forms"]) for f in FAMILIES)
N_T3_FORMS = sum(len(f["t3_user_forms"]) for f in FAMILIES)
N_REPLY_TEMPLATES = sum(len(f["t3_sys_forms"]) for f in FAMILIES)
N_V3 = sum(len(f["v3"]) for f in FAMILIES)

# ---------------- L2-r2：确定性回复句面（--det-reply）的规则（写死） ----------------
# DET：句面 index := 该族「轮3 用户话（省略式）类型」的 index xt（族内两列长度恒等 ⇒ 一一对应）。
#      (family, 省略式类型) → 句面 是函数，与 RNG 无关、与实体/顺序/v3 全部无关。
assert all(len(f["t3_user_forms"]) == len(f["t3_sys_forms"]) for f in FAMILIES), \
    "DET 规则要求每族 t3_user_forms 与 t3_sys_forms 等长"


def det_reply_index(fam, xt: int) -> int:
    """规则 DET：句面 index = 轮3 省略式类型 index（同一 (family, 省略式) 恒同一句面）。"""
    n = len(fam["t3_sys_forms"])
    assert len(fam["t3_user_forms"]) == n, f"族 {fam['key']} 两列长度不等 ⇒ DET 未定义"
    return xt % n


def _t3_form_index(fam, t3: str):
    """只用轮3 文本复原它在族内的句面 index（= 输入可见线索）；识别不出返回 None。"""
    for i, form in enumerate(fam["t3_user_forms"]):
        head, sep, tail = form.partition("{v3}")
        if not sep:
            if t3 == form:
                return i
            continue
        if t3.startswith(head) and (t3.endswith(tail) if tail else True):
            mid = t3[len(head): len(t3) - len(tail)] if tail else t3[len(head):]
            if mid in fam["v3"]:
                return i
    return None


# ---------------- 渲染：与 nanoSeek DialogueStream(用户：/ <resp> / <eos>) 同款 ----------------
def render_ctx(r1u: str, r1s: str, r2u: str, r2s: str, r3u: str) -> str:
    """手写权威格式；main 里用 DialogueStream 逐条核对（必须是同一条流）。

    ★ r2s 为空时**不写** `<resp>` 那一行 —— 与 DialogueStream 一致：`commit("")`
      不会新增任何句子（`append` 把空句过滤掉），所以日志里根本没有那一轮。
      漏掉这条差异就是 3907/7800 条与 `prompt()` 不一致的原因（已实测修正）。
    """
    out = f"用户：{r1u}\n<resp>{r1s}{EOS}\n"
    if r2s:
        out += f"用户：{r2u}\n<resp>{r2s}{EOS}\n"
    else:
        out += f"用户：{r2u}\n"
    return out + f"用户：{r3u}\n<resp>"


# ---------------- 名称对池 ----------------
def _name_pairs(pool):
    """确定性枚举所有合法名对 (a, b)（a ≠ b 且互不为子串）。"""
    return [(a, b) for a in pool for b in pool
            if a != b and a not in b and b not in a]


# ---------------- 生成 ----------------
def gen_records(seed: int = SEED, n_train: int = N_TRAIN, n_test: int = N_TEST,
                shared_vocab: bool = False, det_reply: bool = False) -> list:
    """确定性生成 n_train+n_test 条**互不相同**的 ctx。

    每条样本 = 三轮：其中一轮引入「我自己的」实体 = 正确实体 a，
    另一轮引入「我朋友的」实体 = 干扰实体 b。**哪个先说由序号奇偶严格决定**
    （偶数条 = 我自己先说，奇数条 = 我朋友先说）⇒ 位置与角色严格正交：

        位置规则「取第一个出现的实体」命中率 = 50%（实测见 [INV] D 行）
        角色规则「取『我自己』那个」命中率 = 100%（= 正确实体定义）

    组合空间 = 族 × 顺序2 × (轮1句面3 + 轮2句面3) × 轮3句面4 × 新信息槽4 × 填充语40 × 名对380
             ≈ 5×2×3×4×4×40×380 = 2,918,400,000 ≫ 7800。
    """
    rng = random.Random(seed)
    pairs_tr = _name_pairs(NAMES_TRAIN)
    # ★--shared-vocab：test 也取 NAMES_TRAIN（同一个 380 名对池，rng 消耗条数完全相同
    #   ⇒ 与 r1 唯一不同的就是名字落在哪份词表里）
    pairs_te = pairs_tr if shared_vocab else _name_pairs(NAMES_TEST)
    combos = [(fi, fi1, fi2, xt, ti, v3i)
              for fi, fam in enumerate(FAMILIES)
              for fi1 in range(len(fam["t1_self_forms"]))
              for fi2 in range(len(fam["t2_self_forms"]))
              for xt in range(len(fam["t3_user_forms"]))
              for ti in range(len(T2_TOPICS))
              for v3i in range(len(fam["v3"]))]
    rng.shuffle(combos)
    need = n_train + n_test
    if len(combos) < need:
        raise SystemExit(f"[D68] 组合空间不足：{len(combos)} < {need}")

    recs = []
    for k, (fi, fi1, fi2, xt, ti, v3i) in enumerate(combos[:need]):
        fam = FAMILIES[fi]
        is_test = k >= n_train
        pairs_all = pairs_te if is_test else pairs_tr
        a, b = pairs_all[rng.randrange(len(pairs_all))]
        assert a != b and a not in b and b not in a
        if shared_vocab:
            assert (a in TRAIN_SET) and (b in TRAIN_SET)                    # 共享词表：两侧同池
        else:
            assert (a in TEST_SET) == is_test and (b in TEST_SET) == is_test   # 词表不跨界
        # ★ 顺序位：严格 50/50，与角色/内容/名对一律无关
        self_first = (k % 2 == 0)
        topic = T2_TOPICS[ti]
        v3 = fam["v3"][v3i]
        tail = R1S_TAILS[rng.randrange(len(R1S_TAILS))]
        # ★--det-reply：这次抽样**照抽但丢弃**（保住与 68b 完全相同的 RNG 消耗序 ⇒ ctx 逐条相同），
        #   句面改由输入可见线索（family + 轮3 省略式类型）唯一确定。
        goldi_rng = rng.randrange(len(fam["t3_sys_forms"]))
        goldi = det_reply_index(fam, xt) if det_reply else goldi_rng
        # ★ 哪一句是"我朋友的"完全跟着顺序走；系统话只许复述**对方（朋友）**那个实体，
        #   不许复述正确实体（否则轮2 写成"您朋友的{正确实体}"= 语义错误，也漏答案）
        if self_first:
            r1u = fam["t1_self_forms"][fi1].format(a=a, item=fam["item"], act2=fam["t2_act"])
            r1s = fam["t1_sys_forms"][rng.randrange(len(fam["t1_sys_forms"]))].format(a=a, item=fam["item"])
            r2u = topic + fam["t2_friend_forms"][fi2].format(other=b, act2=fam["t2_act"])
            r2s = fam["t2_sys_forms"][rng.randrange(len(fam["t2_sys_forms"]))].format(other=b)
        else:
            r1u = fam["t1_friend_forms"][fi1].format(other=b, act2=fam["t1_act"])
            r1s = fam["t1_sys_forms"][rng.randrange(len(fam["t1_sys_forms"]))].format(a=b, item=fam["item"])
            r2u = topic + fam["t2_self_forms"][fi2].format(a=a, item=fam["item"], act2=fam["t2_act"])
            r2s = fam["t2_sys_forms"][rng.randrange(len(fam["t2_sys_forms"]))].format(other=b)
        r1s = r1s + tail
        r3u = fam["t3_user_forms"][xt].format(v3=v3)
        gold = fam["t3_sys_forms"][goldi].format(a=a, v3=v3)
        ctx = render_ctx(r1u, r1s, r2u, r2s, r3u)
        pa, pb = ctx.find(a), ctx.find(b)
        # 只有一条实体是"我自己的"，正确实体恒为 a —— 位置与角色无关，这里只做记账
        assert (pa >= 0) and (pb >= 0) and pa != pb
        ecount = sum(1 for n in NAMES if n in ctx)
        recs.append(dict(
            ctx=ctx, target=gold, gold=gold, ent=a, ent_other=b, role=ROLE_SELF,
            role_other=ROLE_OTHER, self_first=self_first,
            turn3_has_ent=((a in r3u) or (b in r3u)),
            t3=r3u, t1_user=r1u, t1_sys=r1s, t2_user=r2u, t2_sys=r2s,
            family=fam["key"], v3=v3, n_turns=3,
            t3_form_idx=xt, det_reply_surface_idx=goldi, det_reply=bool(det_reply),
            ent_count=ecount,
            ent_pos=pa, ent_other_pos=pb,
            ent_pos_rank=(0 if pa < pb else 1),
            t3_chars=len(r3u),
            v3_in_t3=(v3 in r3u), ent_in_target=(a in gold), other_in_target=(b in gold),
            v3_in_target=(v3 in gold),
        ))
    # 位置桶必须两侧均衡（不变量 D 的构造保证）
    n0 = sum(1 for r in recs if r["ent_pos_rank"] == 0)
    n1 = sum(1 for r in recs if r["ent_pos_rank"] == 1)
    assert n0 + n1 == len(recs) and abs(n0 - n1) <= 2, (n0, n1)
    return recs


# ---------------- 不变量检查（逐条断言式计数） ----------------
def _candidates(text: str, vocab=NAMES):
    return sorted(((text.find(n), n) for n in vocab), key=lambda t: t[0])


def _first_entity(text: str, vocab=NAMES):
    """只见位置的最简规则：取文本里**第一个出现**的实体名。"""
    for p, n in _candidates(text, vocab):
        if p >= 0:
            return n
    return None


MARK_SELF = "我自己的"
MARK_OTHER = "我朋友的"


def _one(cands):
    uniq = sorted(set(cands))
    return uniq[0] if len(uniq) == 1 else None


def _ent_after(text: str, lo: int, vocab=NAMES, width: int = 6) -> str:
    """取 text[lo:] 里**最靠前**的实体名。

    窗口宽 6 字（够 "那个{a}" 这种带指示词的槽），并先把窗口截在
    "也/跟/和/，/。/换行/的" 之前，避免窗口里同时出现两个候选。
    """
    seg = text[lo: lo + width]
    cut = min([k for k in (seg.find(ch) for ch in "也跟和，。\n的") if k >= 0] or [len(seg)])
    return _one([n for n in vocab if n in seg[:cut]])


def _self_ent(ctx: str, vocab=NAMES):
    """取「我自己的」槽位上的实体（与顺序无关）。

    模板里名字可能紧跟标记之后（"我自己的{name}"），也可能在标记之前
    （"我叫{name}，我自己的{item}"）⇒ 两侧窗口都看，但**都必须截在同一句内**，
    否则会漏到别的轮次（实测踩过：窗口跨越轮1 会捞到轮2 的干扰实体）。
    """
    i = ctx.find(MARK_SELF)
    if i < 0:
        return None
    seg = ctx[i + len(MARK_SELF): i + len(MARK_SELF) + 10]
    cut = min([k for k in (seg.find(ch) for ch in "也跟和，。\n的") if k >= 0] or [len(seg)])
    hit = [n for n in vocab if n in seg[:cut]]
    if len(hit) == 1:
        return hit[0]
    line0 = ctx.rfind("\n", 0, i)
    pre = ctx[line0 + 1: i]
    prev = ctx[max(0, line0 - 40): line0]
    pre_hit = [n for n in vocab if n in pre and n not in prev]
    return _one(pre_hit)


def _other_ent(ctx: str, vocab=NAMES):
    """取「我朋友的」槽位上的实体（名字在标记后，或"我朋友用/拿{name}的账号"里名字在前）。"""
    j = ctx.find(MARK_OTHER)
    if j >= 0:
        v = _ent_after(ctx, j + len(MARK_OTHER), vocab)
        if v is not None:
            return v
    j2 = max(ctx.find("我朋友用"), ctx.find("我朋友拿"))
    k = ctx.find("的账号", j2) if j2 >= 0 else -1
    if k >= 0:
        v = _one([n for n in vocab if n in ctx[max(0, k - 8): k]]
                 + [n for n in vocab if n in ctx[k + 3: k + 7]])
        if v is not None:
            return v
    return None


def _role_entity(ctx: str, vocab=NAMES):
    """按**角色线索**取实体（与位置无关）：正确实体 = 「我自己的」那个。

    这就是"读角色词"而不是"读第几个出现"—— 两种顺序（自己先说 / 朋友先说）都能对。
    """
    v = _self_ent(ctx, vocab)
    if v is not None:
        return v
    o = _other_ent(ctx, vocab)
    if o is None:
        return None
    return _one([n for n in vocab if n in ctx and n != o])


def check_invariants(recs, tag="all", vocab=NAMES, det_reply=False):
    """返回 (ok, vio, total_vio)；逐条断言 A/B/C/D，另做词表级静态检查。

    det_reply=True 时额外做 **DET 检查**（句面确定性）：句面必须能从**输入文本**复原并按 DET 规则
    复现 target；同一 (family, 省略式类型) 的句面必须唯一。
    """
    ok = Counter()
    vio = Counter()
    for r in recs:
        a = (not r["turn3_has_ent"]) \
            and (r["ent"] not in r["t3"]) and (r["ent_other"] not in r["t3"]) \
            and (r["ent"] in (r["t1_user"] if r["self_first"] else r["t2_user"])) \
            and (r["ent_other"] in (r["t2_user"] if r["self_first"] else r["t1_user"])) \
            and (r["ent"] in r["target"])
        b = r["ent_in_target"] and (not r["other_in_target"]) and r["v3_in_target"]
        ok["A"] += int(a)
        ok["B"] += int(b)
        vio["A"] += int(not a)
        vio["B"] += int(not b)
        # A 加强：轮1 只该出现正确实体；轮3 里连"部分名"都不许冒头（这里查全名即可）
        vio["A_ent_in_t3"] += int((r["ent"] in r["t3"]) or (r["ent_other"] in r["t3"]))
    # C：ctx 唯一 + ctx→target 是函数
    seen = {}
    dup, conflict = 0, 0
    for r in recs:
        if r["ctx"] in seen:
            dup += 1
            if seen[r["ctx"]] != r["target"]:
                conflict += 1
        seen[r["ctx"]] = r["target"]
    ok["C"] += len(seen)
    vio["C_dup"] += dup
    vio["C_conflict"] += conflict

    # D：位置线索与角色线索正交 —— 位置规则的准确率必须 ≈ 50%（离 100% 越远越好）
    n_first_hit = sum(1 for r in recs if _first_entity(r["ctx"]) == r["ent"])
    n_role_hit = sum(1 for r in recs if _role_entity(r["ctx"]) == r["ent"])
    rank_cnt = Counter(r["ent_pos_rank"] for r in recs)
    ok["D"] += len(recs)
    vio["D_pos_only_hit"] += n_first_hit
    d_pos_only = 100.0 * n_first_hit / len(recs)
    d_role = 100.0 * n_role_hit / len(recs)
    # 静态：任何实体都不许出现在 轮3 句面 / 轮2 填充语 / 轮2 系统句中
    static_vio = 0
    for fam in FAMILIES:
        for form in fam["t3_user_forms"]:
            static_vio += sum(1 for n in NAMES if n in form)
    for ent in NAMES:
        static_vio += sum(1 for t in T2_TOPICS if ent in t)
    # 静态：角色词不许与实体名互为子串；轮2 填充语不许含角色词（否则角色线索被污染）
    for n in NAMES:
        static_vio += int((ROLE_SELF in n) or (ROLE_OTHER in n))
    for t in T2_TOPICS:
        static_vio += int((ROLE_SELF in t) or (ROLE_OTHER in t))
    vio["A_static"] += static_vio

    print(f"[INV:{tag}] A 轮1含正确实体且轮3无任何实体(两个都查)：通过={ok['A']} 违例={vio['A']} "
          f"| 轮3 含实体计数={vio['A_ent_in_t3']} | 静态词表级违例(A_static)={vio['A_static']}",
          flush=True)
    print(f"[INV:{tag}] B 目标含正确实体且不含干扰实体：通过={ok['B']} 违例={vio['B']}", flush=True)
    print(f"[INV:{tag}] C ctx→target 是函数：ctx唯一={ok['C']} 重复={vio['C_dup']} "
          f"同ctx不同target={vio['C_conflict']}（违例）", flush=True)
    print(f"[INV:{tag}] D 判定不只靠位置：正确实体位置桶={dict(rank_cnt)} "
          f"| 位置规则「取第一个出现的实体」EM={d_pos_only:.2f}% | 角色规则「我朋友的」取补 EM={d_role:.2f}% "
          f"（位置规则≈50% ⇒ 位置拿不到高分）", flush=True)
    if det_reply:
        # ★DET 检查（L2-r2）：句面由**输入可见线索**唯一确定，且能从输入文本复原
        n_rule_ok, n_bad_rule, n_bad_recover = 0, 0, 0
        seen = {}
        for r in recs:
            fam = FAM_BY_KEY[r["family"]]
            xt = _t3_form_index(fam, r["t3"])
            if xt is None or xt != r["t3_form_idx"]:
                n_bad_recover += 1
                continue
            idx = det_reply_index(fam, xt)
            seen.setdefault((r["family"], xt), set()).add(idx)
            exp = fam["t3_sys_forms"][idx].format(a=r["ent"], v3=r["v3"])
            if exp == r["target"]:
                n_rule_ok += 1
            else:
                n_bad_rule += 1
        n_non_singleton = sum(1 for v in seen.values() if len(v) != 1)
        vio["DET_bad_rule"] += n_bad_rule
        vio["DET_bad_recover"] += n_bad_recover
        vio["DET_non_singleton"] += n_non_singleton
        print(f"[INV:{tag}] DET 句面确定性（--det-reply）：句面可从**输入文本**复原且按 DET 复现 target"
              f"={n_rule_ok}/{len(recs)}（违例={n_bad_rule}）| 输入复原失败={n_bad_recover} "
              f"| (family,省略式)→句面 非单值={n_non_singleton} | 出现过的 (family,省略式) 组合={len(seen)}"
              f"（应={N_TEMPLATE_FAMILIES}×"
              f"{len(FAMILIES[0]['t3_user_forms'])}={N_TEMPLATE_FAMILIES * len(FAMILIES[0]['t3_user_forms'])}）",
              flush=True)
    total_vio = (vio["A"] + vio["B"] + vio["A_ent_in_t3"] + vio["A_static"]
                 + vio["C_dup"] + vio["C_conflict"])
    if det_reply:
        total_vio += vio["DET_bad_rule"] + vio["DET_bad_recover"] + vio["DET_non_singleton"]
    print(f"[INV:{tag}] 违例合计={total_vio}（不含 D 的命中计数，D 是读数不是违例）⇒ "
          f"{'全部不变量成立' if total_vio == 0 else '★数据无效'}", flush=True)
    return ok, vio, total_vio


# ---------------- 指纹（口径与 stages/65::data_fp 逐字一致） ----------------
def data_fp(recs) -> str:
    """对每条 (len(p), p..., len(t), t...) 求 sha256[:16]。recs 需带 'p'/'t' 两个 id 序列。"""
    h = hashlib.sha256()
    for r in recs:
        for seq in (r["p"], r["t"]):
            h.update(struct.pack("<Q", len(seq)))
            if seq:
                h.update(struct.pack(f"<{len(seq)}i", *seq))
    return h.hexdigest()[:16]


# ---------------- 地板（Q3）：只用输入可见信息 + train 统计 ----------------
def _em(gold: str, pred: str) -> int:
    return int((gold or "").strip() == (pred or "").strip())


def _t3_parse(t3: str):
    """只用轮3 文本 + 已知族模板 ⇒ (族 key, 回复句面 index, v3)；识别不出返回全 None。

    ★ 回复句面 index 是**输入里绝不可见**的量（同一族里有多条回复句面）⇒ 任何
      "只看输入"的规则都只能靠 train 统计去猜，这正是地板该有的贴地。
    """
    for fam in FAMILIES:
        for form in fam["t3_user_forms"]:
            head, sep, tail = form.partition("{v3}")
            if not sep:
                if t3 == form:
                    return fam["key"], None, None
                continue
            if t3.startswith(head) and (t3.endswith(tail) if tail else True):
                mid = t3[len(head): len(t3) - len(tail)] if tail else t3[len(head):]
                if mid in fam["v3"]:
                    return fam["key"], None, mid
    return None, None, None


def _bucket_specs():
    """⑤ 计数类特征（口径沿用 r0：轮3 长度 / 实体个数 / 轮数 / 实体位置桶）。"""
    return (
        ("⑤a轮3长度桶(5tok一桶)", lambda r: r.get("t3_tok", r["t3_chars"]) // 5),
        ("⑤b实体个数桶", lambda r: r["ent_count"]),
        ("⑤c轮数桶", lambda r: r["n_turns"]),
        ("⑤d实体位置桶(10字符一桶)", lambda r: r["ent_pos"] // 10),
    )


def _stats(train):
    st = {
        "maj_target": Counter(r["target"] for r in train).most_common(1)[0][0],
        "maj_ent": Counter(r["ent"] for r in train).most_common(1)[0][0],
        "bucket": {},
        "maj_sys": {},
    }
    # 回复句面：输入不可见，规则只能按 train 多数句面猜（各族独立）
    for fam in FAMILIES:
        idxs = []
        for r in train:
            if r["family"] == fam["key"]:
                idxs.append(fam["t3_sys_forms"].index(r["target"])
                            if r["target"] in fam["t3_sys_forms"] else 0)
        st["maj_sys"][fam["key"]] = Counter(idxs).most_common(1)[0][0] if idxs else 0
    for name, keyfn in _bucket_specs():
        per = {}
        for r in train:
            per.setdefault(keyfn(r), []).append(r["target"])
        st["bucket"][name] = {b: Counter(v).most_common(1)[0][0] for b, v in per.items()}
    return st


def _rule_preds():
    def fill_only_t3(r, s):
        key, _si, v3 = _t3_parse(r["t3"])
        if key is None:
            s["miss"] += 1
            return ""
        fam = FAM_BY_KEY[key]
        # 实体位只能填「train 里最常见的实体」（它必不出现于 test 名池）；
        # 回复句面用 train 里该族的多数句面（输入看不见它）。
        return fam["t3_sys_forms"][s["maj_sys"][key]].format(
            a=s["maj_ent"], my=ROLE_SELF, v3=v3)

    rules = [
        ("①复制轮3用户话", lambda r, s: r["t3"]),
        ("②复制轮1系统话", lambda r, s: r["t1_sys"]),
        ("③最常见回复(train多数类)", lambda r, s: s["maj_target"]),
        ("④模板填空只用轮3(实体位填train多数实体)", fill_only_t3),
    ]
    for name, keyfn in _bucket_specs():
        def mk(name=name, keyfn=keyfn):
            def f(r, s):
                return s["bucket"][name].get(keyfn(r), "")   # 没见过的桶 ⇒ 不预测
            return f
        rules.append((name, mk()))
    # ⑥/⑦：两种**只用实体选择**的规则（其余槽位仍按 ④ 的模板填空补全）
    rules.append(("⑥只用位置(取第一个出现的实体)+模板填空", _mk_pick(True)))
    rules.append(("⑦只用角色词(「我朋友的」邻域的实体取补)+模板填空", _mk_pick(False)))
    return rules


def _mk_pick(by_pos: bool):
    """⑥/⑦：实体选择换掉（位置 / 角色词），其余槽位仍按 ④ 的模板填空补全。"""
    def f(r, s):
        key, _si, v3 = _t3_parse(r["t3"])
        if key is None:
            s["miss"] += 1
            return ""
        fam = FAM_BY_KEY[key]
        cand = _first_entity(r["ctx"]) if by_pos else _role_entity(r["ctx"])
        if cand is None:
            s["miss"] += 1
            cand = s["maj_ent"]
        return fam["t3_sys_forms"][s["maj_sys"][key]].format(
            a=cand, my=ROLE_SELF, v3=v3)
    return f


def compute_floors(train, test, tag="test", show_train=True):
    """在 test 上逐条算 EM；返回 (per-rule EM, max_naive 名, max_naive 值, 门槛)。"""
    st = _stats(train)
    rules = _rule_preds()

    def eval_rules(rs):
        rows = {}
        for name, fn in rules:
            per = dict(st)              # 每条规则独立的 miss 计数；st['bucket'] 只读共享
            per["miss"] = 0
            hit = sum(_em(r["target"], fn(r, per)) for r in rs)
            rows[name] = (100.0 * hit / len(rs), per["miss"])
        return rows

    rows_test = eval_rules(test)
    out = {k: v[0] for k, v in rows_test.items()}
    for name, (em, miss) in rows_test.items():
        print(f"[FLOOR] {tag} {name} EM={em:.2f}% (n={len(test)})"
              + (f" | 识别不出族={miss}" if miss else ""), flush=True)
    mx_name = max(out, key=out.get)
    mx = out[mx_name]
    n = len(test)
    se2 = 2.0 * (mx / 100.0 * (1 - mx / 100.0) / n) ** 0.5 * 100.0
    thr = mx + se2
    print(f"[FLOOR] {tag} max_naive={mx:.2f}% 来自={mx_name} | 2SE={se2:.2f}pp ⇒ "
          f"Q1 门槛=max_naive+2SE={thr:.2f}%", flush=True)
    if show_train:
        rows_train = eval_rules(train)
        tr = {k: v[0] for k, v in rows_train.items()}
        for name, (em, _miss) in rows_train.items():
            print(f"[FLOOR] train {name} EM={em:.2f}% (n={len(train)})", flush=True)
        tm_name = max(tr, key=tr.get)
        print(f"[FLOOR] train max_naive={tr[tm_name]:.2f}% 来自={tm_name} ⇒ "
              + ("★数据无效（朴素规则在 train 上≥90%）" if tr[tm_name] >= 90.0
                 else "有效（无朴素规则在 train 上≥90%）"), flush=True)
    return out, mx_name, mx, thr


# ---------------- [CEIL]（L2-r2）：本档"完美实体+完美 v3"下的 EM 上限（实算） ----------------
def compute_ceiling(recs, tag="test"):
    """★实算本档 EM 上限，不写断言。

    口径（写死）：**完美实体 + 完美 v3 + 由输入可见线索唯一确定的句面**。
    句面由 DET 规则从**输入文本**复原（_t3_form_index → det_reply_index），
    两个槽位喂 oracle（r["ent"] / r["v3"]）⇒ 若 DET 与数据构造一致，上限应 = 100.00%。

    第二行是**诚实分解**：{v3} 槽只在每族第 1 种轮3 省略式里可见 ⇒ 其余记录的 v3 输入不可见，
    一个"完美实体 + 完美句面"的模型只能按各族 |v3| 池均匀蒙 ⇒ 报出该期望可达上限。
    """
    n = len(recs)
    hit = miss_recover = vis = 0
    exp_num = 0.0
    for r in recs:
        fam = FAM_BY_KEY[r["family"]]
        xt = _t3_form_index(fam, r["t3"])
        if xt is None:
            miss_recover += 1
            continue
        pred = fam["t3_sys_forms"][det_reply_index(fam, xt)].format(a=r["ent"], v3=r["v3"])
        hit += _em(r["target"], pred)
        if r["v3_in_t3"]:
            vis += 1
            exp_num += 1.0
        else:
            exp_num += 1.0 / len(fam["v3"])
    ceil = 100.0 * hit / n
    visp = 100.0 * vis / n
    exp = 100.0 * exp_num / n
    print(f"[CEIL] {tag} n={n} 口径=完美实体+完美v3+由输入可见线索确定的句面"
          f"（DET(family, 轮3省略式)）⇒ **EM 上限={ceil:.2f}%**（实算命中={hit}/{n}；"
          f"句面从输入文本复原失败={miss_recover}）", flush=True)
    print(f"[CEIL] {tag} 分解：{{v3}} 在输入里可见={visp:.2f}%（n={vis}，可完全学）| 不可见={100.0 - visp:.2f}%"
          f"（n={n - vis}）⇒ 完美实体+完美句面、v3 只能按各族 |v3| 池均匀蒙时"
          f"**期望可达上限={exp:.2f}%**（各族 |v3| 池="
          + ",".join(f"{f['key']}:{len(f['v3'])}" for f in FAMILIES) + "）", flush=True)
    return ceil, visp, exp


def _cmp_68b(train, test):
    """单变量取证：68d 应只有 target 变了，ctx 与 68b 逐条相同（RNG 消耗序一致）。"""
    parts = []
    for split, rs in (("train", train), ("test", test)):
        fp = os.path.join(LOG_DIR, f"68b_dialog_{split}.jsonl")
        if not os.path.exists(fp):
            parts.append(f"{split}=68b缺⇒跳过")
            continue
        with open(fp, encoding="utf-8") as f:
            old = [json.loads(ln) for ln in f if ln.strip()]
        n = min(len(old), len(rs))
        same_ctx = sum(1 for i in range(n) if old[i]["ctx"] == rs[i]["ctx"])
        same_tgt = sum(1 for i in range(n) if old[i]["target"] == rs[i]["target"])
        parts.append(f"{split}: ctx 相同={same_ctx}/{n} target 相同={same_tgt}/{n}"
                     f"（⇒ target 变了 {n - same_tgt}/{n}）")
    print(f"[D68D-CMP] 与 68b 单变量取证：{' | '.join(parts)}", flush=True)


# ---------------- 入口 ----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="S68 L2-r1 跨轮消解式对话数据生成器")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--shared-vocab", action="store_true",
                    help="L2-r1b 唯一变量：test 人名取自与 train 同一份词表；"
                         "落盘改到 logs/68b_dialog_*.jsonl 并打印 [D68B]")
    ap.add_argument("--det-reply", action="store_true",
                    help="L2-r2 唯一变量：回复句面由输入可见线索（family + 轮3 省略式类型）唯一确定"
                         "（规则 DET）；隐含 --shared-vocab；落盘 logs/68d_dialog_*.jsonl，"
                         "打印 [D68D]/[CEIL]")
    a = ap.parse_args(argv)

    shared = bool(a.shared_vocab or a.det_reply)
    recs = gen_records(seed=a.seed, n_train=a.n_train, n_test=a.n_test,
                       shared_vocab=shared, det_reply=a.det_reply)
    train, test = recs[:a.n_train], recs[a.n_train:]
    combos_n = (N_TEMPLATE_FAMILIES * N_FAMILY_FORMS * N_T3_FORMS * N_V3 * len(T2_TOPICS)
                * len(NAMES_TRAIN) * (len(NAMES_TRAIN) - 1))
    print(f"[SPLIT] 固定切分：先 shuffle 组合再切前 {a.n_train}+{a.n_test} 条；"
          f"组合空间={combos_n}；实体词表 train={len(NAMES_TRAIN)} test={len(NAMES_TEST)} "
          f"（{'共享同一份 ⇒ NAMES_TRAIN' if shared else '互不相交'}）", flush=True)

    ok, vio, total_vio = check_invariants(recs, tag="all", det_reply=a.det_reply)

    # ---- tokenizer：逐条编码 + 指纹 + 可加性 + 格式核对 ----
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK_PATH)
    V = tok.get_vocab_size()
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids        # noqa: E731
    dec = lambda ids: tok.decode([int(i) for i in ids], skip_special_tokens=False)  # noqa: E731
    PAD_ID = tok.token_to_id("<pad>")
    EOS_ID = tok.token_to_id("<eos>")
    assert enc("<resp>") == [140] and enc(EOS) == [EOS_ID], (enc("<resp>"), enc(EOS))

    from training.dialogue_stream import DialogueStream
    n_ctx_fmt_mismatch = n_additive_bad = n_rt_bad = 0
    for r in recs:
        r["p"] = enc(r["ctx"])
        r["t"] = enc(r["target"] + EOS)
        r["t3_tok"] = len(enc(r["t3"]))
        if dec(r["p"]) != r["ctx"] or dec(r["t"]) != r["target"] + EOS:
            n_rt_bad += 1
        if enc(r["ctx"] + r["target"] + EOS) != r["p"] + r["t"]:
            n_additive_bad += 1
        st = DialogueStream(enc, window=10 ** 9)
        st.append("用户", r["t1_user"]); st.commit(r["t1_sys"])
        st.append("用户", r["t2_user"]); st.commit(r["t2_sys"])
        st.append("用户", r["t3"])
        if st.prompt() != r["ctx"]:
            n_ctx_fmt_mismatch += 1
    assert max(len(r["p"]) for r in recs) + max(len(r["t"]) for r in recs) <= 512, "ctx+target 超 512"
    train, test = recs[:a.n_train], recs[a.n_train:]
    tr_fp, te_fp = data_fp(train), data_fp(test)
    print(f"[FMT] 与 DialogueStream.prompt() 逐字不一致={n_ctx_fmt_mismatch} | "
          f"enc 可加性破坏={n_additive_bad} | 编解码往返坏={n_rt_bad} | "
          f"V={V} PAD={PAD_ID} EOS={EOS_ID} | ctx tok 中位="
          f"{sorted(len(r['p']) for r in recs)[len(recs)//2]} 最大={max(len(r['p']) for r in recs)} "
          f"| target tok 中位={sorted(len(r['t']) for r in recs)[len(recs)//2]} "
          f"最大={max(len(r['t']) for r in recs)}", flush=True)

    # ---- 落盘（p/t 只在内存里用于指纹与长度统计，不写进 jsonl：69 会用同一 tokenizer 重算） ----
    out_prefix = "68d" if a.det_reply else ("68b" if a.shared_vocab else "68")
    for split, rs in (("train", train), ("test", test)):
        fp = os.path.join(LOG_DIR, f"{out_prefix}_dialog_{split}.jsonl")
        with open(fp, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps({k: v for k, v in r.items() if k not in ("p", "t")},
                                   ensure_ascii=False) + "\n")
        print(f"[OUT] {split} → {fp} ({len(rs)} 条)", flush=True)

    print(f"[D68] n_train={len(train)} n_test={len(test)} 族={N_TEMPLATE_FAMILIES} "
          f"族句面(轮1)={N_FAMILY_FORMS} 回复模板={N_REPLY_TEMPLATES} 新信息槽={N_V3} "
          f"实体词表={len(NAMES_TRAIN) if shared else len(NAMES)}"
          f"(train {len(NAMES_TRAIN)}/"
          f"test {len(NAMES_TRAIN) if shared else len(NAMES_TEST)}"
          f"{' 共享' if shared else ''}) "
          f"| train_fp={tr_fp} test_fp={te_fp}", flush=True)
    print(f"[TPL] 族=" + " ".join(
        f"{f['key']}(我自己×{len(f['t1_self_forms'])}/我朋友×{len(f['t1_friend_forms'])}/"
        f"轮3×{len(f['t3_user_forms'])}/回复×{len(f['t3_sys_forms'])}/{len(f['v3'])}槽)"
        for f in FAMILIES), flush=True)
    print(f"[CONST] 轮2 填充语={len(T2_TOPICS)} 轮2 干扰句式={len(T2U_VARIANTS)} "
          f"轮2 系统句式={len(T2S_VARIANTS)} seed={a.seed} "
          f"| 唯一 ctx={len(set(r['ctx'] for r in recs))}/{len(recs)} | "
          f"turn3_has_ent True 计数={sum(r['turn3_has_ent'] for r in recs)}（必须 0）| "
          f"角色词=正确「{ROLE_SELF}」/干扰「{ROLE_OTHER}」", flush=True)
    print(f"[ENTCNT] ctx 中实体个数分布="
          f"{dict(Counter(r['ent_count'] for r in recs))}（r1 恒=2）| "
          f"正确实体位置桶="
          f"{dict(Counter(r['ent_pos_rank'] for r in recs))}（0=先出现 1=后出现，须两侧均衡）",
          flush=True)

    # ---- 地板 ----
    _floors, mx_name, mx, thr = compute_floors(train, test, tag="test", show_train=True)
    if a.shared_vocab:
        # ★L2-r1b 的臂标识行：规模 + 不变量计数 + 本臂自己的指纹 + 地板（一行读完）
        print(f"[D68B] 共享词表臂（--shared-vocab）：n_train={len(train)} n_test={len(test)} "
              f"族={N_TEMPLATE_FAMILIES} 回复模板={N_REPLY_TEMPLATES} 新信息槽={N_V3} "
              f"轮2填充语={len(T2_TOPICS)} | 实体词表={len(NAMES_TRAIN)}（train/test 共享同一份）"
              f" | train_fp={tr_fp} test_fp={te_fp}"
              f" | 不变量违例合计={total_vio}(A={vio['A']} B={vio['B']} "
              f"A_ent_in_t3={vio['A_ent_in_t3']} A_static={vio['A_static']} "
              f"C_dup={vio['C_dup']} C_conflict={vio['C_conflict']})"
              f" | 地板 max_naive={mx:.2f}%({mx_name}) 2SE={thr - mx:.2f}pp 门槛={thr:.2f}%",
              flush=True)
    if a.det_reply:
        # ★L2-r2：[CEIL] 实算上限 + 臂标识行 + 与 68b 的单变量取证
        compute_ceiling(test, tag="test")
        compute_ceiling(train, tag="train")
        _cmp_68b(train, test)
        print(f"[D68D] 确定性回复臂（--det-reply，隐含共享词表）：n_train={len(train)} n_test={len(test)} "
              f"族={N_TEMPLATE_FAMILIES} 回复模板={N_REPLY_TEMPLATES} 新信息槽={N_V3} "
              f"轮2填充语={len(T2_TOPICS)} | 实体词表={len(NAMES_TRAIN)}（train/test 共享同一份）"
              f" | 句面规则 DET(family,轮3省略式 index)=该 index（族内两列等长 ⇒ 一一对应）"
              f" | train_fp={tr_fp} test_fp={te_fp}"
              f" | 不变量违例合计={total_vio}(A={vio['A']} B={vio['B']} "
              f"A_ent_in_t3={vio['A_ent_in_t3']} A_static={vio['A_static']} "
              f"C_dup={vio['C_dup']} C_conflict={vio['C_conflict']} "
              f"DET_bad_rule={vio['DET_bad_rule']} DET_bad_recover={vio['DET_bad_recover']} "
              f"DET_non_singleton={vio['DET_non_singleton']})"
              f" | 地板 max_naive={mx:.2f}%({mx_name}) 2SE={thr - mx:.2f}pp 门槛={thr:.2f}%",
              flush=True)
    if total_vio:
        print(f"[D68-VERDICT] ★不变量违例={total_vio} ⇒ 数据无效，停。", flush=True)
        return 4
    print("[D68-VERDICT] 不变量全过；地板特征与指纹已落盘。", flush=True)
    print("[DONE] exit=0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
