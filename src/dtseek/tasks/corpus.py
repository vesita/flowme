"""对话语料入口：**唯一真相源** + fail-closed。

为什么需要这个模块（一次真实故障）：
nanoSeek 把清洗后的语料从 `data/chinese/*.txt` 归档进了子目录，而 DTSeek 五张卡
各自硬编码 `data/chinese/*dialogue.txt`，于是 glob **命中 0 个文件**且没有任何报错。
数据集静默退化成"纯合成数据"——真实语料腿整条消失，指标虚高、模板痕迹变重。
这与 `dev-notes/04` §3 记的配比 bug 是同一类失效：静态读代码看不出来。

所以这里做两件事：
1. 语料路径收敛到一处（改一个地方就够，不再有五份拷贝）；
2. 命中为空**直接抛错**，宁可训练起不来，也不要拿退化数据训出一个好看的假指标。
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

#: 语料根目录。默认指向 nanoSeek 清洗后的对话语料（同机相对路径）。
#: 换机器或语料搬迁时设 `DTSEEK_CORPUS_DIR` 覆盖，不要回来改这个常量。
CORPUS_DIR = Path(
    os.environ.get(
        "DTSEEK_CORPUS_DIR",
        "/home/vesita/coding/my/nanoSeek/data/chinese/clean_v3",
    )
)

#: 只取对话类语料；同目录下的长篇小说 / 百科 / 诗歌不参与切片任务。
CORPUS_PATTERN = "*dialogue.txt"

#: 供按 glob 字符串取语料的调用方使用（person / relation 的挖掘函数接收入参）。
CORPUS_GLOB = str(CORPUS_DIR / CORPUS_PATTERN)


def resolve_corpus_files(corpus_glob: str | None = None) -> list[str]:
    """返回语料文件列表；**一个都没命中就抛 FileNotFoundError**。

    `corpus_glob` 给测试与特殊场景覆盖用，默认走 :data:`CORPUS_GLOB`。
    """
    pattern = corpus_glob or CORPUS_GLOB
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"对话语料一个文件都没命中：{pattern}\n"
            f"  这会让数据集静默退化成纯合成数据（真实语料腿消失），故 fail-closed。\n"
            f"  语料在别处时设环境变量 DTSEEK_CORPUS_DIR 指向它的目录。"
        )
    return files
