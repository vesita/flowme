"""④ 渲染（确定性纯函数）+ ⑤ 解引用侧检查与引用映射留存。

`src/dtseek/tasks/render.py` **本实验开始时不存在**（已在全仓 find 过），因此在这里实现
最小版；若该文件事后出现，报告须注明本实验未使用它（见 report）。

  - `render_structure(struct, skel) -> str`：结构指令 → 文本。纯函数：同输入逐字相同，
    不读时钟、不用随机、不依赖 dict 插入顺序之外的任何状态。
  - `deref_verify(text_out, struct, skel, src_text, valid_index)`：**解引用侧**的独立复核
    （dev-notes/16 §7.9 的两道检查里的第 2 道）。它**重新走一遍骨架模板**去核对输出，
    而不是相信渲染器自己报的区间 —— 于是"渲染时偷偷加内容"会被当场抓住。

两道检查的分工：
  结构侧 = `predicates.structure_problems`（挡住结构里的无据命题）
  解引用侧 = 本文件（挡住渲染里偷偷加的内容）
"""
from __future__ import annotations

from skeletons import Skeleton


def render_structure(struct: dict, skel: Skeleton) -> str:
    """结构指令 → 文本。缺块直接抛（调用方 fail-closed 转 reject）。"""
    assign = struct["assign"]
    refs = struct["refs"]
    out: list[str] = []
    for kind, token in skel.lit_tokens():
        if kind == "lit":
            out.append(token)
        else:
            out.append(refs[assign[token]]["surface"])
    return "".join(out)


def deref_verify(text_out: str, struct: dict, skel: Skeleton, src_text: str,
                 valid_index: dict[str, dict]) -> tuple[list[str], list[dict]]:
    """解引用侧：文本每个内容单元必须映射到结构单元，再回到原始 span。

    返回 `(problems, mapping)`；`problems` 非空 = 拒答（不产出半句）。
    mapping 就是**留存的引用映射**：`片段 ↦ 结构 ref id ↦ 原始 span`。
    """
    probs: list[str] = []
    assign = struct.get("assign") or {}
    refs = struct.get("refs") or {}
    mapping: list[dict] = []

    i = 0
    n = len(text_out)
    for kind, token in skel.lit_tokens():
        if kind == "lit":
            if not text_out.startswith(token, i):
                probs.append("literal_mismatch")
                break
            i += len(token)
        else:
            ref = refs.get(assign.get(token, ""), None)
            if ref is None:
                probs.append("schema_missing_slot")
                break
            surf = ref["surface"]
            if not text_out.startswith(surf, i):
                probs.append("ref_render_mismatch")   # 丢块 / 改面 都在这里
                break
            mapping.append({
                "slot": token,
                "ref": ref.get("cid"),
                "frag": surf,
                "pos": i,
                "span": [ref["s0"], ref["e0"]],
                "card": ref.get("card"),
                "class": ref.get("class_name"),
                "type": ref.get("type"),
                "theme": ref.get("theme"),
            })
            i += len(surf)
    else:
        # 正常走完模板：剩下的字符 = 渲染器自己塞的（注入）
        if i != n:
            probs.append("unmapped_output")

    if probs:
        return probs, mapping

    # 映射集合必须与被填槽位集合**完全相等**（双向）
    if {m["ref"] for m in mapping} != set(assign.values()):
        probs.append("ref_set_mismatch")

    # 每条映射：结构 ref ↦ 原始 span 逐字 + 必须属于"被筛为有效"的候选（G2/G3 链）
    for m in mapping:
        ref = refs.get(m["ref"])
        s0, e0 = m["span"]
        if ref is None or (ref["s0"], ref["e0"]) != (s0, e0):
            probs.append("ref_span_mismatch")
            break
        if not (0 <= s0 < e0 <= len(src_text)) or src_text[s0:e0] != m["frag"]:
            probs.append("span_not_verbatim")
            break
        if valid_index.get(m["ref"]) is None:
            probs.append("not_in_valid_set")
            break

    return probs, mapping
