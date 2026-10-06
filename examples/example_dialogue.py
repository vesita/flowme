"""DTSeek 单轮对话演示：**一句话进 → 一句话出**（对比 example_multitask.py 只标切片）。

四段全部走显式调度（dev-notes/16 §3）：调用方声明 type → dispatch 规则策略出计划 →
只跑计划里的卡 → 模板表 + 输入里的锚点片段拼回复。回复的**内容词全部来自输入**。

    uv run python examples/example_dialogue.py --type plain "他不是好人"
    uv run python examples/example_dialogue.py --type plain      # 交互 REPL
    uv run python examples/example_dialogue.py                  # 不声明 type ⇒ 拒答
"""
import argparse
import time

from dtseek.tasks.dialogue import DEFAULT_ATTACH, respond

from dtseek.tasks.engine import DEFAULT_BASE, MultiTaskEngine

RESET = "\033[0m"
BOLD = "\033[1m"


def show(engine: MultiTaskEngine, line: str, type_: str | None):
    """跑一轮对话并打印回复、计划、命中卡、evidence 片段与耗时。"""
    t0 = time.perf_counter()
    rec = respond(engine, line, type_=type_)
    ms = (time.perf_counter() - t0) * 1000

    print("\n" + "=" * 68)
    print(f"输入: {line}")
    print(f"type: {rec['type']}   终止: {rec['terminal']}   耗时: {ms:.1f} ms")
    print("-" * 68)
    print(f"{BOLD}[{rec['kind']}]{RESET} {rec['text']}")
    print("-" * 68)
    print(f"命中卡: {', '.join(rec['cards_run']) or '(未调卡)'}")
    for ev in rec["evidence"]:
        s, e = ev["span"]
        print(f"  evidence: {ev['card']} [{s},{e}] {ev['class_name']}  "
              f"片段『{line[s:e + 1]}』")
    for i, step in enumerate(rec["plan"], 1):
        print(f"  plan[{i}]: {step}")
    if rec["reason"]:
        print(f"  reason: {rec['reason']}")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="DTSeek 单轮对话演示（模板 + 抽取片段）")
    ap.add_argument("text", nargs="?", help="一句话；留空进入交互模式")
    ap.add_argument("--type", default=None, dest="declared_type",
                    help="调用方声明的输入类型（本轮只支持 plain）；不声明 ⇒ 拒答")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--attach", nargs="*", default=None,
                    help="额外挂载的卡产物；默认挂 negation（对话必需位要用）")
    args = ap.parse_args(argv)

    engine = MultiTaskEngine(base_path=args.base)
    for path in (DEFAULT_ATTACH if args.attach is None else args.attach):
        engine.attach(path)

    if args.text:
        show(engine, args.text, args.declared_type)
        return

    print("\n" + "=" * 68)
    print("  DTSeek 单轮对话（type 由你声明，不声明就拒答）")
    print(f"  已挂卡: {', '.join(engine.attached)}")
    print("  输入 exit 退出；/type <t> 换声明的类型")
    print("=" * 68)
    declared = args.declared_type
    print(f"  当前 type: {declared or '(未声明 ⇒ 拒答)'}")
    while True:
        try:
            line = input("\nDTSeek > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            break
        if line.startswith("/type"):
            declared = line.split(maxsplit=1)[1] if len(line.split()) > 1 else None
            print(f"  当前 type: {declared or '(未声明 ⇒ 拒答)'}")
            continue
        show(engine, line, declared)


if __name__ == "__main__":
    main()
