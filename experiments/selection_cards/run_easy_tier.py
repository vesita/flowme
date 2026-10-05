"""易档可学性诊断：不改源码，用猴子补丁让 reply_pick 只用易档负例（tiers=(1,)）。

判别目的：
  易档能学会 ⇒ 瓶颈在「高难负例」（任务设计问题）；
  易档也学不会 ⇒ 瓶颈在「基座不具备比较候选的能力」（能力跃迁问题）。
"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import dtseek.tasks.builtin.reply_pick as rp  # noqa: E402

_orig = rp.build_reply_pick_dataset


def _easy(*args, **kwargs):
    """把 tiers 固定成 (1,)，其余参数透传（调用方用关键字传参）。"""
    kwargs["tiers"] = (1,)
    return _orig(*args, **kwargs)


rp.build_reply_pick_dataset = _easy

sys.argv = ["train_task_card.py", "--card", "reply_pick", "--epochs", "12", "--seed", "42",
            "--out", "experiments/selection_cards/reply_pick_easy.pt"]
runpy.run_path(os.path.join(ROOT, "training", "train_task_card.py"), run_name="__main__")
