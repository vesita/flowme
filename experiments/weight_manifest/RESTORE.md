# RESTORE —— 权重被误删后，怎么从清单判断「还有没有副本 / 能不能再生 / 找谁」

> 适用：`experiments/weight_manifest/` 清单入库之后的任何一次 `.pt` 丢失。
> 清单**不能找回文件**（它只是校验和与线索）；它的作用是把「还能不能救」变成三个有答案的问题。
> 所有命令都在仓库根 `DTSeek/` 下执行，`sha256sum -c` 依赖清单里的相对路径。

## 第 0 步：确认丢了什么（10 秒）

```bash
sha256sum -c experiments/weight_manifest/manifest.sha256 2>&1 | grep -v '成功' | head -50
```

- 全部 `成功`（exit 0）→ 没丢，别往下走。
- 出现 `失败` / `No such file` → 拿到的路径就是丢失项。
- **内容变了但文件还在**（校验失败 ≠ 不见了）：它被改写过，按「被改写」处理 ——
  这比删除更危险，因为路径还在、引用方不会报错。

```bash
# 单个文件的内容是不是还是清单里那一份
grep '<相对路径>$' experiments/weight_manifest/manifest.sha256   # 先查清单里的应有 sha
sha256sum <相对路径>                                              # 再算现值
```

## 决策树

```
丢失的 .pt
├── Q1 还有物理副本吗（全仓同 sha256）
│     命令：grep '^<该文件的 sha256>' experiments/weight_manifest/manifest.sha256
│     ├── 命中 ≥2 条路径 → cp 过来即可，cp 后重跑 sha256sum -c 全绿才算恢复完成
│     └── 只有 1 条（= 真单副本，全仓 246/262 属此类）→ 走 Q2
├── Q2 git 里有吗
│     命令：git ls-files '*.pt' ; git log --all --oneline -- '<path>'
│     └── 实测恒为 0（.gitignore:15-16 忽略 checkpoints/ 与 *.pt，全仓 .pt 从未入库）
│         ⇒ **git 不是恢复路径，任何 "git checkout -- <path>" 都是空操作** → 走 Q3
├── Q3 能不能重新生成
│     查 manifest.json 该条的来源线索 + analysis.json 的 regen 判据
│     ├── 路径在 cache/ 下        → 该实验的数据准备/编码脚本重跑即得（31 个，重跑后 sha 必须比对清单）
│     ├── 同实验目录有产出脚本命中（analysis.regen_evidence.script，16 个）
│     │     → 可重训；但**不保证逐位复现**（数据版本 / seed / 设备都可能不同）
│     │     ⇒ 判据是「能不能用」而不是「sha 还是不是那个」：若下游 D1 校验要求逐位一致，先问 Q4
│     ├── 卡类（train_args.base / split_from 指向别的核）
│     │     → 核还在就能重拆：读 manifest.json 里 split_from 指向的文件是否在场，
│     │       在场则按 save_card 口径重拆；核也不在 → 走 Q4
│     └── 同实验目录无任何脚本命中（regen_evidence.no_script_hit，215 个）
│           ⇒ **没有再生证据**（不等于绝对不能再生，只是仓库里找不到产出方）→ 走 Q4
└── Q4 找谁 / 外部备份
      ├── 按 manifest.json 该条的 referenced_by 找引用它的实验目录 ——
      │   有过实跑记录的地方最可能留了副本（跑过就会写 results/*.json）
      ├── 引用面 ≥3 个模块的枢纽权重（见 MANIFEST.md 单副本风险 B 类）→ 找 Lead 统一调度
      ├── 全仓唯一 + 无脚本产出 + 无外部备份 ⇒ 记为**不可恢复损失**，
      │   照 additivity 案例写进对应 REPORT 的遗留段（14 个 ckpt 即此路不通，已发生一次）
      └── 外部备份（本机备份盘 / 异地）：清单本身可入库，能当**异地备份的核对索引**
```

## 恢复后的验收（必须做，否则等于没恢复）

```bash
sha256sum -c experiments/weight_manifest/manifest.sha256     # M2：全绿
uv run python - <<'PY'                                        # 卡类还要验基座（D1）
import json,torch,hashlib,pathlib
m=json.load(open('experiments/weight_manifest/manifest.json'))
e=next(x for x in m['entries'] if x['path']=='<恢复的路径>')
ck=torch.load(e['path'],map_location='cpu',weights_only=False)
if e['has_doc_encoder']:
    h=hashlib.sha256()
    for k in ck['doc_encoder']:
        h.update(ck['doc_encoder'][k].detach().cpu().contiguous().numpy().tobytes())
    print('doc_sha ok:', h.hexdigest()==e['doc_sha256'])
print('来源线索:', ck.get('train_args'))
PY
```

- **文件 sha 对上** ⇒ 恢复的是同一份文件。
- **doc_sha 对上且 == `dc27db5337d16f18…`** ⇒ 基座与线上一致（D1 通过）。
- **doc_sha 对上但 ≠ 线上** ⇒ 文件恢复成功，但它是 D1 命中卡（清单里 58 条之一），
  按 `dev-notes/21` §15：**取卡前必须验基座**，用前先逐张量对齐。

## 一份不能忘的事实（实测）

- `.pt` 全仓 **0 个被 git 跟踪**、历史 **0 命中** ⇒ 删除 = 立即不可恢复。
- 本清单之前，只有 **7** 个 `.pt` 的文件级 sha256 留在仓库文本里（另 2 个只有 doc_sha）；
  其余 **253** 个是「盘上唯一 + 无任何校验和记录」。`experiments/additivity/` 的 14 个 ckpt 就是这条路的代价。
- 只有 **8 组共 16 个文件**在全仓有第二份（见 `summary.json: dup_groups`），
  且全部集中在 `bag_modules` 与 `free_rule_floor` 两处 —— **其余 246 个文件丢了就是没了**。
