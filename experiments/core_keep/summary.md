# core_keep · P3 与自检汇总

| arm | seed | 可训参数 | 核可训 | s/step | 峰值MB | kd_first | kd_mean | kd_share | kd_share(末epoch) | 步数 |
|---|---|---|---|---|---|---|---|---|---|---|
| F | 42 | 4,840,107 | 1,688,460 | 0.2776 | 2365 | 0.0000 | 0.0000 | 0.00% | 0.00% | 1344 |
| J0 | 42 | 4,840,107 | 1,688,460 | 0.2752 | 2365 | 0.0000 | 0.0000 | 0.00% | 0.00% | 1344 |
| J0 | 43 | 4,840,107 | 1,688,460 | 0.2776 | 2373 | 0.0000 | 0.0000 | 0.00% | 0.00% | 1344 |
| J1 | 42 | 4,840,107 | 1,688,460 | 0.3585 | 2396 | 12.2349 | 20.7630 | 43.74% | 28.49% | 1344 |
| J1 | 43 | 4,840,107 | 1,688,460 | 0.3600 | 2396 | 8.7576 | 24.3627 | 45.35% | 29.68% | 1344 |
| J2 | 42 | 4,840,107 | 1,688,460 | 0.6020 | 4335 | 9.9892 | 20.2342 | 29.90% | 21.67% | 1344 |
| J2 | 43 | 4,840,107 | 1,688,460 | 0.6028 | 4358 | 10.4328 | 21.9458 | 30.71% | 21.87% | 1344 |
| J3 | 42 | 2,318,224 | 1,688,460 | 0.2488 | 2077 | 0.0000 | 0.0000 | 0.00% | 0.00% | 1344 |
| J3 | 43 | 2,318,224 | 1,688,460 | 0.2498 | 2086 | 0.0000 | 0.0000 | 0.00% | 0.00% | 1344 |
| J1c | 42 | 4,840,107 | 1,688,460 | 0.3588 | 2396 | 789.1187 | 79.7974 | 60.06% | 39.90% | 1344 |
| J1c | 43 | 4,840,107 | 1,688,460 | 0.3596 | 2396 | 761.0208 | 87.4772 | 60.50% | 45.24% | 1344 |
| J1l4 | 42 | 4,840,107 | 1,688,460 | 0.3587 | 2396 | 12.2349 | 20.1915 | 73.69% | 55.24% | 1344 |
| J1l16 | 42 | 4,840,107 | 1,688,460 | 0.3592 | 2396 | 12.2349 | 20.1119 | 91.25% | 81.45% | 1344 |

## 三项自检的原始输出（逐档）

### F_s42
    SELFTEST_1 {"arm": "F", "params": {"core_total": 1688460, "core_trainable": 1688460, "heads": {"pronoun": {"total": 630278, "trainable": 630278}, "sentiment": {"total": 630278, "trainable": 630278}, "relation": {"total": 630021, "trainable": 630021}, "person": {"total": 631306, "trainable": 631306}, "negation": {"total": 629764, "trainable": 629764}}, "trainable_total": 4840107}}
    SELFTEST_2 本档无蒸馏/正则项（kd=0, replay=0）；唯一正则是 AdamW weight_decay=1e-4。kd_mean_all=0.000000
    SELFTEST_3 本档不冻结老卡头（无此项）
    ALIGN_CHECK pronoun s42 eval==val: True (n=600)
    ALIGN_CHECK sentiment s42 eval==val: True (n=3200)
    ALIGN_CHECK relation s42 eval==val: True (n=720)
    ALIGN_CHECK person s42 eval==val: True (n=600)
    ALIGN_CHECK negation s42 eval==val: True (n=600)

### J0_s42
    SELFTEST_1 {"arm": "J0", "params": {"core_total": 1688460, "core_trainable": 1688460, "heads": {"pronoun": {"total": 630278, "trainable": 630278}, "sentiment": {"total": 630278, "trainable": 630278}, "relation": {"total": 630021, "trainable": 630021}, "person": {"total": 631306, "trainable": 631306}, "negation": {"total": 629764, "trainable": 629764}}, "trainable_total": 4840107}}
    SELFTEST_2 本档无蒸馏/正则项（kd=0, replay=0）；唯一正则是 AdamW weight_decay=1e-4。kd_mean_all=0.000000
    SELFTEST_3 本档不冻结老卡头（无此项）
    ALIGN_CHECK pronoun s42 eval==val: True (n=600)
    ALIGN_CHECK sentiment s42 eval==val: True (n=3200)
    ALIGN_CHECK relation s42 eval==val: True (n=720)
    ALIGN_CHECK person s42 eval==val: True (n=600)
    ALIGN_CHECK negation s42 eval==val: True (n=600)

### J0_s43
    SELFTEST_1 {"arm": "J0", "params": {"core_total": 1688460, "core_trainable": 1688460, "heads": {"pronoun": {"total": 630278, "trainable": 630278}, "sentiment": {"total": 630278, "trainable": 630278}, "relation": {"total": 630021, "trainable": 630021}, "person": {"total": 631306, "trainable": 631306}, "negation": {"total": 629764, "trainable": 629764}}, "trainable_total": 4840107}}
    SELFTEST_2 本档无蒸馏/正则项（kd=0, replay=0）；唯一正则是 AdamW weight_decay=1e-4。kd_mean_all=0.000000
    SELFTEST_3 本档不冻结老卡头（无此项）
    ALIGN_CHECK pronoun s43 eval==val: True (n=600)
    ALIGN_CHECK sentiment s43 eval==val: True (n=3200)
    ALIGN_CHECK relation s43 eval==val: True (n=720)
    ALIGN_CHECK person s43 eval==val: True (n=600)
    ALIGN_CHECK negation s43 eval==val: True (n=600)

### J1_s42
    SELFTEST_1 {"arm": "J1", "params": {"core_total": 1688460, "core_trainable": 1688460, "heads": {"pronoun": {"total": 630278, "trainable": 630278}, "sentiment": {"total": 630278, "trainable": 630278}, "relation": {"total": 630021, "trainable": 630021}, "person": {"total": 631306, "trainable": 631306}, "negation": {"total": 629764, "trainable": 629764}}, "trainable_total": 4840107}}
    SELFTEST_2 kd_first_step=12.234852 kd_mean_all=20.762968 kd_epoch1=86.545018 kd_last=6.806323 kd_share_first=0.542780 kd_share_last=0.284876 kd_share_all=0.437431
    SELFTEST_2 verdict 非零=True
    SELFTEST_3 本档不冻结老卡头（无此项）
    ALIGN_CHECK pronoun s42 eval==val: True (n=600)
    ALIGN_CHECK sentiment s42 eval==val: True (n=3200)
    ALIGN_CHECK relation s42 eval==val: True (n=720)
    ALIGN_CHECK person s42 eval==val: True (n=600)
    ALIGN_CHECK negation s42 eval==val: True (n=600)

### J1_s43
    SELFTEST_1 {"arm": "J1", "params": {"core_total": 1688460, "core_trainable": 1688460, "heads": {"pronoun": {"total": 630278, "trainable": 630278}, "sentiment": {"total": 630278, "trainable": 630278}, "relation": {"total": 630021, "trainable": 630021}, "person": {"total": 631306, "trainable": 631306}, "negation": {"total": 629764, "trainable": 629764}}, "trainable_total": 4840107}}
    SELFTEST_2 kd_first_step=8.757553 kd_mean_all=24.362683 kd_epoch1=115.081791 kd_last=7.364196 kd_share_first=0.564689 kd_share_last=0.296850 kd_share_all=0.453541
    SELFTEST_2 verdict 非零=True
    SELFTEST_3 本档不冻结老卡头（无此项）
    ALIGN_CHECK pronoun s43 eval==val: True (n=600)
    ALIGN_CHECK sentiment s43 eval==val: True (n=3200)
    ALIGN_CHECK relation s43 eval==val: True (n=720)
    ALIGN_CHECK person s43 eval==val: True (n=600)
    ALIGN_CHECK negation s43 eval==val: True (n=600)

### J2_s42
    SELFTEST_1 {"arm": "J2", "params": {"core_total": 1688460, "core_trainable": 1688460, "heads": {"pronoun": {"total": 630278, "trainable": 630278}, "sentiment": {"total": 630278, "trainable": 630278}, "relation": {"total": 630021, "trainable": 630021}, "person": {"total": 631306, "trainable": 631306}, "negation": {"total": 629764, "trainable": 629764}}, "trainable_total": 4840107}}
    SELFTEST_2 kd_first_step=9.989159 kd_mean_all=20.234163 kd_epoch1=86.361316 kd_last=7.820425 kd_share_first=0.382431 kd_share_last=0.216715 kd_share_all=0.299015
    SELFTEST_2 verdict 非零=True
    SELFTEST_2 replay_mean=13.824877
    SELFTEST_3 本档不冻结老卡头（无此项）
    ALIGN_CHECK pronoun s42 eval==val: True (n=600)
    ALIGN_CHECK sentiment s42 eval==val: True (n=3200)
    ALIGN_CHECK relation s42 eval==val: True (n=720)
    ALIGN_CHECK person s42 eval==val: True (n=600)
    ALIGN_CHECK negation s42 eval==val: True (n=600)

### J2_s43
    SELFTEST_1 {"arm": "J2", "params": {"core_total": 1688460, "core_trainable": 1688460, "heads": {"pronoun": {"total": 630278, "trainable": 630278}, "sentiment": {"total": 630278, "trainable": 630278}, "relation": {"total": 630021, "trainable": 630021}, "person": {"total": 631306, "trainable": 631306}, "negation": {"total": 629764, "trainable": 629764}}, "trainable_total": 4840107}}
    SELFTEST_2 kd_first_step=10.432812 kd_mean_all=21.945762 kd_epoch1=117.376063 kd_last=8.086798 kd_share_first=0.409478 kd_share_last=0.218731 kd_share_all=0.307066
    SELFTEST_2 verdict 非零=True
    SELFTEST_2 replay_mean=13.725178
    SELFTEST_3 本档不冻结老卡头（无此项）
    ALIGN_CHECK pronoun s43 eval==val: True (n=600)
    ALIGN_CHECK sentiment s43 eval==val: True (n=3200)
    ALIGN_CHECK relation s43 eval==val: True (n=720)
    ALIGN_CHECK person s43 eval==val: True (n=600)
    ALIGN_CHECK negation s43 eval==val: True (n=600)

### J3_s42
    SELFTEST_1 {"arm": "J3", "params": {"core_total": 1688460, "core_trainable": 1688460, "heads": {"pronoun": {"total": 630278, "trainable": 0}, "sentiment": {"total": 630278, "trainable": 0}, "relation": {"total": 630021, "trainable": 0}, "person": {"total": 631306, "trainable": 0}, "negation": {"total": 629764, "trainable": 629764}}, "trainable_total": 2318224}}
    SELFTEST_2 本档无蒸馏/正则项（kd=0, replay=0）；唯一正则是 AdamW weight_decay=1e-4。kd_mean_all=0.000000
    SELFTEST_3 pronoun.requires_grad = [False] (n_params=56)
    SELFTEST_3 sentiment.requires_grad = [False] (n_params=56)
    SELFTEST_3 relation.requires_grad = [False] (n_params=56)
    SELFTEST_3 person.requires_grad = [False] (n_params=56)
    SELFTEST_3 verdict: 四张老卡头全部 requires_grad=False ✅
    ALIGN_CHECK pronoun s42 eval==val: True (n=600)
    ALIGN_CHECK sentiment s42 eval==val: True (n=3200)
    ALIGN_CHECK relation s42 eval==val: True (n=720)
    ALIGN_CHECK person s42 eval==val: True (n=600)
    ALIGN_CHECK negation s42 eval==val: True (n=600)

### J3_s43
    SELFTEST_1 {"arm": "J3", "params": {"core_total": 1688460, "core_trainable": 1688460, "heads": {"pronoun": {"total": 630278, "trainable": 0}, "sentiment": {"total": 630278, "trainable": 0}, "relation": {"total": 630021, "trainable": 0}, "person": {"total": 631306, "trainable": 0}, "negation": {"total": 629764, "trainable": 629764}}, "trainable_total": 2318224}}
    SELFTEST_2 本档无蒸馏/正则项（kd=0, replay=0）；唯一正则是 AdamW weight_decay=1e-4。kd_mean_all=0.000000
    SELFTEST_3 pronoun.requires_grad = [False] (n_params=56)
    SELFTEST_3 sentiment.requires_grad = [False] (n_params=56)
    SELFTEST_3 relation.requires_grad = [False] (n_params=56)
    SELFTEST_3 person.requires_grad = [False] (n_params=56)
    SELFTEST_3 verdict: 四张老卡头全部 requires_grad=False ✅
    ALIGN_CHECK pronoun s43 eval==val: True (n=600)
    ALIGN_CHECK sentiment s43 eval==val: True (n=3200)
    ALIGN_CHECK relation s43 eval==val: True (n=720)
    ALIGN_CHECK person s43 eval==val: True (n=600)
    ALIGN_CHECK negation s43 eval==val: True (n=600)

### J1c_s42
    SELFTEST_1 {"arm": "J1", "params": {"core_total": 1688460, "core_trainable": 1688460, "heads": {"pronoun": {"total": 630278, "trainable": 630278}, "sentiment": {"total": 630278, "trainable": 630278}, "relation": {"total": 630021, "trainable": 630021}, "person": {"total": 631306, "trainable": 631306}, "negation": {"total": 629764, "trainable": 629764}}, "trainable_total": 4840107}}
    SELFTEST_2 kd_first_step=789.118713 kd_mean_all=79.797402 kd_epoch1=418.891204 kd_last=14.261555 kd_share_first=0.680237 kd_share_last=0.399043 kd_share_all=0.600552
    SELFTEST_2 verdict 非零=True
    SELFTEST_3 本档不冻结老卡头（无此项）
    ALIGN_CHECK pronoun s42 eval==val: True (n=600)
    ALIGN_CHECK sentiment s42 eval==val: True (n=3200)
    ALIGN_CHECK relation s42 eval==val: True (n=720)
    ALIGN_CHECK person s42 eval==val: True (n=600)
    ALIGN_CHECK negation s42 eval==val: True (n=600)

### J1c_s43
    SELFTEST_1 {"arm": "J1", "params": {"core_total": 1688460, "core_trainable": 1688460, "heads": {"pronoun": {"total": 630278, "trainable": 630278}, "sentiment": {"total": 630278, "trainable": 630278}, "relation": {"total": 630021, "trainable": 630021}, "person": {"total": 631306, "trainable": 631306}, "negation": {"total": 629764, "trainable": 629764}}, "trainable_total": 4840107}}
    SELFTEST_2 kd_first_step=761.020752 kd_mean_all=87.477243 kd_epoch1=417.978697 kd_last=19.877419 kd_share_first=0.678621 kd_share_last=0.452386 kd_share_all=0.604983
    SELFTEST_2 verdict 非零=True
    SELFTEST_3 本档不冻结老卡头（无此项）
    ALIGN_CHECK pronoun s43 eval==val: True (n=600)
    ALIGN_CHECK sentiment s43 eval==val: True (n=3200)
    ALIGN_CHECK relation s43 eval==val: True (n=720)
    ALIGN_CHECK person s43 eval==val: True (n=600)
    ALIGN_CHECK negation s43 eval==val: True (n=600)

### J1l4_s42
    SELFTEST_1 {"arm": "J1", "params": {"core_total": 1688460, "core_trainable": 1688460, "heads": {"pronoun": {"total": 630278, "trainable": 630278}, "sentiment": {"total": 630278, "trainable": 630278}, "relation": {"total": 630021, "trainable": 630021}, "person": {"total": 631306, "trainable": 631306}, "negation": {"total": 629764, "trainable": 629764}}, "trainable_total": 4840107}}
    SELFTEST_2 kd_first_step=12.234852 kd_mean_all=20.191508 kd_epoch1=95.126757 kd_last=5.875602 kd_share_first=0.825829 kd_share_last=0.552386 kd_share_all=0.736929
    SELFTEST_2 verdict 非零=True
    SELFTEST_3 本档不冻结老卡头（无此项）
    ALIGN_CHECK pronoun s42 eval==val: True (n=600)
    ALIGN_CHECK sentiment s42 eval==val: True (n=3200)
    ALIGN_CHECK relation s42 eval==val: True (n=720)
    ALIGN_CHECK person s42 eval==val: True (n=600)
    ALIGN_CHECK negation s42 eval==val: True (n=600)

### J1l16_s42
    SELFTEST_1 {"arm": "J1", "params": {"core_total": 1688460, "core_trainable": 1688460, "heads": {"pronoun": {"total": 630278, "trainable": 630278}, "sentiment": {"total": 630278, "trainable": 630278}, "relation": {"total": 630021, "trainable": 630021}, "person": {"total": 631306, "trainable": 631306}, "negation": {"total": 629764, "trainable": 629764}}, "trainable_total": 4840107}}
    SELFTEST_2 kd_first_step=12.234852 kd_mean_all=20.111891 kd_epoch1=76.813419 kd_last=5.732618 kd_share_first=0.945675 kd_share_last=0.814464 kd_share_all=0.912510
    SELFTEST_2 verdict 非零=True
    SELFTEST_3 本档不冻结老卡头（无此项）
    ALIGN_CHECK pronoun s42 eval==val: True (n=600)
    ALIGN_CHECK sentiment s42 eval==val: True (n=3200)
    ALIGN_CHECK relation s42 eval==val: True (n=720)
    ALIGN_CHECK person s42 eval==val: True (n=600)
    ALIGN_CHECK negation s42 eval==val: True (n=600)
