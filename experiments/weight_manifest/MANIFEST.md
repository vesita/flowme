# 全仓 .pt 权重清单（sha256）

生成：`uv run python experiments/weight_manifest/scan.py`（只读扫描）
扫描时间：2026-10-07 14:06:01 +0800
线上基座 doc_sha256 = `dc27db5337d16f185dd08cc50505390e53d03606b9d307457828f4ecbfdf6b1a`（M0 自校验通过 = True）

**262 个 .pt / 3,380,031,533 字节**；与线上 base 一致 **5** / 不一致 **58** / 不适用 **199**（含加载失败 0）

列：路径 | 字节 | 文件 sha256 前16 | 内含 encoder? | doc_encoder sha256 前16 | 与线上 base 一致? | 来源线索（base / split_from）

## checkpoints/  （29 个，320,868,021 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `arm_a2_seed43.pt` | 16,933,387 | `cad4380efd21af2a` | 是 | `2ef5f66e639f90ad` | 否 | — |
| `arm_aprime_dtseek.pt` | 16,934,411 | `7331af6d23801a77` | 是 | `dd31a053ff5cdd36` | 否 | — |
| `arm_b_freeze_base_dtseek.pt` | 16,936,203 | `7d45183da5a26401` | 是 | `d44b5ce57f54e49e` | 否 | — |
| `arm_c12_warmup12_freeze_dtseek.pt` | 16,937,739 | `593de07991bc51b2` | 是 | `f2a3c6aafa5627d0` | 否 | — |
| `arm_c_warmup4_freeze_dtseek.pt` | 16,936,971 | `e03df9524ea40421` | 是 | `3e13942aaf7e2579` | 否 | — |
| `arm_cprime_headonly_dtseek.pt` | 16,936,843 | `7f7bb3dff8a538dc` | 是 | `dc27db5337d16f18` | 是 | — |
| `arm_lrbase1e4_5card.pt` | 19,474,427 | `390026621223510b` | 是 | `83747014ccb7353c` | 否 | — |
| `arm_neg5_seed42.pt` | 19,473,179 | `6277c4eac5452ff0` | 是 | `801657fb40a803a2` | 否 | — |
| `arm_neg5_seed43.pt` | 19,473,179 | `fbf855670808a0ec` | 是 | `16fd4e809d131122` | 否 | — |
| `arm_remedy_frozen_add_neg.pt` | 19,476,427 | `d2d9894c985ad661` | 是 | `dd31a053ff5cdd36` | 否 | — |
| `base_encoder.pt` | 6,764,277 | `dcf237ae99cef7d3` | 是 | `dc27db5337d16f18` | 是 | — |
| `e5_single_pronoun.pt` | 9,306,397 | `d8894ca1775e2a06` | 是 | `db59ccfb61ea6418` | 否 | — |
| `e5b_single_person.pt` | 9,310,685 | `6d3e80b7da0d8377` | 是 | `405758f686bea018` | 否 | — |
| `e5b_single_pronoun.pt` | 9,306,485 | `f08a1fa2deea1000` | 是 | `db59ccfb61ea6418` | 否 | — |
| `e5b_single_relation.pt` | 9,305,485 | `941608b95d1f459e` | 是 | `221453f406b87e15` | 否 | — |
| `e5b_single_sentiment.pt` | 9,306,725 | `36135f9adcda2486` | 是 | `8f976a68a6c7b3c7` | 否 | — |
| `e5c_single_person.pt` | 9,310,685 | `022dd5407c3bbc54` | 是 | `405758f686bea018` | 否 | — |
| `e5c_single_pronoun.pt` | 9,306,485 | `786b9679859e185f` | 是 | `db59ccfb61ea6418` | 否 | — |
| `e5c_single_relation.pt` | 9,305,485 | `20748dd9732fd4be` | 是 | `221453f406b87e15` | 否 | — |
| `e5c_single_sentiment.pt` | 9,306,725 | `f2f18cdc0ed2425b` | 是 | `8f976a68a6c7b3c7` | 否 | — |
| `multitask_v2_dtseek.pt` | 16,934,795 | `7a0c6a020c0fe9e4` | 是 | `dc27db5337d16f18` | 是 | — |
| `ndb_recheck_base_seed42.pt` | 2,547,365 | `c09f2242d9201d1a` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `ndb_recheck_base_seed43.pt` | 2,547,365 | `870a44ec0cb31e65` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `ndb_recheck_ndb_seed42.pt` | 2,550,474 | `27acd3e1037494ba` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `ndb_recheck_ndb_seed43.pt` | 2,550,474 | `4ff39b2204d23960` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `negation_accept_card.pt` | 2,540,961 | `b1d8d2003b29172b` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `negation_accept_card_e12.pt` | 2,541,273 | `62386f60e3d68070` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `reply_pick_joint.pt` | 9,307,397 | `5a7a4708b64aab15` | 是 | `e70be463fdc814ec` | 否 | — |
| `smoke_test.pt` | 9,305,717 | `cff1bc4d90db39cc` | 是 | `72553fc9ca618922` | 否 | — |

## checkpoints/cards/  （4 个，10,164,584 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `person.pt` | 2,542,077 | `2364296b4f039741` | 否 | `None` | — | split_from=checkpoints/multitask_v2_dtseek.pt |
| `pronoun.pt` | 2,540,731 | `204b3fb2ac06d6ec` | 否 | `None` | — | split_from=checkpoints/multitask_v2_dtseek.pt |
| `relation.pt` | 2,540,281 | `4ddb0f423adcff45` | 否 | `None` | — | split_from=checkpoints/multitask_v2_dtseek.pt |
| `sentiment.pt` | 2,541,495 | `2ae22bcd247529b1` | 否 | `None` | — | split_from=checkpoints/multitask_v2_dtseek.pt |

## experiments/anchored_select/cards/  （6 个，49,105,898 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `anchored_s42.pt` | 2,540,465 | `4b2f41665c6c4fb2` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `anchored_s43.pt` | 2,540,465 | `df8dfd2e4293fcfb` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `joint_s42.pt` | 19,472,267 | `2db6d052ea53a33f` | 是 | `d4c30c3e6d82d1bc` | 否 | — |
| `joint_s43.pt` | 19,472,267 | `e4de700d90e9879f` | 是 | `e344eea3b72f0ca3` | 否 | — |
| `shuf_s42.pt` | 2,540,217 | `5d14746653c87af2` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `shuf_s43.pt` | 2,540,217 | `00b12fec0204dcd5` | 否 | `None` | — | base=checkpoints/base_encoder.pt |

## experiments/bag_modules/cache/  （4 个，417,631,028 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `tok_4169598e4af7_L64.pt` | 82,081,997 | `257acc681e12e98a` | 否 | `None` | — | — |
| `tok_5d0afaf586cc_L64.pt` | 40,057,037 | `bb6f9dce6b3176e5` | 否 | `None` | — | — |
| `tok_bac00ff05e35_L64.pt` | 262,657,997 | `594cf4a449bf4536` | 否 | `None` | — | — |
| `tok_d3c4cb18cbe3_L64.pt` | 32,833,997 | `0da59eb5637d934f` | 否 | `None` | — | — |

## experiments/bag_modules/weights/  （22 个，16,908,610 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `A_s42.pt` | 748,576 | `06fc0e02bb0ff3c2` | 否 | `None` | — | — |
| `A_s42_steps60.pt` | 749,288 | `c2de82152311dd31` | 否 | `None` | — | — |
| `A_s43.pt` | 748,576 | `22056e95f12435e6` | 否 | `None` | — | — |
| `P_s42.pt` | 755,660 | `da90b8d82ef0db7b` | 否 | `None` | — | — |
| `P_s42_steps60.pt` | 756,532 | `3888d792b9063fe9` | 否 | `None` | — | — |
| `P_s43.pt` | 755,660 | `a64d8781ba077040` | 否 | `None` | — | — |
| `UP_s42.pt` | 779,517 | `587f332c2521da83` | 否 | `None` | — | — |
| `UP_s42_steps60.pt` | 780,909 | `74a36bcde31480c2` | 否 | `None` | — | — |
| `UP_s43.pt` | 779,517 | `09791c384e9a90bf` | 否 | `None` | — | — |
| `U_s42.pt` | 772,403 | `d351e5585807dd89` | 否 | `None` | — | — |
| `U_s42_mcls.pt` | 773,557 | `05b201712a0a45d1` | 否 | `None` | — | — |
| `U_s42_mrole.pt` | 773,583 | `b7d3c559b005dcda` | 否 | `None` | — | — |
| `U_s42_mtype-cls.pt` | 773,687 | `1db33eb9a2617d3c` | 否 | `None` | — | — |
| `U_s42_mtype.pt` | 773,583 | `8438c913670c3748` | 否 | `None` | — | — |
| `U_s42_rand.pt` | 773,557 | `2f7a9b008d2dc01b` | 否 | `None` | — | — |
| `U_s42_steps60.pt` | 773,635 | `6c8a79e48eeb820e` | 否 | `None` | — | — |
| `U_s43.pt` | 772,403 | `48876842ba3546ff` | 否 | `None` | — | — |
| `U_s43_mcls.pt` | 773,557 | `34f9b7dd2f50af5f` | 否 | `None` | — | — |
| `U_s43_mrole.pt` | 773,583 | `d2085950a681a9a7` | 否 | `None` | — | — |
| `U_s43_mtype-cls.pt` | 773,687 | `b0ed09dc98bd68a9` | 否 | `None` | — | — |
| `U_s43_mtype.pt` | 773,583 | `b4926df728c001de` | 否 | `None` | — | — |
| `U_s43_rand.pt` | 773,557 | `a11324065456e9fc` | 否 | `None` | — | — |

## experiments/capability_map/cards/  （16 个，41,794,432 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `person_bypass_s42.pt` | 2,615,583 | `e6471d8c0687bebd` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `person_bypass_s43.pt` | 2,615,583 | `4da6a6b3315d09aa` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `person_frozen_s42.pt` | 2,615,583 | `d3ef69521b9c9fe4` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `person_frozen_s43.pt` | 2,615,583 | `0450265f5697b937` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `pronoun_bypass_s42.pt` | 2,611,237 | `827177739bda14ca` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `pronoun_bypass_s43.pt` | 2,611,237 | `ec12adf9c06d3e56` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `pronoun_frozen_s42.pt` | 2,611,237 | `aaf11a05c5b97d44` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `pronoun_frozen_s43.pt` | 2,611,237 | `a3c096b46666ecef` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `relation_bypass_s42.pt` | 2,610,347 | `7f5edf48ebf84083` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `relation_bypass_s43.pt` | 2,610,347 | `b6b22c59aa7dd68b` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `relation_frozen_s42.pt` | 2,610,347 | `7daac68c925439c9` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `relation_frozen_s43.pt` | 2,610,347 | `7345825c9bb9b0b8` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `sentiment_bypass_s42.pt` | 2,611,441 | `cb1fef15d1c92abe` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `sentiment_bypass_s43.pt` | 2,611,441 | `6cd642b864f3b95d` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `sentiment_frozen_s42.pt` | 2,611,441 | `aae6e68692a4d86a` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `sentiment_frozen_s43.pt` | 2,611,441 | `e197b7643a9443a1` | 否 | `None` | — | base=checkpoints/base_encoder.pt |

## experiments/compose_ops/artifacts/  （1 个，6,764,341 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `base_encoder.pt` | 6,764,341 | `ed639c5f2de88238` | 是 | `801657fb40a803a2` | 否 | — |

## experiments/compose_ops/artifacts/cards/  （5 个，12,704,097 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `negation.pt` | 2,539,257 | `ba1a80f9af81b1a2` | 否 | `None` | — | split_from=checkpoints/arm_neg5_seed42.pt |
| `person.pt` | 2,542,141 | `6909648b16743547` | 否 | `None` | — | split_from=checkpoints/arm_neg5_seed42.pt |
| `pronoun.pt` | 2,540,795 | `875e606e43863241` | 否 | `None` | — | split_from=checkpoints/arm_neg5_seed42.pt |
| `relation.pt` | 2,540,345 | `da2b819a6452fe7a` | 否 | `None` | — | split_from=checkpoints/arm_neg5_seed42.pt |
| `sentiment.pt` | 2,541,559 | `c9e16f257f005b56` | 否 | `None` | — | split_from=checkpoints/arm_neg5_seed42.pt |

## experiments/core_generalize/cores/  （11 个，132,816,143 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `C1_s42.pt` | 9,300,501 | `887735f802e26655` | 是 | `00829df5e8752a89` | 否 | — |
| `C1_s43.pt` | 9,300,501 | `dd4a0f7f364fa929` | 是 | `f725386550eb584a` | 否 | — |
| `C1x5_s42.pt` | 9,305,605 | `44dd6f99e1ed6f5e` | 是 | `685923f88837eebe` | 否 | — |
| `C3_s42.pt` | 14,380,211 | `5d38897e9591edd0` | 是 | `d0bb4a512cb82ad5` | 否 | — |
| `C3_s43.pt` | 14,380,211 | `9dea34f18b6b7810` | 是 | `340ff864c8aeb31f` | 否 | — |
| `C5_s42.pt` | 19,464,291 | `bd4ffad9b376152f` | 是 | `a9a990110cfd0bab` | 否 | — |
| `C5_s43.pt` | 19,464,291 | `b4344399ced7d992` | 是 | `5cbcfd1f19ecd8aa` | 否 | — |
| `PRE_idiom_s42.pt` | 9,303,805 | `defef238b45e084f` | 是 | `a7eda430c5ee63a8` | 否 | — |
| `PRE_idiom_s43.pt` | 9,303,805 | `f27fe98c7d688195` | 是 | `3026078aefe3eb31` | 否 | — |
| `PRE_ownership_s42.pt` | 9,306,461 | `ae5089cd479ad2c0` | 是 | `98fad59e0687f1c1` | 否 | — |
| `PRE_ownership_s43.pt` | 9,306,461 | `c5b9750ab5fbea0e` | 是 | `a858f76649e3e77a` | 否 | — |

## experiments/core_keep/cards/  （13 个，253,102,039 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `F_s42.pt` | 19,455,403 | `0fb2db14dec44572` | 是 | `801657fb40a803a2` | 否 | — |
| `J0_s42.pt` | 19,468,451 | `180a4264c9281062` | 是 | `c05c25b1f180074e` | 否 | — |
| `J0_s43.pt` | 19,468,451 | `6571e2ae6f407539` | 是 | `dc1dbadbaf6db608` | 否 | — |
| `J1_s42.pt` | 19,468,451 | `e6e07fa9db92133d` | 是 | `2d084547189adcdc` | 否 | — |
| `J1_s43.pt` | 19,468,451 | `16a489a1fc0b3f4a` | 是 | `3d2c68f7dc8ade8d` | 否 | — |
| `J1c_s42.pt` | 19,474,267 | `e4f27c4647a2aaec` | 是 | `f580e698da020083` | 否 | — |
| `J1c_s43.pt` | 19,474,267 | `a506139bb64e1176` | 是 | `fb4cb8562f4cb9c8` | 否 | — |
| `J1l16_s42.pt` | 19,475,403 | `168061093ac8b5fe` | 是 | `ff2faba586c3579a` | 否 | — |
| `J1l4_s42.pt` | 19,475,091 | `52c50969b99ae8fb` | 是 | `c1aec43ac2fb9f3f` | 否 | — |
| `J2_s42.pt` | 19,468,451 | `db20832f58b7145e` | 是 | `c057d879950f51a5` | 否 | — |
| `J2_s43.pt` | 19,468,451 | `8e23d3af9c9b3fc8` | 是 | `c9f65f941d94bb77` | 否 | — |
| `J3_s42.pt` | 19,468,451 | `4d4f4d11c1d38f06` | 是 | `686697d4754cf642` | 否 | — |
| `J3_s43.pt` | 19,468,451 | `c732df281032095b` | 是 | `db66318a93c7f4f9` | 否 | — |

## experiments/cumulative_add/cards/  （12 个，294,636,772 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `ctrl_idiom_s42.pt` | 24,554,003 | `727201be0412cb90` | 是 | `38da4f217eed6b22` | 否 | — |
| `ctrl_idiom_s43.pt` | 24,554,003 | `6c74dbe1f6fe5a4e` | 是 | `3a2025eb8d55f6fd` | 否 | — |
| `ctrl_ownership_s42.pt` | 24,555,955 | `d024b9d09e55578d` | 是 | `62e415aa8711614c` | 否 | — |
| `ctrl_ownership_s43.pt` | 24,555,955 | `5dd5d150bb30e37c` | 是 | `95c110c1c8084be0` | 否 | — |
| `gate0_s42.pt` | 24,551,499 | `5fd35f05c150e41c` | 是 | `dc27db5337d16f18` | 是 | — |
| `gate0_s43.pt` | 24,551,499 | `69b8c5aa8a2a451e` | 是 | `dc27db5337d16f18` | 是 | — |
| `step1_s42.pt` | 24,551,883 | `9e2ccabd7f92ed2c` | 是 | `a552143a009b26dd` | 否 | — |
| `step1_s43.pt` | 24,551,883 | `ad26e40e8cb87e37` | 是 | `6cf07f3d1a868667` | 否 | — |
| `step2_s42.pt` | 24,552,267 | `7835ea2369d30be9` | 是 | `ea646d4ba752bed2` | 否 | — |
| `step2_s43.pt` | 24,552,267 | `1f4302827b00992f` | 是 | `309c873099194d2b` | 否 | — |
| `step3_s42.pt` | 24,552,779 | `4616029c70955c1e` | 是 | `5856bc0df0d1bf9f` | 否 | — |
| `step3_s43.pt` | 24,552,779 | `00d941ff44b72586` | 是 | `2dc70a14693fa22a` | 否 | — |

## experiments/free_rule_floor/weights/  （10 个，7,690,382 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `A_s42.pt` | 748,576 | `06fc0e02bb0ff3c2` | 否 | `None` | — | — |
| `A_s43.pt` | 748,576 | `22056e95f12435e6` | 否 | `None` | — | — |
| `N_s42.pt` | 771,138 | `8e2cd036f7da418a` | 否 | `None` | — | — |
| `N_s43.pt` | 771,138 | `d7aded935d735987` | 否 | `None` | — | — |
| `UP_s42.pt` | 779,517 | `587f332c2521da83` | 否 | `None` | — | — |
| `UP_s43.pt` | 779,517 | `09791c384e9a90bf` | 否 | `None` | — | — |
| `U_s42.pt` | 772,403 | `d351e5585807dd89` | 否 | `None` | — | — |
| `U_s42_rand.pt` | 773,557 | `2f7a9b008d2dc01b` | 否 | `None` | — | — |
| `U_s43.pt` | 772,403 | `48876842ba3546ff` | 否 | `None` | — | — |
| `U_s43_rand.pt` | 773,557 | `a11324065456e9fc` | 否 | `None` | — | — |

## experiments/funcword_minpair/cache/  （7 个，27,726,847 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `gen_3dc8a93b05fb_L64_M4.pt` | 219,017 | `a4ce23d62e6a0092` | 否 | `None` | — | — |
| `gen_6e13e4eee2b9_L64_M4.pt` | 10,952,713 | `9c15c5103e66d229` | 否 | `None` | — | — |
| `gen_8a879e610439_L64_M4.pt` | 2,914,185 | `fa094bdab3515a8b` | 否 | `None` | — | — |
| `gen_8da3bb789223_L64_M4.pt` | 6,499,913 | `f4a188f74bb96597` | 否 | `None` | — | — |
| `gen_a0efdc6c9188_L64_M4.pt` | 2,914,185 | `378654cb0989013f` | 否 | `None` | — | — |
| `gen_cb97fd12b97a_L64_M4.pt` | 1,526,217 | `35cdeb809c96df30` | 否 | `None` | — | — |
| `gen_fa47a01876ad_L64_M4.pt` | 2,700,617 | `fc23169c3ec02a6e` | 否 | `None` | — | — |

## experiments/funcword_minpair/weights/  （15 个，72,997,765 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `MP+_s42.pt` | 4,866,282 | `eee3f4742d1f5a4c` | 否 | `None` | — | — |
| `MP+_s42_rand.pt` | 4,866,919 | `692737832af86f44` | 否 | `None` | — | — |
| `MP+_s42_steps60.pt` | 4,866,994 | `6295670be9a5c23c` | 否 | `None` | — | — |
| `MP+_s43.pt` | 4,866,282 | `49326aeae59daf16` | 否 | `None` | — | — |
| `MP+_s43_rand.pt` | 4,866,919 | `dd6bfdaffcd1925e` | 否 | `None` | — | — |
| `MP_s42.pt` | 4,865,745 | `789682d96a6e6dfd` | 否 | `None` | — | — |
| `MP_s42_rand.pt` | 4,866,894 | `ce28349597c73f78` | 否 | `None` | — | — |
| `MP_s42_steps60.pt` | 4,866,969 | `fd6b9cbdc780088c` | 否 | `None` | — | — |
| `MP_s43.pt` | 4,865,745 | `3f25bfddd1f0602a` | 否 | `None` | — | — |
| `MP_s43_rand.pt` | 4,866,894 | `842492e6f0c5e2e5` | 否 | `None` | — | — |
| `M_s42.pt` | 4,865,720 | `b695c109e5bcdbe0` | 否 | `None` | — | — |
| `M_s42_rand.pt` | 4,866,869 | `8849bafa33c66b4d` | 否 | `None` | — | — |
| `M_s42_steps60.pt` | 4,866,944 | `a69d5d703d095916` | 否 | `None` | — | — |
| `M_s43.pt` | 4,865,720 | `45fc920683a5c501` | 否 | `None` | — | — |
| `M_s43_rand.pt` | 4,866,869 | `475ed8d77d387114` | 否 | `None` | — | — |

## experiments/gen_data_loop/cards/  （8 个，20,323,456 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `cand_C_s42.pt` | 2,540,277 | `c7d66e667a1594f3` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `cand_C_s43.pt` | 2,540,277 | `c9db87fe926ae65b` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `cand_C_shuf_s42.pt` | 2,540,587 | `98a2f68a24accdea` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `cand_C_shuf_s43.pt` | 2,540,587 | `92620c7c186415e4` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `cand_P_s42.pt` | 2,540,277 | `bb5c48db468dedb3` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `cand_P_s43.pt` | 2,540,277 | `e0419a10a02b553b` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `cand_P_shuf_s42.pt` | 2,540,587 | `8c2a55776d572b0a` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `cand_P_shuf_s43.pt` | 2,540,587 | `ecbd0c645532aa53` | 否 | `None` | — | base=checkpoints/base_encoder.pt |

## experiments/gen_dispatch/cache/  （2 个，9,303,594 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `idiom_base.pt` | 6,763,957 | `bde0e72712937ecf` | 是 | `38da4f217eed6b22` | 否 | — |
| `idiom_card.pt` | 2,539,637 | `1916c31f6059596b` | 否 | `None` | — | — |

## experiments/mode_conditioned_skel/cache/  （4 个，294,132,200 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `enc_6e20fdc8a40e_L64.pt` | 3,546,938 | `cd45349cb150bbca` | 否 | `None` | — | — |
| `enc_73fc851bb2af_L64.pt` | 3,546,938 | `e420079a474acb74` | 否 | `None` | — | — |
| `enc_b9103ab9cb7c_L64.pt` | 3,546,938 | `b2aa3c8cb270035c` | 否 | `None` | — | — |
| `enc_bac00ff05e35_L64.pt` | 283,491,386 | `e28584b3051bb0c0` | 否 | `None` | — | — |

## experiments/mode_conditioned_skel/weights/  （2 个，8,938 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `probe_s42.pt` | 4,469 | `3fbf194e43711c0b` | 否 | `None` | — | — |
| `probe_s43.pt` | 4,469 | `0d25f07163fcf64f` | 否 | `None` | — | — |

## experiments/relation_bg/  （1 个，2,541,927 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `relation_repaired.pt` | 2,541,927 | `5418edd34abf862e` | 否 | `None` | — | base=checkpoints/base_encoder.pt |

## experiments/select_pool/cache/  （1 个，853,947,901 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `clean_28988c64fa91_tok_ctx64_cand32.pt` | 853,947,901 | `2fc1fc30f157e576` | 否 | `None` | — | — |

## experiments/select_pool/weights/  （16 个，10,573,854 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `A_s42.pt` | 660,029 | `a9abdb471cce52e6` | 否 | `None` | — | — |
| `A_s42_rand.pt` | 660,409 | `ce8055461bc6002e` | 否 | `None` | — | — |
| `A_s43.pt` | 660,029 | `c32ef68a0346e502` | 否 | `None` | — | — |
| `Aplus_s42.pt` | 661,993 | `a9767309a7c12790` | 否 | `None` | — | — |
| `Aplus_s42_rand.pt` | 662,063 | `460192c416edb3ff` | 否 | `None` | — | — |
| `Aplus_s43.pt` | 661,993 | `c73a8967c359560d` | 否 | `None` | — | — |
| `B_s42.pt` | 660,029 | `9e4996399284a4c0` | 否 | `None` | — | — |
| `B_s42_rand.pt` | 660,409 | `a71fc45c70795a07` | 否 | `None` | — | — |
| `B_s43.pt` | 660,029 | `e7bdc1a351a4a7c3` | 否 | `None` | — | — |
| `C_s42.pt` | 661,489 | `3e0d3de6eddbae71` | 否 | `None` | — | — |
| `C_s42_rand.pt` | 662,007 | `f86ee9d777f4f2c9` | 否 | `None` | — | — |
| `C_s43.pt` | 661,489 | `8e0efa41c25f6e0c` | 否 | `None` | — | — |
| `D_s42.pt` | 660,279 | `08a6cd4a9ac0307c` | 否 | `None` | — | — |
| `D_s42_rand.pt` | 660,664 | `2cf6aca83a10af8f` | 否 | `None` | — | — |
| `D_s43.pt` | 660,279 | `850d55a58249be14` | 否 | `None` | — | — |
| `D_s43_rand.pt` | 660,664 | `7283b95f80e68ae0` | 否 | `None` | — | — |

## experiments/select_rerank/cache/  （2 个，40,357,432 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `clean_28988c64fa91_ctx64_cand32.pt` | 20,178,701 | `e7db07dfe31f5211` | 否 | `None` | — | — |
| `shortcut_1b8f8106558a_ctx64_cand32.pt` | 20,178,731 | `71e425dfb540eb43` | 否 | `None` | — | — |

## experiments/select_rerank/weights/  （8 个，5,283,560 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `clean_s42.pt` | 660,397 | `8365204fd254b138` | 否 | `None` | — | — |
| `clean_s42_rand.pt` | 660,457 | `7a77464670110840` | 否 | `None` | — | — |
| `clean_s43.pt` | 660,397 | `0514504d385e2a07` | 否 | `None` | — | — |
| `clean_s43_rand.pt` | 660,457 | `68cfdb20bca7c4a7` | 否 | `None` | — | — |
| `shortcut_s42.pt` | 660,433 | `d439185b5e11e749` | 否 | `None` | — | — |
| `shortcut_s42_rand.pt` | 660,493 | `05b96f0da353f157` | 否 | `None` | — | — |
| `shortcut_s43.pt` | 660,433 | `ebfdf2b6eb23e4a2` | 否 | `None` | — | — |
| `shortcut_s43_rand.pt` | 660,493 | `9cbe149da7666225` | 否 | `None` | — | — |

## experiments/select_semantic_joint/weights/  （9 个，66,798,805 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `frozen_s42.pt` | 7,422,005 | `2ecc08faa11436a7` | 否 | `None` | — | — |
| `frozen_s42_rand.pt` | 7,422,195 | `55f0fb35460e16f7` | 否 | `None` | — | — |
| `frozen_s43.pt` | 7,422,005 | `81cc7e73bc7c4ea3` | 否 | `None` | — | — |
| `frozen_s43_rand.pt` | 7,422,195 | `b41d6da065b3ed31` | 否 | `None` | — | — |
| `joint_s42.pt` | 7,421,967 | `ec674ac292771760` | 否 | `None` | — | — |
| `joint_s42_rand.pt` | 7,422,157 | `553aec913992144d` | 否 | `None` | — | — |
| `joint_s43.pt` | 7,421,967 | `fb30e785664e93ce` | 否 | `None` | — | — |
| `joint_s43_rand.pt` | 7,422,157 | `a30ccca335878634` | 否 | `None` | — | — |
| `jointprobe_s42.pt` | 7,422,157 | `47106e40139857a6` | 否 | `None` | — | — |

## experiments/skeleton_leak/cache/  （6 个，279,964,636 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `enc_17f00c349a76_L64.pt` | 36,856,826 | `6bb2cd25db0d9eaf` | 否 | `None` | — | — |
| `enc_4169598e4af7_L64.pt` | 88,593,338 | `c54618f4bcd15d00` | 否 | `None` | — | — |
| `enc_58b315d8f6ba_L64.pt` | 39,691,706 | `1994916e0193a27e` | 否 | `None` | — | — |
| `enc_cbcab631ffcd_L64.pt` | 36,856,826 | `43b04414e825df24` | 否 | `None` | — | — |
| `enc_d3c4cb18cbe3_L64.pt` | 35,439,354 | `729619cc9950b09a` | 否 | `None` | — | — |
| `enc_eb7c161971a2_L64.pt` | 42,526,586 | `e153889950b5f11d` | 否 | `None` | — | — |

## experiments/struct_supervision/cache/  （3 个，5,956,315 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `gen_5d0afaf586cc_L64_M4.pt` | 3,179,785 | `ea7b4d967dfe9144` | 否 | `None` | — | — |
| `gen_a1ea86699c61_L64_M4.pt` | 169,609 | `5a68a2b1f27338db` | 否 | `None` | — | — |
| `gen_d3c4cb18cbe3_L64_M4.pt` | 2,606,921 | `4dbb8f8d9add60cd` | 否 | `None` | — | — |

## experiments/struct_supervision/weights/  （13 个，63,262,628 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `A_s42.pt` | 4,865,720 | `3f6474348319280e` | 否 | `None` | — | — |
| `A_s42_rand.pt` | 4,866,869 | `1fea3e5ab29ad662` | 否 | `None` | — | — |
| `A_s42_steps60.pt` | 4,866,944 | `c0b64ee0702f9666` | 否 | `None` | — | — |
| `A_s43.pt` | 4,865,720 | `ecb3bf1a2f8abe50` | 否 | `None` | — | — |
| `A_s43_rand.pt` | 4,866,869 | `7e7591909ea0d5a2` | 否 | `None` | — | — |
| `B_s42.pt` | 4,865,720 | `99c5dfd6cad9e7cc` | 否 | `None` | — | — |
| `B_s42_rand.pt` | 4,866,869 | `e198a940aac9d1d3` | 否 | `None` | — | — |
| `B_s42_steps60.pt` | 4,866,944 | `0caefd34d9158f56` | 否 | `None` | — | — |
| `B_s43.pt` | 4,865,720 | `d59cabd922c050f7` | 否 | `None` | — | — |
| `B_s43_rand.pt` | 4,866,869 | `464a9511ae58af29` | 否 | `None` | — | — |
| `C_s42.pt` | 4,865,720 | `98fcd5253811fd4b` | 否 | `None` | — | — |
| `C_s42_steps60.pt` | 4,866,944 | `4ad1caaa0fd37b38` | 否 | `None` | — | — |
| `C_s43.pt` | 4,865,720 | `96fffda74a1b4795` | 否 | `None` | — | — |

## experiments/syllogism_card/weights/  （10 个，10,983,866 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `A_s42.pt` | 1,097,833 | `a1b915582ed9722a` | 否 | `None` | — | — |
| `A_s43.pt` | 1,097,833 | `9873edf676af11f1` | 否 | `None` | — | — |
| `TA_s42.pt` | 1,097,861 | `058f7304af4cdfb8` | 否 | `None` | — | — |
| `TA_s42_rand.pt` | 1,099,217 | `b537de7a3b9c6823` | 否 | `None` | — | — |
| `TA_s43.pt` | 1,097,861 | `4d31dc3deae21f77` | 否 | `None` | — | — |
| `TA_s43_rand.pt` | 1,099,217 | `74fdf6bce75ee865` | 否 | `None` | — | — |
| `T_s42.pt` | 1,097,833 | `b42b84fc0a1fa604` | 否 | `None` | — | — |
| `T_s42_rand.pt` | 1,099,189 | `b40b842c4bd3fc6c` | 否 | `None` | — | — |
| `T_s43.pt` | 1,097,833 | `1d511b477438dc00` | 否 | `None` | — | — |
| `T_s43_rand.pt` | 1,099,189 | `a8f66aec26aa332a` | 否 | `None` | — | — |

## experiments/two_channel_head/cache/  （2 个，27,347,858 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `gen_4169598e4af7_L64_M4.pt` | 6,512,905 | `53b4938f62deeb43` | 否 | `None` | — | — |
| `gen_bac00ff05e35_L64_M4.pt` | 20,834,953 | `82d77ef3d6e03407` | 否 | `None` | — | — |

## experiments/two_channel_head/weights/  （16 个，14,168,260 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `A_s42.pt` | 660,029 | `a6b8ea8a4d822999` | 否 | `None` | — | — |
| `A_s42_rand.pt` | 660,473 | `8030599caf739710` | 否 | `None` | — | — |
| `A_s42_steps30.pt` | 660,509 | `7aecaa10e14fc0ec` | 否 | `None` | — | — |
| `A_s42_steps60.pt` | 660,509 | `5fa250d95c6e7a9a` | 否 | `None` | — | — |
| `A_s43.pt` | 660,029 | `b26ad3a89aa64bdf` | 否 | `None` | — | — |
| `B_s42.pt` | 747,627 | `11f4a2998c72ed5c` | 否 | `None` | — | — |
| `B_s42_rand.pt` | 748,214 | `7b2341f4f571b2e4` | 否 | `None` | — | — |
| `B_s42_steps30.pt` | 748,259 | `67824a173b909f52` | 否 | `None` | — | — |
| `B_s43.pt` | 747,627 | `e917a67d89018103` | 否 | `None` | — | — |
| `C_s42.pt` | 748,960 | `1317e62bf89bcfc0` | 否 | `None` | — | — |
| `C_s42_steps30.pt` | 749,672 | `c73c118aae01d422` | 否 | `None` | — | — |
| `C_s43.pt` | 748,960 | `8b4b970d8ef112c6` | 否 | `None` | — | — |
| `D_s42.pt` | 1,406,348 | `7d67fc0b727613ad` | 否 | `None` | — | — |
| `D_s42_steps30.pt` | 1,407,348 | `5c7bf55160351e83` | 否 | `None` | — | — |
| `D_s42_steps60.pt` | 1,407,348 | `57dd4d3b213442b4` | 否 | `None` | — | — |
| `D_s43.pt` | 1,406,348 | `b2a3cda11d2c575a` | 否 | `None` | — | — |

## experiments/value_card/cards/  （4 个，10,165,344 字节）
| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |
|---|---:|---|---|---|---|---|
| `shuf_s42.pt` | 2,541,305 | `bc7c0bcd5436454b` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `shuf_s43.pt` | 2,541,305 | `64a4b730a07dba72` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `value_s42.pt` | 2,541,367 | `f1084e9d52fc2303` | 否 | `None` | — | base=checkpoints/base_encoder.pt |
| `value_s43.pt` | 2,541,367 | `813afb4eb1504376` | 否 | `None` | — | base=checkpoints/base_encoder.pt |


---

# 汇总（实测）

- `.pt` 总数：扫描 **262** / 清单 **262** / 差 **0**
- 总字节：**3,380,031,533 B**（3.15 GiB）；其中 `/cache/` 目录 31 个（已单独标注）
- 与线上 base：一致 **5** / 不一致 **58** / 不适用 **199**（加载失败 0）
- 内含 `doc_encoder` 的产物：**63**；纯卡/解码头/缓存：**199**
- 物理副本：16 个文件落在 8 组同 sha 组内；**全仓 sha 唯一（真单副本）246 个**

## 单副本风险清单

**A 类（无任何校验和有记录）**：253 / 262。本清单之前，全仓文本中能找到的 .pt 文件级 sha256 只有 7 条 （a1_shas.json 收录：`checkpoints/base_encoder.pt`, `checkpoints/cards/person.pt`, `checkpoints/cards/pronoun.pt`, `checkpoints/cards/relation.pt`, `checkpoints/cards/sentiment.pt`, `experiments/compose_ops/artifacts/base_encoder.pt`, `experiments/compose_ops/artifacts/cards/negation.pt`），另有 2 个只有 doc_sha 记录；其余 **253** 个此前**无任何校验和记录**，其中 **237** 个连同 sha 的第二份拷贝都没有。

**B 类（单副本 + 被 ≥2 个实验/模块引用）**：26 个（引用统计已排除 `experiments/card_catalog/`，因为它逐条列了每个文件，会灌水）。引用数 = 提到该文件名的不同文件数（`.py/.sh/.md/.json`）。

| 路径 | 引用文件数 | 引用方（实验/模块数） | 副本 |
|---|---:|---|---|
| `checkpoints/base_encoder.pt` | 162 | 35：`dev-notes`, `experiments/additivity`, `experiments/adversarial_routing`, `experiments/anchored_select`, `experiments/capability_map`, `experiments/compose_ops` … | 1 |
| `experiments/compose_ops/artifacts/base_encoder.pt` | 162 | 35：`dev-notes`, `experiments/additivity`, `experiments/adversarial_routing`, `experiments/anchored_select`, `experiments/capability_map`, `experiments/compose_ops` … | 1 |
| `checkpoints/cards/sentiment.pt` | 21 | 9：`experiments/compose_ops`, `experiments/core_probe`, `experiments/e2e_d1`, `experiments/mode_conditioned_skel`, `experiments/out_invariants`, `experiments/prod_card_audit` … | 1 |
| `checkpoints/multitask_v2_dtseek.pt` | 14 | 9：`README.md`, `dev-notes`, `experiments/adversarial_routing`, `experiments/core_keep`, `experiments/ndb_recheck`, `experiments/prod_card_audit` … | 1 |
| `experiments/compose_ops/artifacts/cards/sentiment.pt` | 21 | 9：`experiments/compose_ops`, `experiments/core_probe`, `experiments/e2e_d1`, `experiments/mode_conditioned_skel`, `experiments/out_invariants`, `experiments/prod_card_audit` … | 1 |
| `checkpoints/cards/person.pt` | 16 | 8：`dev-notes`, `experiments/e2e_d1`, `experiments/mode_conditioned_skel`, `experiments/out_invariants`, `experiments/prod_card_audit`, `experiments/sentence_mode` … | 1 |
| `checkpoints/cards/pronoun.pt` | 16 | 8：`experiments/e2e_d1`, `experiments/mode_conditioned_skel`, `experiments/out_invariants`, `experiments/prod_card_audit`, `experiments/rank_budget`, `experiments/sentence_mode` … | 1 |
| `checkpoints/cards/relation.pt` | 16 | 8：`experiments/e2e_d1`, `experiments/mode_conditioned_skel`, `experiments/ndb_global`, `experiments/out_invariants`, `experiments/prod_card_audit`, `experiments/sentence_mode` … | 1 |
| `experiments/compose_ops/artifacts/cards/person.pt` | 16 | 8：`dev-notes`, `experiments/e2e_d1`, `experiments/mode_conditioned_skel`, `experiments/out_invariants`, `experiments/prod_card_audit`, `experiments/sentence_mode` … | 1 |
| `experiments/compose_ops/artifacts/cards/pronoun.pt` | 16 | 8：`experiments/e2e_d1`, `experiments/mode_conditioned_skel`, `experiments/out_invariants`, `experiments/prod_card_audit`, `experiments/rank_budget`, `experiments/sentence_mode` … | 1 |
| `experiments/compose_ops/artifacts/cards/relation.pt` | 16 | 8：`experiments/e2e_d1`, `experiments/mode_conditioned_skel`, `experiments/ndb_global`, `experiments/out_invariants`, `experiments/prod_card_audit`, `experiments/sentence_mode` … | 1 |
| `checkpoints/negation_accept_card.pt` | 14 | 6：`experiments/card_flow`, `experiments/compose_ops`, `experiments/counterfactual`, `experiments/gen_dispatch`, `experiments/pointer_explain`, `experiments/skeleton_two_tier` | 1 |
| `experiments/compose_ops/artifacts/cards/negation.pt` | 11 | 5：`dev-notes`, `experiments/compose_ops`, `experiments/e2e_d1`, `experiments/prod_card_audit`, `src` | 1 |
| `checkpoints/arm_neg5_seed42.pt` | 6 | 4：`dev-notes`, `experiments/adversarial_routing`, `experiments/compose_ops`, `experiments/prod_card_audit` | 1 |
| `checkpoints/arm_aprime_dtseek.pt` | 3 | 3：`dev-notes`, `experiments/additivity`, `experiments/negation_card` | 1 |
| `checkpoints/negation_accept_card_e12.pt` | 8 | 3：`dev-notes`, `experiments/compose_ops`, `experiments/out_invariants` | 1 |
| `experiments/anchored_select/cards/joint_s42.pt` | 13 | 3：`experiments/anchored_select`, `experiments/core_grad_probe`, `experiments/core_update_probe` | 1 |
| `experiments/anchored_select/cards/shuf_s42.pt` | 7 | 3：`experiments/anchored_select`, `experiments/gen_data_loop`, `experiments/value_card` | 1 |
| `experiments/anchored_select/cards/shuf_s43.pt` | 4 | 3：`experiments/anchored_select`, `experiments/gen_data_loop`, `experiments/value_card` | 1 |
| `experiments/select_semantic_joint/weights/joint_s42.pt` | 13 | 3：`experiments/anchored_select`, `experiments/core_grad_probe`, `experiments/core_update_probe` | 1 |
| `experiments/value_card/cards/shuf_s42.pt` | 7 | 3：`experiments/anchored_select`, `experiments/gen_data_loop`, `experiments/value_card` | 1 |
| `experiments/value_card/cards/shuf_s43.pt` | 4 | 3：`experiments/anchored_select`, `experiments/gen_data_loop`, `experiments/value_card` | 1 |
| `experiments/cumulative_add/cards/ctrl_idiom_s42.pt` | 8 | 2：`dev-notes`, `experiments/gen_dispatch` | 1 |
| `experiments/gen_dispatch/cache/idiom_card.pt` | 2 | 2：`dev-notes`, `experiments/gen_dispatch` | 1 |
| `experiments/select_pool/cache/clean_28988c64fa91_tok_ctx64_cand32.pt` | 2 | 2：`experiments/select_pool`, `experiments/select_semantic_joint` | 1 |
| `experiments/select_rerank/cache/clean_28988c64fa91_ctx64_cand32.pt` | 18 | 2：`experiments/select_pool`, `experiments/two_channel_head` | 1 |

**再生证据（实测：同实验目录 `.py`/`.sh` 是否出现该文件名）**：有产出脚本命中 **16**；在 `cache/` 下（由该实验数据准备脚本重生成）**31**；同实验目录无脚本命中 **215**（= 无再生证据，删了只能靠外部备份）。

## D1 命中清单（卡内 doc_encoder ≠ 线上 base，逐条 neq/26 与 max|Δ|）

共 **58** 条；与 `experiments/card_catalog/scan_raw.json` 逐条对账：**58 条一致 / 0 条不一致**。

| 文件 | neq/26 | max\|Δ\| | card_catalog neq | card_catalog maxd | 对账 |
|---|---:|---:|---:|---:|---|
| `checkpoints/arm_a2_seed43.pt` | 26/26 | 0.281995 | 26 | 0.281995 | 一致 |
| `experiments/core_keep/cards/J1c_s43.pt` | 26/26 | 0.269304 | 26 | 0.269304 | 一致 |
| `checkpoints/arm_neg5_seed43.pt` | 26/26 | 0.240070 | 26 | 0.240070 | 一致 |
| `experiments/cumulative_add/cards/step3_s43.pt` | 26/26 | 0.130666 | 26 | 0.130666 | 一致 |
| `experiments/cumulative_add/cards/step3_s42.pt` | 26/26 | 0.125700 | 26 | 0.125700 | 一致 |
| `experiments/core_generalize/cores/C1x5_s42.pt` | 26/26 | 0.119661 | 26 | 0.119661 | 一致 |
| `checkpoints/arm_neg5_seed42.pt` | 26/26 | 0.112020 | 26 | 0.112020 | 一致 |
| `experiments/compose_ops/artifacts/base_encoder.pt` | 26/26 | 0.112020 | 26 | 0.112020 | 一致 |
| `experiments/core_keep/cards/F_s42.pt` | 26/26 | 0.112020 | 26 | 0.112020 | 一致 |
| `checkpoints/e5b_single_relation.pt` | 26/26 | 0.101223 | 26 | 0.101223 | 一致 |
| `checkpoints/e5c_single_relation.pt` | 26/26 | 0.101223 | 26 | 0.101223 | 一致 |
| `experiments/core_keep/cards/J1c_s42.pt` | 26/26 | 0.095022 | 26 | 0.095022 | 一致 |
| `checkpoints/e5_single_pronoun.pt` | 26/26 | 0.093632 | 26 | 0.093632 | 一致 |
| `checkpoints/e5b_single_pronoun.pt` | 26/26 | 0.093632 | 26 | 0.093632 | 一致 |
| `checkpoints/e5c_single_pronoun.pt` | 26/26 | 0.093632 | 26 | 0.093632 | 一致 |
| `checkpoints/e5b_single_sentiment.pt` | 26/26 | 0.092483 | 26 | 0.092483 | 一致 |
| `checkpoints/e5c_single_sentiment.pt` | 26/26 | 0.092483 | 26 | 0.092483 | 一致 |
| `experiments/cumulative_add/cards/step2_s43.pt` | 26/26 | 0.090971 | 26 | 0.090971 | 一致 |
| `checkpoints/e5b_single_person.pt` | 26/26 | 0.087153 | 26 | 0.087153 | 一致 |
| `checkpoints/e5c_single_person.pt` | 26/26 | 0.087153 | 26 | 0.087153 | 一致 |
| `checkpoints/reply_pick_joint.pt` | 26/26 | 0.086623 | 26 | 0.086623 | 一致 |
| `experiments/cumulative_add/cards/step2_s42.pt` | 26/26 | 0.085453 | 26 | 0.085453 | 一致 |
| `checkpoints/smoke_test.pt` | 26/26 | 0.078725 | 26 | 0.078725 | 一致 |
| `checkpoints/arm_b_freeze_base_dtseek.pt` | 26/26 | 0.078565 | 26 | 0.078565 | 一致 |
| `checkpoints/arm_lrbase1e4_5card.pt` | 26/26 | 0.078450 | 26 | 0.078450 | 一致 |
| `experiments/core_generalize/cores/PRE_idiom_s42.pt` | 26/26 | 0.072498 | 26 | 0.072498 | 一致 |
| `experiments/core_generalize/cores/PRE_idiom_s43.pt` | 26/26 | 0.072323 | 26 | 0.072323 | 一致 |
| `checkpoints/arm_c_warmup4_freeze_dtseek.pt` | 26/26 | 0.069620 | 26 | 0.069620 | 一致 |
| `checkpoints/arm_aprime_dtseek.pt` | 26/26 | 0.068806 | 26 | 0.068806 | 一致 |
| `checkpoints/arm_remedy_frozen_add_neg.pt` | 26/26 | 0.068806 | 26 | 0.068806 | 一致 |
| `experiments/anchored_select/cards/joint_s42.pt` | 26/26 | 0.055426 | 26 | 0.055426 | 一致 |
| `experiments/cumulative_add/cards/ctrl_idiom_s43.pt` | 26/26 | 0.053217 | 26 | 0.053217 | 一致 |
| `experiments/core_generalize/cores/C3_s42.pt` | 26/26 | 0.049445 | 26 | 0.049445 | 一致 |
| `experiments/core_generalize/cores/C1_s43.pt` | 26/26 | 0.047952 | 26 | 0.047952 | 一致 |
| `experiments/core_generalize/cores/PRE_ownership_s43.pt` | 26/26 | 0.047588 | 26 | 0.047588 | 一致 |
| `experiments/cumulative_add/cards/ctrl_ownership_s43.pt` | 26/26 | 0.047427 | 26 | 0.047427 | 一致 |
| `experiments/cumulative_add/cards/step1_s43.pt` | 26/26 | 0.047286 | 26 | 0.047286 | 一致 |
| `experiments/anchored_select/cards/joint_s43.pt` | 26/26 | 0.045897 | 26 | 0.045897 | 一致 |
| `experiments/core_keep/cards/J3_s42.pt` | 26/26 | 0.045032 | 26 | 0.045032 | 一致 |
| `experiments/core_keep/cards/J3_s43.pt` | 26/26 | 0.044621 | 26 | 0.044621 | 一致 |
| `experiments/cumulative_add/cards/ctrl_idiom_s42.pt` | 26/26 | 0.043752 | 26 | 0.043752 | 一致 |
| `experiments/gen_dispatch/cache/idiom_base.pt` | 26/26 | 0.043752 | 26 | 0.043752 | 一致 |
| `experiments/core_keep/cards/J0_s43.pt` | 26/26 | 0.042691 | 26 | 0.042691 | 一致 |
| `experiments/core_generalize/cores/C5_s42.pt` | 26/26 | 0.042646 | 26 | 0.042646 | 一致 |
| `experiments/core_generalize/cores/C3_s43.pt` | 26/26 | 0.042015 | 26 | 0.042015 | 一致 |
| `experiments/cumulative_add/cards/step1_s42.pt` | 26/26 | 0.041425 | 26 | 0.041425 | 一致 |
| `experiments/core_generalize/cores/C5_s43.pt` | 26/26 | 0.040750 | 26 | 0.040750 | 一致 |
| `experiments/core_generalize/cores/C1_s42.pt` | 26/26 | 0.040459 | 26 | 0.040459 | 一致 |
| `experiments/core_generalize/cores/PRE_ownership_s42.pt` | 26/26 | 0.040419 | 26 | 0.040419 | 一致 |
| `experiments/core_keep/cards/J2_s42.pt` | 26/26 | 0.039991 | 26 | 0.039991 | 一致 |
| `experiments/core_keep/cards/J1l16_s42.pt` | 26/26 | 0.038428 | 26 | 0.038428 | 一致 |
| `experiments/core_keep/cards/J0_s42.pt` | 26/26 | 0.038053 | 26 | 0.038053 | 一致 |
| `experiments/cumulative_add/cards/ctrl_ownership_s42.pt` | 26/26 | 0.037052 | 26 | 0.037052 | 一致 |
| `experiments/core_keep/cards/J1_s42.pt` | 26/26 | 0.036138 | 26 | 0.036138 | 一致 |
| `experiments/core_keep/cards/J2_s43.pt` | 26/26 | 0.035948 | 26 | 0.035948 | 一致 |
| `experiments/core_keep/cards/J1_s43.pt` | 26/26 | 0.035399 | 26 | 0.035399 | 一致 |
| `experiments/core_keep/cards/J1l4_s42.pt` | 26/26 | 0.035148 | 26 | 0.035148 | 一致 |
| `checkpoints/arm_c12_warmup12_freeze_dtseek.pt` | 26/26 | 0.034431 | 26 | 0.034431 | 一致 |

与线上一致的 5 个：`checkpoints/arm_cprime_headonly_dtseek.pt`、`checkpoints/base_encoder.pt`、`checkpoints/multitask_v2_dtseek.pt`、`experiments/cumulative_add/cards/gate0_s42.pt`、`experiments/cumulative_add/cards/gate0_s43.pt`

## 与既有记录对账

- `prod_card_audit/results/a1_shas.json`：`base_encoder.pt` doc_sha、`multitask_v2_dtseek.pt` doc_sha、`compose_ops/artifacts/base_encoder.pt` doc_sha、`arm_neg5_seed42.pt` doc_sha、`base_encoder.pt` file_sha —— **5/5 与本次一致**。
- `card_catalog/scan_raw.json`：现存 262 条全部在该 337 扫描集内（新增/改名 = 0）；58 条不一致的 neq/max|Δ| **逐条一致**。
- `dev-notes/21` 两例：`ctrl_idiom_s42` 本次 max|Δ|=0.043752（dev-notes 记 .0438）**一致**；`e5c_single_sentiment.pt` 本次 neq=26/26、max|Δ|=0.092483（CATALOG:278 记 0.0925；dev-notes/21:324 记 .0925）**一致**。
- 「不一致」是**文件内自带 doc_encoder 与线上基座不同**；另有 5 个 v1 卡不带 encoder（如 negation.pt），其基座由 `train_args.split_from` 指向 `arm_neg5_seed42.pt`（doc_sha `801657fb…` ≠ 线上 `dc27db53…`）⇒ 同为 D1 命中，但按口径归入「不适用（卡内无 encoder）」，其在产命中由 `prod_card_audit` 记录。

## 来源线索（卡内 train_args，实测读取）

| 类别 | 个数 | 说明 |
|---|---:|---|
| `train_args.base = checkpoints/base_encoder.pt` | 39 | 卡声明用**线上基座**，与 D1 一致 |
| `train_args.split_from = checkpoints/multitask_v2_dtseek.pt` | 4 | multitask_v2 与线上**逐位相同**（doc_sha 同 `dc27db53…`）⇒ 溯源一致 |
| `train_args.split_from = checkpoints/arm_neg5_seed42.pt` | 5 | 来源核 doc_sha `801657fb…` **≠ 线上 `dc27db53…`** ⇒ **溯源 D1 命中**（见下） |
| 无 `train_args`（OLD_CKPT / ENC / HEAD / CACHE） | 214 | 自带 encoder 或与基座无关 |

**溯源 D1 命中（split_from 核 ≠ 线上，5 条）**：

| 卡 | split_from | split_from doc_sha16 | 线上 doc_sha16 |
|---|---|---|---|
| `experiments/compose_ops/artifacts/cards/negation.pt` | `checkpoints/arm_neg5_seed42.pt` | `801657fb40a803a2` | `dc27db5337d16f18` |
| `experiments/compose_ops/artifacts/cards/person.pt` | `checkpoints/arm_neg5_seed42.pt` | `801657fb40a803a2` | `dc27db5337d16f18` |
| `experiments/compose_ops/artifacts/cards/pronoun.pt` | `checkpoints/arm_neg5_seed42.pt` | `801657fb40a803a2` | `dc27db5337d16f18` |
| `experiments/compose_ops/artifacts/cards/relation.pt` | `checkpoints/arm_neg5_seed42.pt` | `801657fb40a803a2` | `dc27db5337d16f18` |
| `experiments/compose_ops/artifacts/cards/sentiment.pt` | `checkpoints/arm_neg5_seed42.pt` | `801657fb40a803a2` | `dc27db5337d16f18` |

> `experiments/compose_ops/artifacts/cards/negation.pt` 是 `DEFAULT_ATTACH`（`dialogue.py:105`），即**在产路径**；与 `prod_card_audit/a1_shas.json` 的 `card_provenance_core_doc_sha256=801657fb…` **对账一致**。
