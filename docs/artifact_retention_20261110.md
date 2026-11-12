# 产物保留与清理候选清单（2026-11-10）

生成时间：2026-11-10T23:36:39.035668+08:00。项目根目录：`/projects/_hdd/isaachdd/EWM`。

本文件只列清理候选，没有执行删除。清单逐一覆盖 `results/`、`tmp/`、`logs/` 的所有一级条目；`review` 一律先保留，不能当成删除批准。后续出现的新目录不在本清单内。机器可读版本见 [artifact_retention_20261110.json](artifact_retention_20261110.json)。

## 必须保留的最终实验

五模型的最后 ID 和全部当前协议 OOD（光照、模糊、青灰背景、速度）均保留完整视频、预测视频、轨迹、逐回合 JSON、初始状态、场景/源码指纹和统计；不只保留 summary。背景和速度尚有运行中/待运行单元，预生成场景清单也必须保留。

| 模型 | 最终 ID 来源 | 已核验结果 |
| --- | --- | --- |
| action_only | `results/piper_action_eval_262022_step080000_continuous_min8_5090_20261105/id`<br>`results/piper_action_eval_262022_step080000_continuous_min8_error_record_5090_20261106_v2/id` | 152/400 (38.00%) |
| rgb_action | `results/piper_rgb_action_eval_262022_step080000_continuous_min8_error_record_5090_20261106/id` | 153/400 (38.25%) |
| ewm_stride2 | `results/piper_ewm_eval_262022_step080000_continuous_min8_error_record_5090_20261106_v2/id` | 210/400 (52.50%) |
| ewm_stride1 | `results/piper_ewm_eval_276490_step080000_continuous_min8_decode_pro6000_20261107/id` | 213/400 (53.25%) |
| ewm_asymmetric | `results/piper_ewm_eval_243038_step080000_continuous_min8_5090_20261106/id` | 233/400 (58.25%) |

Action-only 的 ID 必须合并两处：原 min8 目录的 **conveyor、turntable_cups、rotating_tube、tennis_rolling、tennis_falling** 共120/250；v2目录的 **moving_carton、tennis_parabolic、tennis_container** 共32/150。合计152/400=38%；不能只保留“最新v2”，也不能误按任务列表前5/后3划分。

| 模型 | 光照 OOD | 模糊 OOD | 背景 OOD | 速度 OOD |
| --- | --- | --- | --- | --- |
| action_only | `results/piper_three_ood_262022_step080000_continuous_min8_decode_5090_20261107/action_only/ood` | `results/piper_three_ood_262022_step080000_continuous_min8_decode_5090_20261107/action_only/ood` | `results/piper_action_eval_262022_step080000_background_greygreen_min8_5090_20261109/ood` | `results/piper_five_speed_ood_step080000_continuous_min8_decode_5090_20261110/action_only/ood` |
| rgb_action | `results/piper_three_ood_262022_step080000_continuous_min8_decode_5090_20261107/rgb_action/ood` | `results/piper_three_ood_262022_step080000_continuous_min8_decode_5090_20261107/rgb_action/ood` | `results/piper_three_background_greygreen_recovery_5090_20261110/rgb_action/ood` | `results/piper_five_speed_ood_step080000_continuous_min8_decode_5090_20261110/rgb_action/ood` |
| ewm_stride2 | `results/piper_three_ood_262022_step080000_continuous_min8_decode_5090_20261107/joint/ood` | `results/piper_three_ood_262022_step080000_continuous_min8_decode_5090_20261107/joint/ood` | `results/piper_ewm_eval_262022_step080000_background_greygreen_min8_decode_pro6000_20261109/ood` | `results/piper_five_speed_ood_step080000_continuous_min8_decode_5090_20261110/ewm_stride2/ood` |
| ewm_stride1 | `results/piper_ewm_eval_276490_step080000_continuous_min8_decode_pro6000_20261107/lighting_ood` | `results/piper_ewm_eval_276490_step080000_continuous_min8_decode_pro6000_20261107/blur_ood` | `results/piper_three_background_greygreen_recovery_5090_20261110/ewm_stride1/ood` | `results/piper_five_speed_ood_step080000_continuous_min8_decode_5090_20261110/ewm_stride1/ood` |
| ewm_asymmetric | `results/piper_ewm_eval_243038_step080000_continuous_min8_decode_pro6000_20261108/ood` | `results/piper_ewm_eval_243038_step080000_continuous_min8_decode_pro6000_20261108/ood` | `results/piper_three_background_greygreen_recovery_5090_20261110/ewm_asymmetric/ood` | `results/piper_five_speed_ood_step080000_continuous_min8_decode_5090_20261110/ewm_asymmetric/ood` |

同一模型的光照、模糊路径相同时，条件位于各任务下的 `lighting_ood/`、`blur_ood/`；间隔1 EWM 的条件各自是顶层套件。速度五组采用新建的统一套件；旧背景+速度混合目录中已有速度部分结果列为 review，在新套件完成并校验前不丢弃。

## 当前任务与模型保护

- Job1732584尚在运行；`tmp/background_resume_three_pro6000_20261110/` 是当前代码入口，不能删除或改路径。
- 任务末尾 `repair.py verify` 会重新验证旧新EWM背景的391个受保护回合：逐回合结果/steps的SHA和视频size/mtime。相关旧背景视频即使已完成也不能删、重编码或改名。
- 五个正式run目录完整保留，包括 `resume_state_step_080000/`。`checkpoint.pt` 与 `checkpoint_step_080000.pt` 是硬链接，删一个名称不等于释放一份模型空间。
- 快照里的datasets、pretrained、envs、isaacsim601、third_party为指向真实目录的符号链接，不能把它们当成多余副本。
- 当前与后续WAM数据入口 `tmp/training_data_cache/piper_rgb_dynamic_v1/`、环境、正式权重都保留。
- WAM两个吞吐目录的 `benchmark_probe.py` 还导入旧 `wam_random_serial_smoke_20261110/telemetry.py`，不能仅留最新benchmark目录。

五组正式模型目录：

- `tmp/ewm_formal_ada_serial_compile_20261104/runs/action_only`
- `tmp/ewm_formal_ada_serial_compile_20261104/runs/rgb_action`
- `tmp/ewm_formal_ada_serial_compile_20261104/runs/joint`
- `OpenWAM/outputs/ewm_joint_stride1_ada_20261106`
- `OpenWAM/outputs/ewm_eight_no_memory_no_alignment_243038`

## event / RGB 冗余度实验

以下条目完整保留，包含输入 `.npz`、拟合基底、脚本、报告、日志及图；不是临时垃圾：

- `tmp/event_rgb_latent_pilot_20261104`
- `tmp/latent_expanded_20261104`
- `tmp/latent_expanded_20261104_results.json`
- `tmp/latent_expanded_20261104_subsample.json`
- `tmp/latent_pca_decoder_20261104`
- `tmp/latent_pca_middle_20261104`
- `tmp/latent_pca_reconstruction_20261104`
- `tmp/latent_predictability_probe_20261104`
- `tmp/latent_statistics_visualization_20261104`
- `tmp/latent_visualization_20261104`

`docs/latent_representation_analysis_20261104/`、原始 `datasets/`、LeanVAE权重及共享normalization一并保留。此前已按用户授权删除的event_frames缓存，本轮不重建；统计分析可使用现存latent复现，重新编码和部分像素级对照需要先重建事件帧缓存。

## 条目统计

| 目录 | keep | candidate | review |
| --- | ---: | ---: | ---: |
| `results/` | 15 | 18 | 5 |
| `tmp/` | 54 | 7 | 215 |
| `logs/` | 21 | 13 | 253 |

`candidate` 是明确路径的候选，不含任何通配删除命令；路径互不嵌套，也不覆盖受保护目录。`review` 主要是历史采集、诊断、控制器/记忆消融，不能因名称较旧或位于tmp就删。保留项中仍可能存在可重建小缓存；本轮没有进一步拆删。

## results/：候选（18项）

| 路径 | 依据 |
| --- | --- |
| `results/piper_action_eval_224077_step080000` | 旧模型或旧执行协议的测评，已不在指定五模型最终ID/OOD集合中；仅列候选，不执行删除。 |
| `results/piper_action_eval_227797_step080000` | 旧模型或旧执行协议的测评，已不在指定五模型最终ID/OOD集合中；仅列候选，不执行删除。 |
| `results/piper_action_eval_234507_step080000` | 旧模型或旧执行协议的测评，已不在指定五模型最终ID/OOD集合中；仅列候选，不执行删除。 |
| `results/piper_action_eval_237659_step080000` | 旧模型或旧执行协议的测评，已不在指定五模型最终ID/OOD集合中；仅列候选，不执行删除。 |
| `results/piper_action_eval_239872_step080000` | 旧模型或旧执行协议的测评，已不在指定五模型最终ID/OOD集合中；仅列候选，不执行删除。 |
| `results/piper_action_eval_262022_step080000_continuous_5090_20261105` | 旧模型或旧执行协议的测评，已不在指定五模型最终ID/OOD集合中；仅列候选，不执行删除。 |
| `results/piper_action_eval_262022_step080000_continuous_min4_5090_20261105` | 旧模型或旧执行协议的测评，已不在指定五模型最终ID/OOD集合中；仅列候选，不执行删除。 |
| `results/piper_action_eval_262022_step080000_continuous_min8_error_record_5090_20261106` | 已核实无任何 trial_*_result.json，属于被替代的预检/占位目录；最终测评在保留目录。 |
| `results/piper_eight_eval_218011_step040000` | 旧模型或旧执行协议的测评，已不在指定五模型最终ID/OOD集合中；仅列候选，不执行删除。 |
| `results/piper_eight_eval_218011_step080000` | 旧模型或旧执行协议的测评，已不在指定五模型最终ID/OOD集合中；仅列候选，不执行删除。 |
| `results/piper_ewm_eval_262022_step080000_continuous_5090_20261105` | 旧模型或旧执行协议的测评，已不在指定五模型最终ID/OOD集合中；仅列候选，不执行删除。 |
| `results/piper_ewm_eval_262022_step080000_continuous_min4_5090_20261105` | 旧模型或旧执行协议的测评，已不在指定五模型最终ID/OOD集合中；仅列候选，不执行删除。 |
| `results/piper_ewm_eval_262022_step080000_continuous_min8_5090_20261105` | 旧模型或旧执行协议的测评，已不在指定五模型最终ID/OOD集合中；仅列候选，不执行删除。 |
| `results/piper_ewm_eval_262022_step080000_continuous_min8_error_record_5090_20261106` | 已核实无任何 trial_*_result.json，属于被替代的预检/占位目录；最终测评在保留目录。 |
| `results/piper_memory_eval_229899_step080000` | 旧模型或旧执行协议的测评，已不在指定五模型最终ID/OOD集合中；仅列候选，不执行删除。 |
| `results/piper_no_memory_eval_229910_step080000` | 旧模型或旧执行协议的测评，已不在指定五模型最终ID/OOD集合中；仅列候选，不执行删除。 |
| `results/piper_three_background_greygreen_after277129_pro6000_20261109` | 已核实无任何 trial_*_result.json，属于被替代的预检/占位目录；最终测评在保留目录。 |
| `results/piper_three_background_greygreen_recovery_pro6000_20261110` | 已核实无任何 trial_*_result.json，属于被替代的预检/占位目录；最终测评在保留目录。 |

## results/：保留（15项）

| 路径 | 依据 |
| --- | --- |
| `results/piper_action_eval_262022_step080000_background_greygreen_min8_5090_20261109` | 五模型最终ID/OOD的唯一或组合来源，或用户已制作的论文展示图片；保留完整内容。 |
| `results/piper_action_eval_262022_step080000_continuous_min8_5090_20261105` | 五模型最终ID/OOD的唯一或组合来源，或用户已制作的论文展示图片；保留完整内容。 |
| `results/piper_action_eval_262022_step080000_continuous_min8_error_record_5090_20261106_v2` | 五模型最终ID/OOD的唯一或组合来源，或用户已制作的论文展示图片；保留完整内容。 |
| `results/piper_conveyor_wrist_events_4x1_20261106` | 五模型最终ID/OOD的唯一或组合来源，或用户已制作的论文展示图片；保留完整内容。 |
| `results/piper_eight_mainview_2x4_20261106` | 五模型最终ID/OOD的唯一或组合来源，或用户已制作的论文展示图片；保留完整内容。 |
| `results/piper_ewm_eval_243038_step080000_continuous_min8_5090_20261106` | 五模型最终ID/OOD的唯一或组合来源，或用户已制作的论文展示图片；保留完整内容。 |
| `results/piper_ewm_eval_243038_step080000_continuous_min8_decode_pro6000_20261108` | 五模型最终ID/OOD的唯一或组合来源，或用户已制作的论文展示图片；保留完整内容。 |
| `results/piper_ewm_eval_262022_step080000_background_greygreen_min8_decode_pro6000_20261109` | 五模型最终ID/OOD的唯一或组合来源，或用户已制作的论文展示图片；保留完整内容。 |
| `results/piper_ewm_eval_262022_step080000_continuous_min8_error_record_5090_20261106_v2` | 五模型最终ID/OOD的唯一或组合来源，或用户已制作的论文展示图片；保留完整内容。 |
| `results/piper_ewm_eval_276490_step080000_continuous_min8_decode_pro6000_20261107` | 五模型最终ID/OOD的唯一或组合来源，或用户已制作的论文展示图片；保留完整内容。 |
| `results/piper_five_speed_ood_step080000_continuous_min8_decode_5090_20261110` | 五模型最终ID/OOD的唯一或组合来源，或用户已制作的论文展示图片；保留完整内容。 |
| `results/piper_rgb_action_eval_262022_step080000_continuous_min8_error_record_5090_20261106` | 五模型最终ID/OOD的唯一或组合来源，或用户已制作的论文展示图片；保留完整内容。 |
| `results/piper_rolling_events_4x1_20261106` | 五模型最终ID/OOD的唯一或组合来源，或用户已制作的论文展示图片；保留完整内容。 |
| `results/piper_three_background_greygreen_recovery_5090_20261110` | 五模型最终ID/OOD的唯一或组合来源，或用户已制作的论文展示图片；保留完整内容。 |
| `results/piper_three_ood_262022_step080000_continuous_min8_decode_5090_20261107` | 五模型最终ID/OOD的唯一或组合来源，或用户已制作的论文展示图片；保留完整内容。 |

## results/：待复核（默认保留）（5项）

| 路径 | 依据 |
| --- | --- |
| `results/piper_action_eval_262022_step080000_background_speed_min8_5090_20261109` | 旧背景材质与未补齐速度 OOD 混在同一套件；新速度尚未完成，保留已有速度回合作为对照，不能整目录删除。 |
| `results/piper_ewm_243038_262022_controller_cross_5090_20261106` | 旧/新模型×执行协议交叉对照，不是最终主表；是否作为控制器消融留待确认。 |
| `results/piper_ewm_243038_262022_controller_cross_5090_20261106_v2` | 控制器交叉对照及275582任务证据；是否保留论文消融留待确认。 |
| `results/piper_ewm_eval_243038_step080000` | 包含旧 buffered16/Graph/exec10 多种对照，不能仅凭非最新名称判定无价值。 |
| `results/piper_ewm_eval_262022_step080000_background_speed_min8_decode_pro6000_20261109` | 旧背景材质和早期速度 OOD 部分结果；新速度完成前保留，需按 condition 子目录复核。 |

## tmp/：候选（7项）

| 路径 | 依据 |
| --- | --- |
| `tmp/ewm-cpu-regression-20261104` | 单次CPU回归启动器及日志；测试源码保留于项目测试目录，不是指定模型结果或latent实验。 |
| `tmp/ewm_eval_continue_20261022` | 早期218011测评续跑启动准备，不属于指定五模型最终结果。 |
| `tmp/ewm_eval_id_5090_20261022` | 早期218011模型ID测评工作目录，不属于指定五模型最终结果。 |
| `tmp/ewm_eval_import_fix_20261022` | 早期测评导入修复临时产物，非五组最终实验；删除前可选留存其中patch/报告。 |
| `tmp/ewm_eval_latest_20261022` | 早期218011最新checkpoint评估启动准备，不属于指定五模型最终结果。 |
| `tmp/ewm_eval_no_cudagraph_20261022` | 早期禁用Graph的测评调试工作目录，已被当前正式协议替代。 |
| `tmp/ewm_eval_slurm_headers_20261022` | 早期Slurm头部修复工作目录，旧测评任务已结束；只作为历史整理候选。 |

## tmp/：保留（54项）

| 路径 | 依据 |
| --- | --- |
| `tmp/background_greygreen_priority_277119_277129_20261109` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/background_resume_three_pro6000_20261110` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/background_speed_append_277119_277129_20261109` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/checkpoint_cleanup_20261028` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/dataset_layout_cleanup_20261018` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/deployment` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/event_rgb_latent_pilot_20261104` | event/RGB冗余度、PCA/预测性/重建实验的输入latent、源码、报告和图，用户明确保留。 |
| `tmp/ewm-causal-audit-20261104` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/ewm-formal-direct-20261104` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/ewm_243038_262022_controller_cross_20261106` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/ewm_243038_continuous8_complete_275582_20261106` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/ewm_243038_ood_append_277129_pro6000_20261108` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/ewm_276490_id_ood_pro6000_20261107` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/ewm_action_joint_262022_min8_5090_20261105` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/ewm_action_remaining_joint_262022_min8_record_20261106` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/ewm_formal_ada_serial_compile_20261104` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/ewm_formal_joint_stride1_ada_after262022_20261106` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/ewm_formal_joint_stride1_l40_20261104` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/ewm_formal_joint_stride1_l40s_fix_20261105` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/ewm_formal_workerfix_20261104` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/ewm_integrity_audit_20261108` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/ewm_no_memory_no_alignment_20261027` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/ewm_rgb_action_262022_min8_5090_20261106` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/ewm_three_ood_262022_5090_20261107` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/latent_expanded_20261104` | event/RGB冗余度、PCA/预测性/重建实验的输入latent、源码、报告和图，用户明确保留。 |
| `tmp/latent_expanded_20261104_results.json` | event/RGB冗余度、PCA/预测性/重建实验的输入latent、源码、报告和图，用户明确保留。 |
| `tmp/latent_expanded_20261104_subsample.json` | event/RGB冗余度、PCA/预测性/重建实验的输入latent、源码、报告和图，用户明确保留。 |
| `tmp/latent_pca_decoder_20261104` | event/RGB冗余度、PCA/预测性/重建实验的输入latent、源码、报告和图，用户明确保留。 |
| `tmp/latent_pca_middle_20261104` | event/RGB冗余度、PCA/预测性/重建实验的输入latent、源码、报告和图，用户明确保留。 |
| `tmp/latent_pca_reconstruction_20261104` | event/RGB冗余度、PCA/预测性/重建实验的输入latent、源码、报告和图，用户明确保留。 |
| `tmp/latent_predictability_probe_20261104` | event/RGB冗余度、PCA/预测性/重建实验的输入latent、源码、报告和图，用户明确保留。 |
| `tmp/latent_statistics_visualization_20261104` | event/RGB冗余度、PCA/预测性/重建实验的输入latent、源码、报告和图，用户明确保留。 |
| `tmp/latent_visualization_20261104` | event/RGB冗余度、PCA/预测性/重建实验的输入latent、源码、报告和图，用户明确保留。 |
| `tmp/ood_boundary_revision_20261108` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/ood_protocol_revision_20261108` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/openwam_random_1step_20261110` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/piper_asymmetric_a40_e16_20261103` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/piper_old_data_removal_20261019` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/piper_rgb_baselines_20261018` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/piper_single_rgb_train_20261103` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/project_organization_20261110` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/stop_after_background_277119_277129_20261109` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/storage_cleanup_20261021` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/storage_cleanup_20261104T102112Z` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/storage_cleanup_event_dino_20261110` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/storage_cleanup_selected_20261103` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/three_background_greygreen_after277129_pro6000_20261109` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/training_cache` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/training_data_cache` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/training_setup` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/wam_batch16_32_smoke_20261110` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/wam_lora_alignment_20261110` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/wam_random_serial_smoke_20261110` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |
| `tmp/wam_speed_optimization_20261110` | 正式模型、当前任务/历史结果依赖、WAM后续实验、部署数据或审计追溯；不按tmp名称清理。 |

## tmp/：待复核（默认保留）（215项）

| 路径 | 依据 |
| --- | --- |
| `tmp/action_only_narrow_227797` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/carton_boundary_v3_20261029` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/carton_lowwall_20261029` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/carton_no_memory_comparison_20261029` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/carton_replace_20261030` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/carton_stack_v2_20261029` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/carton_tail_review_20261029` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/carton_trial000_audit_20261029` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/carton_trial002_audit_20261029` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/continual_prediction_grasp_audit` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/conveyor_dense_no_latency` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/conveyor_memory_80k_eval` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/conveyor_memory_latency_fix` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/conveyor_memory_no_latency` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/conveyor_memory_ood_latency_eval` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/conveyor_memory_ttt_ood_eval` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/conveyor_speed_ood_latency_eval` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/conveyor_speed_strata_eval` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/conveyor_speed_strata_latency_eval` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/cups_drawer_fix_20261017` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/event_action_correction_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/event_failure_review` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/event_frame_cache_build_20261018` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/event_laststep_ttt_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/event_lora_ablation_v2` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/event_lora_continual_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/event_lora_ttt_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/event_prediction_correction_v2` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/event_prediction_position_audit` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm-ddp-resume-ada-compile-20261104` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm-ddp-resume-ada-compile-cuda-fix-20261104` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm-ddp-resume-audit-20261104` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm-formal-after-resume-248817-20261104` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm-simple-smoke-3muzuj7v` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_action_262022_eval_5090_continuous_20261105` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_action_joint_262022_min4_5090_20261105` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_action_only_20261024` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_action_only_eval_20261025` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_action_zero_20261025` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_ada_eight_compile` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_alignment_short_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_cached_ada_speed20_20261021` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_cached_l40_speed20_20261021` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_continuous_inference_20261103` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_continuous_min4_development_20261105` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_conveyor_launch` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_cpu_denoise_124_20261106` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_cups_joint4_20261029` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_cups_texture_20261029` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_ddp_resume_ada_compile_20261104` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_ddp_resume_ada_compile_cuda_fix_20261104` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_dino_fixed_batch_20261021` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_dino_six_cache` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_dual_eval_allow_compile_error_20261029` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_dynamic_training_20261021` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_eager_zero_latency_20261029` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_eight_formal_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_eval_compile_20261022` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_formal_cached_ada_20261021` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_gaussian_actions_20261024` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_input_opt_20261021` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_io_profile_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_joint_262022_eval_5090_continuous_20261105` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_joint_cache` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_l40_eight_compile` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_memory120_20261021` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_memory120_speed20_20261021` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_memory_eval_20261029` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_memory_eval_no_bf16_cache_20261029` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_memory_eval_pro6000_20261029` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_memory_no_alignment_20261027` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_memory_smoke` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_memory_speed` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_tail_mask_20261024` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_temporal_ensemble_20261106` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_window_routing_v3_20261026` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ewm_zero_noise_actions_20261024` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/groot_dynamic_eager_20261024` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/groot_dynamic_single_20261024` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/groot_eight_formal_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/laststep_motion_audit` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/laststep_outcome_audit` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/memory_consistency_20261029` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ood_rescued_motion_audit` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_240022_consistency_20261102` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_action_after_collect_20261101` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_action_bounds_20261023` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_action_eval_237659_20261101` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_action_first_four_retry_20261027` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_action_rgb2_compile_exec10_20261102` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_action_rgb2_latency_exec10_20261102` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_all_task_wrist_20261019` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_background_beige_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_background_ood_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_ball_skins_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_blur_ood_20261021` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_bowl_depth_tolerance_20261102` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_bowl_grasp_diagnosis_20261101` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_bowl_grasp_flow_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_bowl_handoff_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_bowl_tube_props_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_carrier_shadows_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_carton_expert20_20261022` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_carton_face_grasp_20261022` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_carton_grasp_smoothing_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_carton_recollect_20261022` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_carton_second_grasp_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_chunk16_exec10_20261103` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_container30_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_container_direction_audit_20261101` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_container_front_pan_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_container_long_flight_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_container_paired_launch_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_container_spatial_20261027` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_container_table_landing_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_container_v2_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_container_wide_v4_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_container_writer_fix_20261027` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_contour_quality_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_conveyor_capture_restore_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_conveyor_speed_protocol_20261021` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_conveyor_topdown_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_cups_success_audit_242049_20261102` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_dataset_quality_20261021` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_drawer_drop_20261019` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_dynamic_collection_20261021` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_dynamic_validation` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_edge_and_container40_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_eight_collect_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_eight_collection_runs` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_eight_eval_20261021` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_eight_expert20_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_eight_id_max_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_eight_refresh_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_eight_refresh_20261021` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_eight_serial_launcher_20261021` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_event_fix_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_event_noise_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_event_quality_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_event_threshold_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_ewm_243038_compile_exec10_5090_20261103` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_ewm_243038_first_four_graph_exec16_worker_5090_20261103` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_ewm_243038_graph_exec16_allow_error_5090_20261103` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_ewm_243038_graph_exec16_worker_5090_20261103` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_ewm_ada_resubmit_20261102` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_ewm_ada_resume_20261102` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_falling_height_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_falling_ready_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_falling_v2_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_first_two_props_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_five_narrow_recollect_20261102` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_four_front_unified_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_four_task_shadows_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_four_tasks_showcase_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_four_tasks_showcase_v2` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_front_rack_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_lighting_ood_20261021` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_low_geometry_20261030` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_marble_tabletop_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_moving_basket_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_narrow_recollect_20261025` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_narrow_v2_20261025` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_ood_unified_20261021` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_parabolic_exec4_20261027` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_parabolic_exec4_20261101` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_parabolic_v2_20261019` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_prop_scatter_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_props_trim_falling_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_push_t_pose_v2` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_remove_table_rail_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_restore_narrow_defaults_20261101` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_rgb2_eval_20261102` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_rgb2_eval_after_train_20261102` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_rgb2_train_20261101` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_rgb_twoframe_20261101` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_rolling_frontal_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_rolling_success_region_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_rolling_tolerance4mm_20261021` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_rolling_v2_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_room_background_20261019` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_room_default_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_room_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_room_wrist_shadows_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_rotating_tube_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_square_nut_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_stack_light_3000_20261019` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_static_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_tabletop_props_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_tasks_promotion_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_tennis_camera_20261030` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_tennis_framing_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_tennis_one_each_20261031_152500` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_tennis_repair_20261026` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_tennis_wrist_flight_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_three_expert_refine_20261101` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_tube_cube_area_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_tube_gravity_release_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_tube_half_insert_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_tube_layout_static_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_tube_speed_id20_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_tube_visibility_20261031` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_turntable_cups_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_turntable_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_two_expert_fixes_20261017` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_wrist_reinstall_20261019` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_wrist_visibility_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/piper_wrist_visual_fix_20261019` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/push_t_debug_208921` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/push_t_fix_20261018` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/push_t_pose_v2_20261017` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/tennis_extremes_audit_20261027` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/tennis_validation` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/trial_006_h264_pred_events_failure.mp4` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/ttt_fixed_probe` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `tmp/worker_exit_debug_20261104` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |

## logs/：候选（13项）

| 路径 | 依据 |
| --- | --- |
| `logs/piper_action_237659_id_238366.log` | 不在指定最终五模型测评中的旧评测任务日志，关联当前代码与正式结果不依赖其内容。 |
| `logs/piper_action_262022_cont5090_273942.log` | 不在指定最终五模型测评中的旧评测任务日志，关联当前代码与正式结果不依赖其内容。 |
| `logs/piper_action_eight_id_225261.log` | 不在指定最终五模型测评中的旧评测任务日志，关联当前代码与正式结果不依赖其内容。 |
| `logs/piper_action_first_four_id_228501.log` | 不在指定最终五模型测评中的旧评测任务日志，关联当前代码与正式结果不依赖其内容。 |
| `logs/piper_action_first_four_id_228599.log` | 不在指定最终五模型测评中的旧评测任务日志，关联当前代码与正式结果不依赖其内容。 |
| `logs/piper_action_tennis_id_228245.log` | 不在指定最终五模型测评中的旧评测任务日志，关联当前代码与正式结果不依赖其内容。 |
| `logs/piper_eight_id80k_221108.log` | 不在指定最终五模型测评中的旧评测任务日志，关联当前代码与正式结果不依赖其内容。 |
| `logs/piper_eight_id80k_resume_221296.log` | 不在指定最终五模型测评中的旧评测任务日志，关联当前代码与正式结果不依赖其内容。 |
| `logs/piper_eight_id_218950.log` | 不在指定最终五模型测评中的旧评测任务日志，关联当前代码与正式结果不依赖其内容。 |
| `logs/piper_eight_id_218996.log` | 不在指定最终五模型测评中的旧评测任务日志，关联当前代码与正式结果不依赖其内容。 |
| `logs/piper_eight_id_219013.log` | 不在指定最终五模型测评中的旧评测任务日志，关联当前代码与正式结果不依赖其内容。 |
| `logs/piper_eight_id_219147.log` | 不在指定最终五模型测评中的旧评测任务日志，关联当前代码与正式结果不依赖其内容。 |
| `logs/piper_six_id_220095.log` | 不在指定最终五模型测评中的旧评测任务日志，关联当前代码与正式结果不依赖其内容。 |

## logs/：保留（21项）

| 路径 | 依据 |
| --- | --- |
| `logs/checkpoint_cleanup_20261031.json` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/checkpoint_cleanup_20261108_015539.json` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/deployment` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/ewm_243038_262022_2x2_5090_275582.log` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/ewm_eight_action_only` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/ewm_eight_train` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/ewm_formal_ada_serial_262022.log` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/ewm_joint_stride1_ada_276490.log` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/ewm_joint_stride1_l40_262686.log` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/fastwam_lora_verify_207991.log` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/openwam_random_1step_1732728.log` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/piper_action_joint_262022_min4_5090_274408.log` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/piper_action_remaining_joint_262022_min8_record_275074.log` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/piper_bg_speed_five_5090_1732584.log` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/piper_ewm_276490_id_ood_pro6000_277129.log` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/piper_rgb_action_262022_min8_5090_276282.log` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/piper_rgb_baselines` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/piper_three_background_greygreen_1196780.log` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/piper_three_ood_262022_5090_277119.log` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/training_envs` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |
| `logs/training_weights` | 最终五模型训练/测评、当前运行任务、WAM实验或部署/清理审计日志。 |

## logs/：待复核（默认保留）（253项）

| 路径 | 依据 |
| --- | --- |
| `logs/action_only_after_collect_237659.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/bowl_grasp_depth_238772.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/conveyor_dense_no_latency` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/conveyor_memory_80k_eval` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/conveyor_memory_latency_fix` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/conveyor_memory_no_latency` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/conveyor_memory_ood_latency_eval` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/conveyor_memory_ttt_ood_eval` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/conveyor_speed_ood_latency_eval` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/conveyor_speed_strata_eval` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/conveyor_speed_strata_latency_eval` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/data_collection_archive_20261018` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/event_action_correction_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/event_frame_cache_build_20261018` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/event_laststep_ttt_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/event_lora_ablation_v2` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/event_lora_continual_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/event_lora_ttt_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/event_prediction_correction_v2` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_a6000_loop_smoke_245653.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_a6000_smoke_245652.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_action_zero_225634.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_ada_eight_compile` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_alignment_short_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_cached_ada_speed20` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_cached_l40_speed20` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_cups_joint4_234278.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_cups_texture_234331.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_cups_texture_234356.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_cups_texture_234376.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_cups_texture_234407.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_cups_texture_234437.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_ddp_resume_ada_compile_247639.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_ddp_resume_ada_compile_248817.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_dino_fixed_batch` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_dino_six_cache` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_dynamic_cache_217159.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_gaussian_actions_223474.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_input_opt20` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_io_profile_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_l40_eight_compile` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_memory120_speed20` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_memory_smoke` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_memory_speed` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_protocol_tests_245475.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_protocol_tests_245486.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_zero_actions_223094.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ewm_zero_actions_223210.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/groot_eight_train` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ood_rescued_motion_audit` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_action_eight_zero_235172.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_action_rgb2_compile10_241976.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_action_rgb2_eval_240022.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_action_rgb2_latency10_242049.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_background_beige_214702.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_carton_cache_221081.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_carton_collect_220178.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_carton_expert20_220077.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_carton_face_219917.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_carton_face_v03_219917.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_carton_face_v04_219917.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_carton_fast_220014.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_carton_fast_220023.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_carton_fast_220040.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_carton_fast_220052.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_carton_lowwall_234211.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_carton_lowwall_234213.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_carton_lowwall_234220.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_carton_lowwall_cache_234460.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_container30_214702.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_container40_214684.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_container40_5090.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_container40_render_214702.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_container_front_pan_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_container_long_flight_215200.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_container_long_flight_215200_v02.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_container_long_flight_215200_v03.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_container_long_flight_final_215200.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_container_long_flight_tight_215200.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_container_paired_launch_215200.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_container_spatial_cache_229562.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_container_spatial_collect_228590.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_container_spatial_collect_229561.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_container_v2_20261020` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_conveyor_capture_restore_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_drawer_drop_212628.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207107.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207441.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207442.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207449.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207496.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207513.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207514.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207519.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207523.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207524.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207531.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207537.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207550.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207560.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207566.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207589.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207598.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207612.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207613.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207647.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207648.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207660.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207670.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207671.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207685.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207686.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207717.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207723.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207727.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207734.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_dynamic_207736.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_edge_diagnostic_214702.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_edge_fixed_214702.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_eight_collect_20261031_237655.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_eight_collect_216144.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_eight_id_max_215200.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_ewm_243038_compile10_244164.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_ewm_243038_first4_graph16_worker_244662.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_ewm_243038_graph16_244481.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_ewm_243038_graph16_worker_244626.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_ewm_eight_blur_235184.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_falling_213660_physics_v01.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_falling_213660_physics_v02.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_falling_213660_physics_v03.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_falling_213660_physics_v04.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_falling_213709_physics_v05.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_falling_213709_render_fast.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_falling_213709_render_slow.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_falling_ready_213709_physics.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_falling_ready_213709_render.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_five_narrow_cache_242895.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_five_narrow_collect_242894.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_four_tasks_showcase_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lighting_216978.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_cpu_213365.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213503_v11_physics.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213503_v12_physics.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v05.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v05_retry.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v06.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v07.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v07_render.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v08.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v09.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v10_render.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v11_render.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v12_physics.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v13_physics.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v13_render.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v14_render.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v15_render.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v16_negative_render.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v16_physics.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v16_positive_render.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v17_negative_render.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v17_physics.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v17_positive_render.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v18_negative_render.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v18_physics.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v18_positive_render.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v19_negative_render.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v19_physics.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_interactive_213513_v19_positive_render.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_physics_213391.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_v2_213105.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_v2_213108.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_v2_213360.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_lob_v2_213376.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_marble_tabletop_214702.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_marble_tabletop_v02_214702.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_marble_tabletop_v03_214702.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_memory_eager_zero_233764.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_memory_eight_id_233387.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_memory_eight_id_233444.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_memory_eight_id_233462.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_memory_eight_id_233597.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_memory_eight_id_233598.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_moving_basket_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_narrow_cache_227546.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_narrow_collect_226158.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_no_memory_eager_zero_233765.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_no_memory_last_five_234503.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_no_memory_last_six_234177.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_parabolic_exec4_228601.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_parabolic_exec4_239406.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_promotion_reference_207119.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_remove_stand_236193.err` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_remove_stand_236193.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_rolling_213660_v01_negative.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_rolling_213660_v01_physics.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_rolling_213660_v01_positive.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_rolling_213660_v02_fast.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_rolling_213660_v02_negative.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_rolling_213660_v02_physics.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_rolling_213660_v02_positive.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_rolling_213660_v02_slow.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_rolling_front_213660_physics_v01.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_rolling_front_213660_physics_v02.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_rolling_front_213660_render_fast.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_rolling_front_213660_render_positive.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_room_207399.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_room_background_212664_0.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_room_background_212673_0.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_room_background_212678_0.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_room_background_212682_1.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_room_background_212682_2.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_room_background_212682_3.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_room_background_212718_0.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_room_background_212718_1.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_room_background_212718_2.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_room_background_212718_3.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_rotating_tube_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_stack_light3000_213094.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_table_landing_215188.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_table_landing_215195.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_tennis_camera_236098.err` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_tennis_camera_236098.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_tennis_lower_236117.err` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_tennis_lower_236117.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_tennis_lower_236119.err` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_tennis_lower_236119.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_three_no_latency_222068.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_three_preflight_239203.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_turntable_cups_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_turntable_v1` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_two_exec4_223827.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_all_212480_0.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_all_212480_1.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_all_212480_2.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_all_212480_3.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_render_212366.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_render_212381.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_render_212387.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_render_212389.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_render_212394.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_render_212400.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_render_212450.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_render_212463.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_render_212470.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_render_212475.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_replay_212496_0.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_replay_212496_1.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_replay_212496_2.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_replay_212496_3.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_replay_212517_4294967294.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/piper_wrist_replay_212522_4294967294.log` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |
| `logs/ttt_fixed_probe` | 历史实验/审计/采集或调试产物，尚未确认是否有独立价值或其他引用；不列入删除候选。 |

## 已执行验证

- 所有一级条目均且仅记录一次，路径存在。
- 候选路径互不嵌套；候选不覆盖任何最终ID/OOD、checkpoint或latent实验保护目录。
- 五模型最终ID各8任务×50回合已从套件JSON核验；Action-only按两个套件组合核验。
- 四个标为占位的旧结果目录已确认没有任何 `trial_*_result.json`。
- 本次未删除文件、未改动实验产物、未运行训练/测评或新增GPU任务；没有commit/push。
