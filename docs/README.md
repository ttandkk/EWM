# 文档索引

当前入口以根 [README](../README.md) 和 [正式测评协议](evaluation_protocol.md) 为准。

| 文档 | 用途 |
| --- | --- |
| [正式测评协议](evaluation_protocol.md) | continuous/min8、等效延迟、编译和视觉预测记录 |
| [产物保留清单](artifact_retention_20261110.md) | 五模型最终 ID/OOD、分段补测、冗余度实验及删除候选；不执行删除 |
| [八任务采集协议](piper_eight_task_collection_eval_protocol_v1.md) | 数据和任务判据；历史段落与当前更新分开阅读 |
| [八任务训练计划](ewm_experiment_plan_20261104.md) | 262022 三组训练来源和预算；推理安排以正式测评协议为准 |
| [Piper RGB 基线](piper_rgb_baselines.md) | OpenWAM/FastWAM 数据接入、LoRA、无梯度累积 |
| [LeanVAE EWM](../OpenWAM/docs/ewm_wan_vae.md) | 事件编码与模型实现 |
| [RGB-action](../OpenWAM/docs/ewm_rgb_action_piper.md) | RGB 视觉目标与动作联合训练契约 |
| [上传目录清单](github-upload-tree.txt) | 当前源码发布包含的目录与排除范围 |

`latent_representation_analysis_20261104/`、相关图表及 `tmp/latent_*`、`tmp/event_rgb_latent_pilot_20261104/` 属于 event/RGB 冗余度分析，按产物清单完整保留。`tmp/` 和旧 `archive/` 仅在本地保留。研究笔记和验收记录中的历史绝对路径描述原工作区，不代表这些产物随源码发布。
