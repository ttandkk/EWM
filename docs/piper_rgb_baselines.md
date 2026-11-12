# Piper 八任务 RGB 基线数据接入

> 2026-11-10发布整理：当前八任务接入以本文件第一节为准，下方2026-10-18段落保留历史状态。FastWAM固定版本和本地Piper/LoRA补丁位于`benchmark/src/piper_bench/resources/deployment/`；当前安装器按所选profile使用它们，已有工作树不覆盖。最新B16/B32随机初始化吞吐实验及源码按 [产物保留清单](artifact_retention_20261110.md) 保存，它们不是正式预训练模型的收敛验证。

## 当前协议：2026-11-10 Wan LoRA 对齐

本节覆盖下方 2026-10-18 历史接入配置。FastWAM 和 OpenWAM 现在使用当前动态八任务：conveyor、moving_carton、turntable_cups、rotating_tube、tennis_rolling、tennis_falling、tennis_parabolic、tennis_container。每任务 50 回合，共 400 回合、91,935 帧。

RGB-only 导出目录为 `tmp/training_data_cache/piper_rgb_dynamic_v1/`。400 个轻量 Parquet 只含动作、状态及索引；800 个视频使用相对软链接复用原始 RGB。仍使用 24 个连续动作和 `k,k+2,...,k+24` 的 13 帧双视角 RGB，水平拼接为 256×512，不读取事件或物体真值。`prepare --tasks ...` 仍支持显式指定其他八任务，不覆盖已有导出。

两套视频 LoRA 均为 rank=16、alpha=16、dropout=0，缩放 alpha/rank=1。Wan 的全部 30 个 Transformer block 均覆盖以下 10 个 Linear（共 300 处、600 个 A/B 张量）：

- `self_attn.q/k/v/o`
- `cross_attn.q/k/v/o`
- `ffn.0`、`ffn.2`

| 模块 | FastWAM | OpenWAM | 训练状态 |
| --- | --- | --- | --- |
| 视频 LoRA A/B | `video_expert.blocks.*` | `video_backbone.dit.blocks.*` | 训练 |
| 视频原始权重、bias、patch/text/time embedding、norm、modulation、输出头 | `video_expert` | `video_backbone.dit` | 冻结 |
| 原生动作专家全部参数 | `action_expert` | `action_backbone` | 全量训练，未加 LoRA |
| 当前状态投影 | `proprio_encoder` | `proprio_encoder` | 全量训练 |
| VAE、T5 及可选图像编码器 | 原生冻结组件；T5 默认使用缓存 | `video_backbone` 中对应组件 | 冻结 |

只冻结参数，不能把含 LoRA 的视频 Transformer 整体放进 `no_grad`，否则会切断适配器和动作分支所需的梯度。OpenWAM 保留原生完整 safetensors checkpoint 格式，保存/重建包含 LoRA；FastWAM 继续使用现有紧凑适配器＋动作 checkpoint 格式。

两套默认均为 BF16、AdamW、学习率 1e-4、weight_decay=0、梯度检查点开启。FastWAM 原 MoT tensor core 会拒绝该开关，本次补齐完整 joint layer 的 non-reentrant checkpoint 执行路径，并比较重计算前后输出与全部可训练梯度。按用户要求，**不使用梯度累积：gradient_accumulation_steps=1**。每张 GPU 的 batch 由实际容量短测确定，各模型允许不同。配置支持 `PIPER_BATCH_SIZE` 环境变量；未测时默认 1 仅为启动值，并非测得的最大值。

训练预算统一为 **30,000 次优化器更新**。FastWAM 使用 `max_steps=30000`（原生即更新次数，覆盖 epoch 设置）；OpenWAM 使用 `training.max_optimizer_steps=30000`，同时 `num_epochs=null,max_steps=null`。OpenWAM 原来的 `max_steps` 仍保留微批次含义；新预算模式下进度条和 `save_steps` 按更新计数，checkpoint 文件名仍保留微批次 global_step，完整恢复状态另存精确 opt_step。epoch 尾部不足一个累积周期时也计入真实更新次数。不同 batch 下样本总量不同，比较时同时报告 batch、更新数和样本量。

保留各基线原生动作噪声调度、动作初始化与尾部 padding 策略，未把不同模型强行变为相同算法。

```bash
# 已导出当前 400 回合；重复 prepare 会拒绝覆盖。
bash scripts/train_piper_rgb_baseline.sh prepare

# 必须在分配到的单 GPU 上运行；先生成 FastWAM 的八条文本缓存。
# 探测不保存权重，每个候选在独立进程中完成真实优化器更新。
python3 tmp/wam_lora_alignment_20261110/probe_batch.py fastwam
python3 tmp/wam_lora_alignment_20261110/probe_batch.py openwam

# 将各自报告的 selected_batch 赋给 PIPER_BATCH_SIZE 后启动训练。
# 命令本身不申请 GPU；当前未启动正式训练。
PIPER_BATCH_SIZE=<测得的整数> bash scripts/train_piper_rgb_baseline.sh fastwam
PIPER_BATCH_SIZE=<测得的整数> bash scripts/train_piper_rgb_baseline.sh openwam
```

容量脚本先倍增、再二分，最后复测；若到达搜索上限仍未 OOM，会明确标记为“搜索上限内可用”，不会伪称真实显存上限。非 CUDA OOM 错误直接停止。GPU 容量及完整训练稳定性仍需实际短测。

## 2026-10-18 历史接入记录

## 数据范围与隔离

GR00T N1.7、OpenWAM 和 FastWAM 共用现有八任务、每任务 50 条成功训练轨迹，共 400 条、193,573 帧。原始 `datasets/piper_*` 不修改、不重采。

| 任务目录 | 轨迹数 | 帧数 |
| --- | ---: | ---: |
| piper_stack | 50 | 22,685 |
| piper_conveyor | 50 | 13,763 |
| piper_drawer_cup | 50 | 36,844 |
| piper_square_nut | 50 | 16,394 |
| piper_moving_carton | 50 | 24,951 |
| piper_turntable_cups | 50 | 27,862 |
| piper_rotating_tube | 50 | 15,722 |
| piper_push_t | 50 | 35,352 |

兼容数据位于 `tmp/training_data_cache/piper_rgb_baselines_v1/`，其下每任务一个 LeRobot v2.1 目录。每回合生成只含白名单列的轻量 Parquet；800 个 RGB 视频使用相对软链接复用原始文件，不重编码、不复制视频。移动数据时必须同时保留源目录及相对关系，或将软链接实体化。

允许输入为前视 RGB `observation.images.cam0`、腕视 RGB `observation.images.cam_wrist`、8 维 `observation.state` 和任务指令；监督为 7 维 `action`。Parquet 额外保留时间戳及帧/回合/任务索引。**event 帧、原始事件、事件缓存、物体位姿/速度/活动状态、末端真值等均不导出，也不参与模型训练。**

`meta/source_manifest.json` 记录源路径及源文件大小/修改时间；转换时逐回合核对持久化后的全部动作、状态和索引，源文件变动会使验证失败。`manifest.json` 只在全部八任务导出成功后出现；现有导出目录不会被脚本覆盖。

## 时间、动作和归一化

- 采样率 30 Hz，预测 `action[k:k+24]`，即 0.8 秒动作窗口。保留所有原始动作值和顺序。
- 动作前六维是采集器写出的 IK-relative 控制指令，末维为夹爪指令；它不是关节位置目标，也不直接等同于米/弧度。状态为 8 维关节读数。接入不再次差分、不转换旋转表示、不从状态重建动作。
- OpenWAM/FastWAM 使用 25 个控制时刻中的偶数时刻 RGB：`k, k+2, …, k+24`，共 13 帧，符合 Wan 的 `4n+1` 约束；两相机水平拼为 256×512，前视在左、腕视在右。状态仅以当前帧作为模型条件，未来 RGB 用作世界模型监督。
- GR00T 输入当前两路 RGB，24×7 有效动作由原生 processor 填充到预训练头的 40×132，并用 mask 排除填充维度。`ABSOLUTE/NON_EEF/DEFAULT` 是 GR00T 的“不再转换已记录指令”配置，不表示 Piper 采用绝对位姿控制。
- 三套入口使用全训练数据的 state/action min-max 归一化。OpenWAM/FastWAM 读取导出时生成的全局统计；GR00T 原生混合数据流程汇合各任务 min/max，得到相同的范围。统计仅来自训练数据；测评复用训练统计。
- 不把训练回合随机分成窗口级“验证集”。后续验证/ID/OOD 测评使用独立回合和种子，遵循八任务测评协议。
- FastWAM 原生工厂在没有 `data.val` 时把训练集对象也返回为 val；本配置明确设置 `eval_every: 0`，不把这种训练集评估作为验证结果。
- 保留各原生采样/尾部策略：OpenWAM 丢弃没有下一帧的末行，短窗口填充并 mask；FastWAM 使用原生 LeRobot padding；GR00T 使用完整 24 步窗口。各任务样本频率默认随有效窗口数变化，并非每任务严格等权；正式比较时应同时报告采样策略、更新步数和算力。

## 接入文件

| 模型 | 配置/适配器 | 训练行为 |
| --- | --- | --- |
| OpenWAM | `configs/train_openwam_piper_rgb.yaml`、`configs/dataloader/piper_rgb.yaml`、`openwam/dataloader/piper_rgb.py` | 原始 dual-system RGB/action 联合训练；action_dim=7、state_dim=8 |
| FastWAM | `configs/data/piper_rgb.yaml`、`configs/task/piper_rgb.yaml` | 原生 LeRobot 加载器；默认复用已有 Wan LoRA + ActionDiT 配置 |
| GR00T N1.7 | `examples/Piper/piper_config.py` | 原生新 embodiment；默认冻结视觉/语言骨干，训练 projector 和 diffusion head |

本次是数据接入，未改变模型算法。OpenWAM 默认仍是完整联合 WAM，不应将它等同于已有 action-only 变体；其正式训练预算需单独短测。FastWAM 的 LoRA 配置也应在实验记录中明确标注。未启动正式训练。

## 使用

从 EWM 根目录运行。训练命令应放在已分配 GPU 的作业中；此脚本本身不会申请 GPU 或自动提交训练。

```bash
# 已准备好时无需再执行；拒绝覆盖已有导出。
bash scripts/train_piper_rgb_baseline.sh prepare

# FastWAM 冻结 T5 文本缓存，只编码八条任务指令；支持 CPU。
bash scripts/train_piper_rgb_baseline.sh fastwam-text-cache

# 正式训练入口；各自额外参数原样交给原生 CLI。
bash scripts/train_piper_rgb_baseline.sh openwam training.max_steps=10000
bash scripts/train_piper_rgb_baseline.sh fastwam max_steps=10000
bash scripts/train_piper_rgb_baseline.sh groot --max-steps 10000
```

权重和环境使用已部署的本地版本，现有环境选择入口为 `deployment/activate_training.sh`；启动器开启 HF 离线模式及 `DIFFSYNTH_SKIP_DOWNLOAD=true`，禁止隐式下载。FastWAM 通过原生 `scripts/train_zero1.sh` 启用 DeepSpeed ZeRO-1；OpenWAM 使用其原生 torchrun/ZeRO-2 入口。通过 `PIPER_RGB_DATA_ROOT` 覆盖兼容数据目录，通过 `NPROC_PER_NODE` 指定 GPU 数。默认每 GPU microbatch=1、累积 16 次只是保守起点，未代表测得的训练速度或可完成的预算。

## 验证

`scripts/verify_piper_rgb_baselines.py` 在对应模型环境运行，调用真实加载器及 GR00T/FastWAM 原生 collator，覆盖每任务第 0、25、49 回合的起始、中间和尾部窗口，共每模型 24 个样本。检查双 RGB 尺寸、时间索引、动作反归一化、有效维度/填充 mask 和文本缓存；FastWAM 验证直接调用 `_get`，避免随机重试掩盖某个回合失败。

```bash
envs/lerobot/bin/python scripts/verify_piper_rgb_baselines.py export
envs/openwam/bin/python scripts/verify_piper_rgb_baselines.py openwam
# 后两项先 source deployment/activate_training.sh 对应环境。
python scripts/verify_piper_rgb_baselines.py fastwam
python scripts/verify_piper_rgb_baselines.py groot
```

验证报告：`tmp/piper_rgb_baselines_20261018/*_validation.json`；CPU 作业日志：`logs/piper_rgb_baselines/`。这些检查不包含大模型 GPU 前反向或闭环成功率；正式训练前仍需使用实际分配的 GPU 做短测。

2026-10-18 已执行结果：

- 导出作业 `210444` 完成，400 回合、193,573 行持久化核对通过；400 个 Parquet 的列/数值和 800 个视频链接检查通过，原数据文件指纹不变。
- 作业 `210452`：OpenWAM 24/24、GR00T 24/24 原生样本通过。
- 作业 `210454`：八条 T5 文本缓存完成，0 条指令截断。
- 作业 `210468`：通过 FastWAM 正式 `runtime.build_datasets` 工厂验证，24/24 原生样本通过。此前工厂复核 `210465` 因验证脚本误以为无 val 配置会返回 `None` 失败；实际行为是返回训练集别名，修正为检查该行为及 `eval_every == 0` 后通过，未修改上游工厂或放宽数据检查。
- Python AST、`bash -n`、三个仓库 `git diff --check`、GR00T 原生 CLI `--help` 检查通过。本轮仅使用 CPU；未跑 GPU 前反向和正式训练。

本次源码文件清单：

- 新增 `scripts/prepare_piper_rgb_baselines.py`：RGB-only 导出及全局归一化统计。
- 新增 `scripts/verify_piper_rgb_baselines.py`：原生加载器、processor/collator 和源数据完整性验证。
- 新增 `scripts/train_piper_rgb_baseline.sh`：三个基线及文本缓存的统一入口。
- 新增 `OpenWAM/openwam/dataloader/piper_rgb.py`；修改 `OpenWAM/openwam/dataloader/registry.py`，注册数据集。
- 新增 `OpenWAM/configs/dataloader/piper_rgb.yaml`、`OpenWAM/configs/train_openwam_piper_rgb.yaml`。
- 新增 `FastWAM/configs/data/piper_rgb.yaml`、`FastWAM/configs/task/piper_rgb.yaml`。
- 新增 `Isaac-GR00T/examples/Piper/piper_config.py`。
- 新增本文档 `docs/piper_rgb_baselines.md`。

此外生成兼容数据/文本缓存约 84 MiB，以及 `tmp/piper_rgb_baselines_20261018/` 下的 CPU 验证 sbatch、审计文件和 JSON 报告；HF 本地 Parquet 缓存位于已有 `tmp/training_cache/` 树。未改依赖、lockfile、原始数据或已有未提交实现，未删除文件，未执行 commit/push。
