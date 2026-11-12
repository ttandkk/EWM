# 简化 EWM 八任务实验计划

> 2026-11-10 文档整理：本文保留2026-11-04训练计划及当时状态，历史“尚未启动”等记录不是当前进度。正式推理已更新为continuous/min8及回合后视觉解码，见 [当前测评协议](evaluation_protocol.md)；五模型最终结果与分段补测见 [产物索引](artifact_retention_20261110.md)。本次不改写原训练配置或既有结果。

记录日期：2026-11-04。协议版本：`ewm_simple_eight_v1_continuous`。

用户已确认后续严格执行此前实验方案，并提出统一采用 DynamicVLA 的推理方式。本记录保留三组模型、24 步预测时域、数据与训练预算，将主评测调度确定为持续推理。此前“无等效延迟、固定执行 10 步”的安排转为后续对照，不与主结果混表。

本文是后续实验的执行依据。历史迁移说明、旧 checkpoint 配置和旧启动脚本默认值均不能覆盖本文；用户后续明确调整时，更新版本和受影响的实验批次。记录方案不表示实现已全部接通，也不表示已经提交训练或评测作业。

## 固定比较对象

| 方法 | 当前观测 | 预测与训练目标 |
| --- | --- | --- |
| action-only | 双相机各一帧 RGB、当前 state、指令 | 24 步 action；不保留事件 expert 或闲置事件参数 |
| RGB-action | 双相机各一帧 RGB、当前 state、指令 | 24 步 action + 双相机未来 RGB |
| 简单 EWM（joint） | 双相机各一帧 RGB、当前 state、指令、当前事件 | 24 步 action + 双相机未来事件 |

RGB-action 与 joint 的视觉预测时域统一为 **24 个原始控制 tick**，视觉编码输入的时间下采样统一为 `event_temporal_downsample=2`。动作仍覆盖全部 24 个 tick；“视觉时域 24”不代表送入 VAE 的未来帧数也为 24。保持当前 LeanVAE 时间压缩及有效标签 mask 语义。

只保留上述三组作为正式比较：不恢复记忆、事件触发重规划、额外事件对齐模块、双时刻 RGB 历史、TTT 或分层结构。必要的观测/动作时间戳对齐和过期动作丢弃属于执行协议，必须保留。

这项实验比较完整方法，差异同时包括事件输入、视觉训练目标和推理开销，不能把成功率差异直接归因于某一个 loss。

## 固定数据与训练

- 使用当前八任务，每任务 50 条，共 400 条轨迹；尾部补齐后共有 **91,935 个训练窗口**，任务均衡采样，任务内均匀采样。
- 三组使用相同轨迹、训练 anchor、pre-action 观测约定、动作归一化、尾部 padding 和有效标签 mask。不能因 RGB-action 尚未适配而删掉短轨迹或尾部窗口。
- 数据目录为仓库 `datasets/piper_<task>`；逐任务清单与当前来源指纹见文末。固定采集场景、资产、相机、控制频率及成功判据；不在这一轮重采或混入旧导出。
- pooled normalization：`tmp/training_data_cache/piper_dynamic_v1/shared_normalization.json`，SHA256 为 `7e4291e15964624aaff460aeda60e9de697d396f67e875f111868ba2225aedd3`。提交前重新验证实际文件与数据来源。
- 可训练部分从头初始化，首轮训练 seed=42；DINOv3、T5、LeanVAE 使用当前本地冻结权重。action-only 不加载不用的 VAE。相同 seed 不宣称不同网络的共享参数必然逐项相同。
- 视觉 expert 宽度 768、action expert 宽度 512、12 层、8 heads；共享部分配置一致，记录各组实际参数量。不得为一组单独扩大模型。
- 每组 **80,000 optimizer steps**，global batch=64（两卡各 32，梯度累积 1），BF16；学习率 `1e-4`，AdamW betas=(0.9,0.95)、weight decay=0.01、grad clip=1；cosine 80k，warmup 5%，最低学习率比例 0.01。
- 每 20k 保存 checkpoint（20k/40k/60k/80k），同时保存完整恢复状态。恢复同一 run 必须使用完整状态，不能只载入权重后重置优化器和学习率。
- GPU 型号、编译、缓存、预处理和依赖环境在短测完成后一次性固定并写入 manifest。同种语义的缓存必须经过一致性检查；三组不能因缓存差异改变输入精度或数据。
- 提交前冻结实际源码快照、resolved config、权重/数据/normalization 指纹；作业启动时复核。排队期间改了源目录也不能悄悄改变已冻结实验。

## 统一推理与执行协议

三组主评测采用 DynamicVLA 的 Continuous Inference 和 Latency-aware Action Streaming 思路，使用本项目的 `continuous` 控制器：

1. 每次取最新控制步观测，预测完整 24 步动作，始终最多一个请求在途；前一次推理完成且取得新观测后，立即开始下一次。
2. 等待新结果期间，执行旧计划中仍对应当前时刻的动作。
3. 新计划根据观测时刻和送达时刻丢弃过期前缀，用有效新预测替换旧计划后续动作；首个计划也按相同规则处理。
4. 整块过期的回复不执行。无有效动作时位移/旋转增量为零，保持夹爪命令；不能重复执行上一条相对位移。等待、整块过期、回合结束时未送达的计划分别记录。
5. **每次预测 24 步，实际执行多少步由下一次预测送达决定**，不固定为 4、6、10 或 24。统一调度和测量口径，允许不同模型因实际延迟而有不同的重规划间隔。

本项目继续使用 **30 Hz** 控制、每次 **4 步 flow 采样**。复用 DynamicVLA 调度，不引入其两时刻 RGB 输入，也不把本项目改成 25 Hz 或 20 步预测。

主表使用现有**离线等效延迟**：测量 `action_reply_round_trip_seconds`，按 `ceil(latency_seconds * 30)` 转换为仿真控制步，让场景在相应仿真时间内继续演化。它不是实机墙钟并行测试，报告中必须保留 `wall_clock_realtime=false`。三组使用同型号 GPU、同控制器、同编译策略、同计时范围和预热规则；冷启动/编译耗时单独记录，正式回合仍计入首个计划的在线推理等待。

正式评测不额外解码未来 RGB/event 用于视频展示，以免占用推理线程改变后续请求时机。模型生成动作所需的视觉 latent 联合去噪仍保留并计入耗时。可视化另开诊断运行。

现有命令行默认控制器仍为 `buffered`，本次记录不修改运行默认值。正式命令必须显式指定：

```text
--latency-aware --latency-controller continuous --sampling-steps 4
```

`--execution-horizon` 在 continuous 下不决定重规划间隔。正式提交时必须验证实际解析后的配置，不能仅凭脚本文件名推断。出现延迟超过有效预测覆盖、长时间队列耗尽或不同动作语义时，先修复/诊断，不能静默切回 buffered、延长 horizon 或从回合失败中删去这些情况。

### 每个 plan 实际执行动作数的统计

直接按 `trial_*_result.json` 中 `latency_timing.inference_requests` 的请求 ID 统计，并与逐步日志核对：

- `N_submitted`：已发出的计划数。
- `N_received`：回合结束前已经送达的计划数，包括送达时整块过期、执行 0 步的计划。
- `N_used`：实际执行至少 1 步的计划数。
- 主均值：`sum(executed_actions of received plans) / N_received`；同时报告 `N_used` 对应均值和长度分布。分母为 0 时记 N/A。
- 已送达计划按原始实际执行长度计数，包括回合结束截断的计划；最终未送达计划单列为 censored，不与整块过期的 0 步计划混淆。
- 每组同时报告端到端延迟 mean/p50/p95、重规划间隔、执行长度 mean/p50/p95、丢弃前缀、整块过期率、启动等待和队列耗尽 hold 步数、成功率与完成时间。

现有请求记录已有 `executed_actions`；当前 `horizon_statistics(chunks)` 只汇总形成的执行片段，不能直接代替上述“包括 0 步计划”的口径。正式实验前补齐统一汇总。

## 执行顺序和验收

| 阶段 | 内容 | 通过条件 |
| --- | --- | --- |
| 0 接口对齐 | 补齐 RGB-action 八任务训练、尾部 mask、双视角未来 RGB、视觉 horizon/stride，以及同一 continuous 评测入口；核对 action-only 的观测/归一化/动作执行链 | 三组预检使用同一数据与时钟协议；不携带已移除功能 |
| 1 短测 | 每组 100 个训练 step，加少量闭环接口试跑 | 数据、mask、梯度、保存/加载和动作数值正常；连续推理时间戳、过期前缀、hold、请求计数正确；吞吐/显存可接受 |
| 2 正式训练 | 三组 seed=42，各 80k；20k/40k 用独立 dev 场景做诊断 | 遵守统一训练预算；若改变实现或训练条件，三组受影响结果重新归入新批次 |
| 3 主评测 | 每组固定 80k checkpoint，8 任务 × 每任务 50 回合 ID，统一 continuous | 复用相同测试场景清单/种子/时间上限/成功判据；每组 400 回合，共 1,200 回合；记录全部请求统计与失败 |
| 4 复验与消融 | 主实验完成后增加 seed=43、44；再使用固定权重比较固定执行 4/10 步；随后做 OOD | 每次只改变一个因素；与 continuous 主表区分；额外预算单独记录 |

阶段 1 的 100 step 只用于工程验收，不作为策略能力结论。先补齐 RGB-action 再开始正式三组训练，不能先跑两组、之后为第三组改变数据或评测口径。

主结果固定使用 **80k**，不能根据测试集表现挑选“最好 checkpoint”。dev 与正式测试采用不同场景/随机流；正式测试三组复用同一份 scene manifest，沿用现有 scene seed=20280002、policy seed=10000 的生成规则并冻结具体清单。20k/40k 的 dev 诊断预算另记，不计入 1,200 个主测试回合。

报告逐任务成功数/50、八任务宏平均和 95% 置信区间；对三组差值使用相同场景的配对统计。首轮只有一个训练 seed，区间只反映评测抽样不确定性，不能冒充训练 seed 方差。每个任务正式 50 回合必须补齐；中断结果只能标为部分结果/下界。跨模型不直接比较总 loss 大小。

重点检查 `tennis_falling`：结合逐步日志/视频区分启动等待错过窗口、预测/跟踪偏差、闭合时机、抓住后掉落等原因；action-only 同时检查训练与推理的状态/动作尺度、增量语义和输入时刻。故障分类依据实际轨迹，不预先假定唯一原因。

后续无延迟、固定执行 4/10 步的对照用于分离控制频率和在线延迟影响；每个小对照固定其他条件。OOD 优先运动模糊与速度，再做光照、背景；物理渲染变化和仅 RGB 后处理必须明确区分，不能混为同一种条件。

首轮核心预算为 **3 次 80k 训练 + 1,200 个正式 ID 回合**；短测、dev、后续种子与 OOD 单独计账。

## 当前实现状态

- 已有八任务 joint/action-only/RGB-action 训练入口。joint 与 RGB-action 均为 action=24、视觉时域=24、视觉 stride=2；RGB-action 使用双视角未来 RGB，并对动作和未来帧实施尾部 mask。
- 三组已接入 `ContinuousLatencyActionStream` 和八任务评测入口；RGB-action 正式动作请求不解码未来 RGB。请求统计已区分实际执行、零执行过期和回合末未送达计划。该控制器仍采用离线等效延迟，而非墙钟实时并行。
- 2026-11-04 已运行相关 CPU 回归 103 项、三组策略的模拟闭环循环回归 12 项，均通过；真实 RGB-action 八任务预检为 400 条轨迹、91,935 窗口，检查 24 个真实窗口和 100 组双卡均衡采样。action-only 全量预检无动作裁剪，归一化往返最大误差约 `4.5e-8`，400 个轨迹末尾的动作 mask 均正确。预检记录见 [RGB-action](../tmp/ewm-simple-smoke-3muzuj7v/rgb_preflight.json) 和 [action-only](../tmp/ewm-simple-smoke-3muzuj7v/action_preflight.json)。
- 用户要求短测改用 RTX A6000。原 Ada 作业 `245503` / `245524` 在开始前已取消；新的三组 100 step 短训练作业为 `245652`（2 × A6000），三组 45 步自由落体 continuous 闭环验收为依赖作业 `245653`（1 × A6000），仅在短训练全部成功后启动。两者使用同一份冻结源码、数据、归一化和权重指纹；作业完成前不能宣称 GPU 短测通过，短回合不作为成功率结果。
- 集群对 A6000 自动限制主机内存：双卡训练分配 44 GiB，单卡闭环分配 22 GiB，低于脚本原申请值。短测需要验证该额度是否足够；资源不足时保留失败记录并调整作业资源，不改变模型或数据条件。
- 正式 80k 训练和 1,200 回合 ID 评测尚未启动。旧 checkpoint 不充当本轮同配置基线：历史 joint 243038 为 action=40/event=16；action-only 237659 的数据/normalization 不同；239872 还使用两时刻 RGB。

## DynamicVLA 核查记录

核查日期：2026-11-04；官方代码版本：`434dc7105cf6d559018d15af5fecbcdad44f036e`。

| 项目 | 已核实事实 |
| --- | --- |
| 预测 horizon | 官方训练配置 `CHUNK_SIZE=20`、`N_ACTION_STEPS=20`；不能把通用配置类默认值 50 当作论文设置 |
| 执行方式 | streaming 按当前观测 index 与预测观测 index 的差值跳过过期动作，新结果更新队列 |
| 仿真控制周期 | 官方环境 `sim.dt=0.04`、`decimation=1`，即 25 Hz |
| 单次推理延迟 | 论文附录 Table IV 的 16 层模型在 RTX A6000 上为 0.226 s |
| 实际执行均值 | 在本次检查的论文和官方代码中未找到直接报告的跨回合统计值 |

**推算而非实测：**忽略额外通信/调度等待、队列耗尽及回合边界，在持续推理稳态下，一份新计划约在下一次推理完成后被替换，因此 `25 × 0.226 = 5.65`，对应约 **5–6 个实际执行动作/plan**。20 是预测长度；扣除过期前缀后的剩余长度也不是实际执行均值。

若仅将相同的 226 ms 代入本项目 30 Hz 时钟，则为 6.78 个 tick；当前向上取整实现对应约 7 步。这只是量纲示例，不能代替三组 EWM 的实际测量，也不能据此把执行长度硬编码成 7。

论文还写有约 88 Hz，但没有在该句明确区分 plan 频率与动作吞吐。`20 / 0.226 ≈ 88.5` 在数值上吻合动作吞吐，这是解释上的推断；本文使用明确标为 inference time 的 Table IV 数据，不将 88 Hz 解释成每秒 88 次完整 plan。

资料：

- [DynamicVLA 论文，III-C/III-D 与附录 A/C/D](https://arxiv.org/html/2601.22153v1)。
- [官方训练配置](https://github.com/hzxie/DynamicVLA/blob/434dc7105cf6d559018d15af5fecbcdad44f036e/configs/dynamicvla.yaml)。
- [官方 streaming 队列与过期前缀逻辑](https://github.com/hzxie/DynamicVLA/blob/434dc7105cf6d559018d15af5fecbcdad44f036e/policies/dynamicvla/modeling_dynamicvla.py#L510-L565)。
- [官方仿真控制周期](https://github.com/hzxie/DynamicVLA/blob/434dc7105cf6d559018d15af5fecbcdad44f036e/simulations/configs/env_cfg.py#L137-L143)。

## 数据版本清单

以下摘自 event=24 后实际完成的训练预检记录；只是记录该次预检结果，正式提交前仍须重新核对磁盘数据。`windows` 使用 `pad_tail=true` 后的训练窗口数，不使用旧 source export 中去尾的窗口数。

| task | 轨迹数 | frames / padded windows |
| --- | ---: | ---: |
| `conveyor` | 50 | 15,993 |
| `moving_carton` | 50 | 25,181 |
| `turntable_cups` | 50 | 23,285 |
| `rotating_tube` | 50 | 14,750 |
| `tennis_rolling` | 50 | 4,235 |
| `tennis_falling` | 50 | 2,024 |
| `tennis_parabolic` | 50 | 3,822 |
| `tennis_container` | 50 | 2,645 |

固定各源导出的 `input_signature` 与 `collection_sha256`，不能只凭目录名称和帧数判断版本：

```json
{
  "conveyor": {
    "input_signature": "c01736c091c96e45e81ac717fc1484d3012b59ca47704f3151c772d8a05b7a74",
    "collection_sha256": "627fc2c230a50de824fde763cf87a2302aa042b5b807acddd125ca8d854b8677"
  },
  "moving_carton": {
    "input_signature": "93a092309f10a5a3e2a40233477f27f56d197dc3bc3ee05d01c7691d72263bfe",
    "collection_sha256": "aa4008c7f79f0dd57abd68efd49a2552fca328757357143443032d2a2c6465fe"
  },
  "turntable_cups": {
    "input_signature": "a11d9f5464f248ea1cace82229ca0133255d2a75803ab3fdeaa69322b10f22ff",
    "collection_sha256": "4835c314f4b6be8a341b7039ed2af0c17b7e9ed21a4d0cb91a9ec596ed19766f"
  },
  "rotating_tube": {
    "input_signature": "1fd008e35f86a83659f16503d6446b3a3bd903d457b8c41fcbe0eaffc7fbee30",
    "collection_sha256": "f3651126dfd4986a6b396e9a3b9bc5098c1b4b7ca966df31eeeff23011fc1d16"
  },
  "tennis_rolling": {
    "input_signature": "b4bf43f1c2d44a2e619ddf3394b0e10834a16918ebe113cde6e28e91cb03670f",
    "collection_sha256": "630185dc7e128fba99caefbead473b7bb20bf151439090b0a369aa6d938fc34e"
  },
  "tennis_falling": {
    "input_signature": "b7249027ea7ba3c747b6122acd3750fed04b8d6c839662a292f438cf8d564df2",
    "collection_sha256": "58c422ac86fef953e57b9f5e01f2d59167dd2930b07942ab4f8dd3b97b59bbd6"
  },
  "tennis_parabolic": {
    "input_signature": "0bd7fc5e9a41c3d9d33bea556586907393b93a7b606b10bb70c15db254dfd1ae",
    "collection_sha256": "16da46d0f872062b8334fe690667edc925fb4dd28793251504decb3a078bac9f"
  },
  "tennis_container": {
    "input_signature": "983f09b1d551661e64b00d5ce55a1cde75698868974a9ecd7268e8ede0c54f1b",
    "collection_sha256": "62aa7a3fb630868e5f3aac0e69e5b58feb0b6db7b3621fc9e1dfefa3346d8738"
  }
}
```

## 2026-11-04 worker 清理与正式训练准备

- A6000 三组 100 step 短训练在保存最终 checkpoint 后，joint 与 RGB-action 的持久 DataLoader worker 在析构清理时出现 SIGABRT，而主进程仍可能返回 0。真实 joint 数据以 batch=32、4 个 spawn/persistent worker 重现了该现象；底层 C++ 终止的具体库原因尚未确认。
- 训练结束时改为停止提交新样本、排空已有预取并显式关闭 worker，预取异常向上报告。真实 joint 和 RGB-action 各完成一次同 batch/worker 数的退出对照，均正常；新增清理回归 5 项通过。该处理依赖当前 PyTorch 私有迭代器接口，必须使用冻结环境并以双卡真实恢复与退出结果为放行条件。
- 计算节点 CPU 回归作业 `246608`：worker 清理、恢复及三组数据集相关测试共 29 项通过。双卡 A6000 验收作业 `246829` 已提交并暂时 hold，等待缓存内容哈希和修复后正式源码快照；验收包括三组各 4 步、从第 2 步恢复至第 4 步，比较权重、优化器、调度器、两 rank 随机状态、loss 和 worker 错误。尚未运行或通过，不能据此宣布正式训练已放行。
- 正式准备脚本对冻结源码、归一化和预训练权重使用 SHA256；对约 90 GiB 的 400 个 DINO 缓存数组逐文件计算并在作业启动时重算 SHA256。训练导出的 parquet、视频和事件原始文件仍沿用文件大小、mtime、导出元数据及 collection 指纹，不能宣称已逐字节全量哈希。全量缓存哈希生成中。
- 400 个 DINO 缓存数组的完整 SHA256 已于 2026-11-04 生成（约 762 秒）。含 worker 修复的正式源码和共同输入清单已冻结到 `tmp/ewm_formal_workerfix_20261104/`；准备时复用上述哈希并复核缓存清单、大小和 mtime，验收及每个正式作业启动时仍会逐文件重算完整哈希。
- 双卡恢复验收作业 `246829` 已放行排队。三组 80k 作业：action-only `247316`、joint `247317`、RGB-action `247318`；均使用 `afterok:246829`，只有验收作业成功才会启动，验收失败不得把作业提交解释为训练已开始。A6000 每组申请两卡；集群并发额度可能使作业继续排队。

## 2026-11-04 Ada 编译串行训练调整

用户调整为 2 × RTX 6000 Ada、训练启用 `torch.compile`、三组在一个作业内依次运行，并直接申请 3 天；上述 A6000 验收 `246829` 和三个训练作业 `247316`/`247317`/`247318` 均在运行前取消，不再执行。

- 固定源码与输入的 Ada 批次为 `tmp/ewm_formal_ada_serial_compile_20261104/`。顺序为 action-only → joint → RGB-action；三组各 80k、每 20k 保存 policy 和完整恢复状态。正式训练作业为 `247640`，申请 2 × 6000ada、8 CPU、128 GiB、72 小时。
- Ada 双卡编译恢复验收作业为 `247639`，正式作业设置 `afterok:247639`。验收检查三组两个 rank 的编译启用、从第 2 步恢复至第 4 步的状态与损失以及 worker 清理；截至提交时验收与正式训练均尚未开始。验收作业单独申请 3 小时，编译及全量数据指纹校验耗时尚未在本配置实测。
- 串行脚本在组完成且退出检查通过后写完成标志；同一批次重提交时跳过已确认完成的组，并从未完成组最近的完整状态恢复。首次 20k 保存前中断将没有可恢复的完整状态，脚本会保留现场并停止。集群没有证据表明达到 72 小时会自动续期或重排，超时时需要重新提交续跑；若 80k 状态存在但缺少通过退出检查的完成标志，也会停下等待检查。
- 更新：`247639` 的三组从头训练均完成 4 步，但恢复入口经 Accelerate 导入 DeepSpeed 时因缺少 CUDA toolkit 路径失败；`247640` 已在运行前取消。恢复验收启动器补齐 CUDA 12.8 环境并提前检查 unwrap，冻结源码的 7 项 CPU 恢复测试通过，已重提验收 `248817`，GPU 验收仍待执行。
- 用户要求验收成功后自动开始正式训练，现提交 `248934`，设置 `afterok:248817`，资源仍为 2 × 6000ada、8 CPU、128 GiB、72 小时。依赖启动器 `tmp/ewm-formal-after-resume-248817-20261104/train_after_resume.sbatch` 继承 CUDA 修复，在启动前确认三份 `comparison.json` 均通过，再执行同一冻结批次的串行训练脚本；训练顺序、每组 80k 和每 20k 保存完整状态均不变。正式输出为该批次的 `runs/{action_only,joint,rgb_action}`，顶层日志为 `logs/ewm_formal_ada_after_resume_248934.log`。提交依赖作业不代表验收已通过或正式训练已开始。

## 2026-11-04 用户确认先启动正式训练

- 验收 `248817` 的三组连续训练和第 2 步恢复训练均正常完成至第 4 步，CUDA 环境、worker 退出、两 rank RNG、步数及策略权重容差检查通过；优化器一阶动量与部分 loss/梯度范数超过一致性容差，验收因此失败。差异来源仍待对照定位，不宣称恢复一致性已通过。
- 用户明确要求先提交正式训练。原依赖作业 `248934` 已取消；现提交 `262022`，不设置验证作业依赖，使用 `tmp/ewm-formal-direct-20261104/train_formal.sbatch` 启动同一冻结批次。资源仍为 2 × RTX 6000 Ada、8 CPU、128 GiB、72 小时；action-only → joint → RGB-action 串行，每组从头训练 80k，BF16、compile=true、global batch=64，每 20k 保存 policy 和完整恢复状态。提交时 `runs/` 尚不存在。
- 保留 CUDA 12.8 设置、冻结源码/输入检查、运行锁、检查点和退出检查。提交前复核 355 份源码哈希、预训练权重、归一化、数据指纹及 400 个 DINO 数组的大小/mtime；作业启动时仍重算全部 DINO 内容 SHA256。冻结源码和失败的比较报告未修改。日志为 `logs/ewm_formal_ada_serial_262022.log`，正式输出仍为 `tmp/ewm_formal_ada_serial_compile_20261104/runs/`。

## 2026-11-04 追加 L40 时间采样间隔 1 的 EWM 对照

- 用户要求在 Ada 三组正式任务 `262022` 之外，并行提交一组 EWM joint，视觉时间采样间隔改为 1，使用 2 × L40、32 小时；用户明确选择开启实验性 L40 编译。已提交独立作业 `262686`，无依赖关系，不修改或取消 Ada 作业。L40 分区按 2 GPU 分配 8 CPU、96 GiB 主机内存。
- 本组参数为 `model.event_temporal_downsample=1`、`training.compile_core=true`、`training.experimental_l40_compile=true`；动作和事件预测终点仍为 24 个控制 tick，视觉使用未来 24 帧，经时间压缩 4 后每相机有 6 个未来 latent 时间片。其余沿用同一八任务数据、seed=42、BF16、global batch=64、80k optimizer steps、每 20k 保存 policy 和完整状态，从头训练。该组硬件与时间采样均不同，结果应单独标注，不作为只改变采样间隔的严格同硬件对照。
- 冻结批次为 `tmp/ewm_formal_joint_stride1_l40_20261104/`，源码和输入清单复用 Ada 冻结版本；独立 `experiment.json`、`overrides.json`、`resolved_config.yaml` 记录实际配置。启动脚本检查型号、源码/输入指纹及完整 DINO 内容哈希；训练后检查双 rank 编译启用、退出和 80k 完整恢复状态，不把 eager fallback 标为编译成功。
- 提交前 Hydra 配置解析与时间采样几何检查通过，使用冻结源码的现有 joint 时间采样测试 3 项通过；355 个源码哈希、输入指纹和 400 个 DINO 数组大小/mtime 匹配。L40 GPU 训练尚未执行，不宣称实验性编译或长程稳定性已验证；恢复一致性差异仍未解决。
- 正式输出为 `OpenWAM/outputs/ewm_joint_stride1_l40_20261104/`，日志为 `logs/ewm_joint_stride1_l40_262686.log`。32 小时是作业时间上限，训练仍以 80k 步为目标；达到时限中止时保留完整检查点，未设置自动续跑。

## 2026-11-05 L40S 兼容修复及重提

- 作业 `262686` 在启动前失败：Slurm 的 `gpu:l40` 分配了两张 NVIDIA L40S，启动脚本只接受 L40，训练没有开始。
- 修复启动器的 L40/L40S 型号检查，并使训练代码在明确开启 `training.experimental_l40_compile=true` 时也允许 L40S 双卡编译；其他 GPU 的编译保护不变。冻结批次 `tmp/ewm_formal_joint_stride1_l40s_fix_20261105/` 仅变更该训练代码，重新生成源码清单和校验和；原批次保留。
- 按原资源及训练配置重提作业 `273909`：2 × Slurm `gpu:l40`、32 小时、joint、时间采样间隔 1、实验性编译启用，提交时排队等待资源。型号判断的 CPU 回归测试、脚本语法、Slurm 提交预检和 355 个冻结源码哈希检查通过；L40S 的实际编译及训练稳定性尚未验证。

## 2026-11-05 连续推理最少执行 4 步

- 用户要求连续推理每个已启用动作计划至少执行 4 个有效动作后再切换，作为该控制器默认设置；连续推理与执行仍可并行，新结果覆盖未启用的旧候选，并按观测时刻丢弃过期前缀。推理仍单请求在途，已到达未启用的候选与真实在途请求分开记录。
- 当前计划耗尽、无有效计划或回合结束时允许不足 4 步，记录原因；无有效动作时只保持夹爪、相对位移和旋转归零。记录请求间隔、实际切换间隔、被覆盖和回合结束未启用的计划，不能把推理请求频率误写为切换频率。
- `--latency-controller continuous` 默认 `--min-execution-steps 4`，指定 1 可复现原立即切换调度；`buffered`/`serial` 的执行设置保持原有行为。八任务和单任务 manifest 记录实际下限，避免与旧协议结果混用。
- 用户要求取消旧测评 `273942` 和依赖测评 `274007`，两者已取消；旧测评保留部分结果，不作为完整的 4 步下限结果。本次实现通过 CPU 单元和模拟闭环检查，真实 GPU/Isaac 测评尚未运行。
