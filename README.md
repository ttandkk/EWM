# EWM：事件世界模型与 Piper 八任务实验

本仓库包含 EWM、Action-only、RGB-action 的训练与闭环测评代码，Piper 双相机 RGB/event 数据采集，以及 OpenWAM/FastWAM 的 Piper LoRA 基线适配。

当前八任务为 `conveyor`、`moving_carton`、`turntable_cups`、`rotating_tube`、`tennis_rolling`、`tennis_falling`、`tennis_parabolic`、`tennis_container`。

## 安装与统一入口

任务、传感器、成功判据、采集与测评的唯一实现已迁入可安装的 `benchmark/src/piper_bench`；EWM、OpenWAM、FastWAM 的模型加载与归一化由独立的 `policies/src/piper_policies` 服务负责。仿真不再读取模型的 Hydra 配置。

当前源码发布不包含根目录 `migration/`、`Event-DataCollection/`、`assets/` 和 `archive/`。运行所需的小型场景资产、依赖锁和 FastWAM 补丁随 `benchmark` 包提供；新部署使用以下统一入口。旧实验记录保留在原工作区，不属于发布内容。

```bash
# 从仓库启动安装器只需要系统 Python；安装到新的独立前缀。
./ewm setup --profile simulation --prefix /path/to/piper-runtime --dry-run
./ewm setup --profile simulation --prefix /path/to/piper-runtime --accept-eula
export PIPER_CONFIG=/path/to/piper-runtime/deployment.json
./ewm assets download
./ewm doctor --render --events --output deployment_report.json
./ewm demo --task tennis_rolling --oracle --output-dir demo_tennis
```

本次部署采用命令行原生安装，范围为相互隔离的仿真与采集 writer 环境；不要求容器，模型侧环境安装暂不纳入本次验收。NVIDIA 授权确认、驱动要求、Python 环境、53 个公开/包内资产及仍需联网的 NVIDIA 材质见 [部署指南](docs/deployment.md)。`doctor` 会分别报告环境、双 RGB/腕相机、EVIS、Oracle；只有真实检查通过才算部署验收。当前验证范围见 [迁移验证记录](docs/migration-validation.md)。

只需 CPU 场景规划时，可以在独立 Python 环境中安装轻量包：

```bash
python -m pip install './benchmark[client]'
piper-bench evaluate --plan-only --output-dir new_scene_plan
# 也支持从源码运行 ./ewm，以及安装后的 python -m piper_bench。
```

新旧测评入口默认均为 `continuous_min8_v1`，`background_ood` 默认青灰桌面 `piper_eight_dynamic_background_ood_v5_grey_green`。八任务的采样/判据/相机版本和固定时间预算由 [suite 清单](benchmark/src/piper_bench/resources/suites/piper_eight_v1.json) 固定；历史 buffered/min4 与背景 v4 必须显式选择。诊断性改变控制参数、seed、时间预算或 flow 步数需标记 `--interface-only`。

如需策略测评，可在另行准备的模型环境运行 [独立策略服务](policies/README.md)，再将 ready 文件交给仿真进程：

```bash
./ewm evaluate --task tennis_rolling --ready-file /tmp/ewm-policy.json \
  --condition background_ood --output-dir new_eval --headless --enable_cameras
./ewm replay --result-dir new_eval --rejudge-only --output-dir terminal_audit
./ewm replay --result-dir new_eval --trial 0 --output-dir fixed_action_replay
# 采集导出还需要独立 writer profile；所有命令支持 --help。
./ewm collect --help
./ewm validate --help
```

整套测评使用 `--policy-url` 或 task→URL 的 `--policy-map`，可用 `--preflight` 先检查服务合同。恢复严格检查场景、模型和源码身份，数据路径可以迁移。原始训练集的旧签名仅覆盖元数据，不能当作全部视频/动作的内容校验。源码冻结与 Slurm 资源参数见 [launch 说明](launch/README.md)。

## 当前实验与正式测评入口

保留五组 80k 模型；预测时域来自各自 checkpoint 配置，测评不修改网络结构。

| 模型 | 训练来源 | 动作 / 视觉预测时域 | 视觉时间间隔 |
| --- | --- | --- | --- |
| Action-only | 262022 | 24 / 无 | 不适用 |
| RGB-action | 262022 | 24 / 24 | 2 |
| 对称 EWM | 262022 | 24 / 24 | 2 |
| 对称 EWM，间隔 1 | 276490 | 24 / 24 | 1 |
| 非对称 EWM | 243038 | 40 / 16 | 2 |

正式测评使用 [eval_piper_final.sh](scripts/eval_piper_final.sh) 和 [统一协议配置](scripts/eval_profiles/piper_eight_final.json)：30 Hz、continuous、不间断推理并丢弃过期动作、每份活动计划至少执行 8 步、4 步 flow、实测回复等效延迟。开启编译和上下文 KV 缓存，关闭 CUDA Graph；编译动作误差仅记录。视觉预测在回合结束后解码，不计入等效延迟，Action-only 不录制视觉预测。

详细说明见 [正式测评协议](docs/evaluation_protocol.md)。每条件为 8 任务 × 50 回合；`speed_ood` 的每任务 50 回合包含 25 慢速和 25 快速。背景使用青灰桌面 v5。`eval_piper_eight_dynamic.sh` 与统一入口使用相同默认协议，旧配置通过 `--execution-profile legacy_buffered_min4_v1` 显式选择。

```bash
# 在已部署环境的仓库根目录运行；只打印命令，不申请 GPU。
bash scripts/eval_piper_final.sh \
  --checkpoint-dir /path/to/selected/run \
  --checkpoint-step 80000 \
  --output-dir /path/to/new/evaluation --conditions id --dry-run

# CPU 预检会检查真实 checkpoint 并写场景清单；不申请 GPU。
bash scripts/eval_piper_final.sh \
  --checkpoint-dir /path/to/selected/run \
  --output-dir /path/to/new/evaluation --conditions id --preflight

# 以下命令会提交 GPU 作业；按测评范围选择资源和时限。
EWM_EVAL_GRES=gpu:rtx5090:1 EWM_EVAL_TIME=12:00:00 \
  bash scripts/eval_piper_final.sh \
  --checkpoint-dir /path/to/selected/run \
  --output-dir /path/to/new/evaluation --conditions id
```

OOD 条件名为 `lighting_ood blur_ood background_ood speed_ood`，可通过 `--conditions` 指定一个或多个。多条件测评须设置足够的 `EWM_EVAL_TIME`；不要覆盖已有实验目录或运行中的冻结源码。

## 训练与基线

EWM 三组八任务训练预设：

- `OpenWAM/configs/train_ewm_eight_action_only.yaml`
- `OpenWAM/configs/train_ewm_eight_rgb_action.yaml`
- `OpenWAM/configs/train_ewm_eight.yaml`

训练预算和数据契约见 [八任务实验计划](docs/ewm_experiment_plan_20261104.md)；该文历史推理安排由当前正式测评协议覆盖。数据及 DINO/event 缓存需要提前准备；已清理过的缓存不能仅凭旧配置路径视为仍然存在。推理不需要全部训练缓存。

OpenWAM/FastWAM 的视频 LoRA 和动作分支配置见 [Piper RGB 基线](docs/piper_rgb_baselines.md)，入口为 `scripts/train_piper_rgb_baseline.sh`。当前采用 BF16、无梯度累积；随机初始化容量和吞吐短测不能替代正式预训练权重或收敛验证。

## 文件与结果导航

| 目录 | 内容 |
| --- | --- |
| `OpenWAM/` | EWM/OpenWAM 模型、数据加载器、训练配置、推理服务与测试 |
| `benchmark/` | 可安装的任务、传感器、采集/测评、版本化 suite 与部署锁 |
| `policies/` | EWM/OpenWAM/FastWAM 独立推理服务和模型资源迁移 |
| `launch/` | 完整源码冻结与独立 Slurm 资源配置 |
| `deployment/` | 命令行安装入口，以及现有环境的选择辅助脚本 |
| `scripts/` | 当前采集、训练、正式测评入口及数据准备工具 |
| `docs/` | [文档索引](docs/README.md)、当前协议、实验和产物索引 |
| `results/`、`logs/` | 本地测评产物和作业日志，不随源码上传 |
| `tmp/` | 本地实验、冻结源码和部分正式模型，不能整目录删除 |

五模型最终 ID/OOD 路径、分段补测关系，以及 `results/`、`tmp/`、`logs/` 的保留/删除候选见 [产物保留清单](docs/artifact_retention_20261110.md)。event/RGB 冗余度分析的 latent、PCA、图表、脚本与输入依赖单独保护。清单只提供建议，不执行清理。

## 部署与源码来源

本次新部署使用 [固定安装器和锁](docs/deployment.md)，独立安装仿真与 LeRobot writer 环境。第三方版本位于 [发布锁](benchmark/src/piper_bench/resources/deployment/simulation.lock.json)，由安装器按所选 profile 恢复。训练或策略测评所需的模型环境、数据和权重属于后续可选工作，不在本次部署验收范围内。

FastWAM 的固定上游版本、Piper/LoRA 补丁和补丁结果指纹都在 `benchmark/src/piper_bench/resources/deployment/` 内；可选的 `policy-fastwam` 安装计划使用这些材料，尚不代表全新模型环境验收通过。GR00T 属于历史基线，源码适配未纳入本次精简发布。

`deployment/runtime_env.sh` 与 `deployment/activate_training.sh` 供已有本地环境的实验启动器使用，不执行安装。新机器优先使用 `./ewm setup` 和生成的 `PIPER_CONFIG`。`scripts/` 中仍有针对旧账号或旧目录的历史作业脚本，新仿真、采集和测评使用 `./ewm`；集群资源入口见 `launch/README.md`。

数据、权重、环境、视频、缓存和作业输出不进入源码仓库。OpenWAM 最初源版本为 `f6d9f1059eb63a8a76dc60e07da2dad21615a161`，采集源码最初版本为 `7b2967f72ae27a21cc7ae4c1020992905679d251`；本项目包含两者之上的本地实现。相关许可和来源说明随 [包内 notices](benchmark/src/piper_bench/resources/licenses/) 与 OpenWAM 源码保留。
