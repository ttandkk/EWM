# CVPR2027 project guide

This file defines the local conventions for agents and contributors working in this repository.

> **2026-10-12 migration priority:** The destination account cannot access the source account directory. Follow the detailed independent-deployment handoff at the end of this file. Source-machine absolute paths below are historical references; destination jobs must use the new account workspace. Do not submit GPU jobs from the exhausted source account. The user explicitly chose fresh environment installation; do not retry cross-account sharing or permissions as the migration plan.

## Paths and layout

The canonical workspace is `/projects/_hdd/hdd/myproject/CVPR2027`. The server also exposes it through `/projects/hdd/myproject/CVPR2027`; use the latter in Slurm scripts and user-facing paths.

- `OpenWAM/`: main source tree, Hydra configs, tests, and durable training outputs.
- `pretrained/`: local DINOv3, T5, Wan2.2 and LeanVAE checkpoints. Reuse these weights; do not download duplicate copies.
- `datasets/`: exported Piper event dataset and its normalization metadata.
- `docs/`: architecture specifications, implementation status, and experiment reports.
- `logs/`: all stdout/stderr logs from Slurm or long-running local processes.
- `tmp/`: isolated experimental code, exploratory scripts, and temporary experiment results. Do not place experimental changes in `OpenWAM/` until they are validated and explicitly promoted.

Keep durable model/data changes in `OpenWAM/`; keep one-off diagnostic code in `tmp/`. Do not modify or delete user files outside the task scope.

## GPU nodes and Slurm

All current GPU nodes are in `cluster02`. Inspect available resource names before a new hardware request:

```bash
sinfo -o '%P %G %D %N'
```

Interactive RTX 6000 Ada session:

```bash
srun --partition=cluster02 --gres=gpu:6000ada:1 --time=02:00:00 --pty bash
```

RTX Pro 6000 batch jobs use:

```bash
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:pro6000:1
#SBATCH --time=02:00:00
```

Submit a batch script and monitor it with:

```bash
sbatch path/to/job.sbatch
squeue -j <job_id> -o '%.18i %.8T %.10M %.20j %.20R'
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Only cancel the specific task that you started:

```bash
scancel <job_id>
```

Never issue an unscoped `scancel` command.

## Logs, outputs, and checkpoints

Use absolute log paths under `/projects/hdd/myproject/CVPR2027/logs` in every Slurm header:

```bash
#SBATCH --output=/projects/hdd/myproject/CVPR2027/logs/<experiment>_%j.log
#SBATCH --error=/projects/hdd/myproject/CVPR2027/logs/<experiment>_%j.log
```

Store structured metrics and experiment metadata under the corresponding run directory, normally `OpenWAM/outputs/<experiment>/` for promoted runs or `tmp/<experiment>/results/` for exploratory work.

Do not save checkpoints for smoke, scale, or overfit diagnostics unless requested. The EWM trainer supports:

```bash
training.save_checkpoints=false training.save_steps=0
```

## EWM experiment conventions

The production default is frozen **joint BF16 LeanVAE-16 latent flow**, configured in `OpenWAM/configs/model/ewm.yaml` and `OpenWAM/configs/train_ewm.yaml`. See [the integration guide](OpenWAM/docs/ewm_wan_vae.md) for implementation details and validation.

### Three EWM training modes

The model accepts `model.training_mode=joint|event_only|rgb_action`. Run all three through `OpenWAM/scripts/train.py` from the `OpenWAM/` directory; selecting a mode also requires its matching dataset type and codec. The default forecast horizon is 24 frames, and the commands below start a full training run unless training overrides are supplied.

| Mode and config | Data and conditioning | Prediction and loss |
| --- | --- | --- |
| `joint` — `configs/train_ewm.yaml` | Piper `datasets/piper_stack_fronttight_v3_50/job177683`; current dual RGB, current one-bin event frame, 8-D state and instruction | Future event frames plus aligned 7-D actions; `loss_event + loss_action` |
| `event_only` — `configs/train_ewm_event_only.yaml` | Completed REASSEMBLE 30 Hz and DVLA 25 Hz event-frame exports; current RGB/event and source-specific prompt. Source action and state are not read. | Future event frames only; `loss_event`. The action expert is disabled. |
| `rgb_action` — `configs/train_ewm_rgb_action_piper.yaml` | The same Piper dataset; current dual RGB, 8-D state and instruction. Event-frame payloads are not read per sample. | Future wrist RGB at original 256×256 resolution plus aligned 7-D actions; `loss_rgb + loss_action`. Uses `leanvae16_rgb_joint` to encode real RGB. |

Piper's `meta/piper_temporal_contract.json` defines `action[k]` as the command after observation `k` and before observation `k+1`. Both action-training modes supervise `action[k:k+24]`; RGB-action supervises RGB frames `k+1:k+25`. Its default target is wrist RGB while both RGB cameras condition the model. To predict both RGB views, set `model.event_camera_mode=dual` and `dataloader.event_camera_mode=dual` together. RGB-action checkpoints use `ewm_rgb_action_latent_v1`; event-mode checkpoints use `ewm_wan_latent_v2`.

Launch examples:

```bash
cd /projects/hdd/myproject/CVPR2027/OpenWAM
export LD_LIBRARY_PATH=/projects/hdd/myproject/CVPR2027/envs/torchcodec_ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY=/projects/hdd/myproject/CVPR2027/envs/openwam/bin/python

EWM_DATASET_DIR=/projects/hdd/myproject/CVPR2027/datasets/piper_stack_fronttight_v3_50/job177683 \
  "$PY" scripts/train.py --config-name train_ewm
"$PY" scripts/train.py --config-name train_ewm_event_only
"$PY" scripts/train.py --config-name train_ewm_rgb_action_piper
```

The event-only preset points to both `datasets/reassemble_training_eventframes_10ms` and `datasets/dvla-place7obj-roll-25fps-events-40ms-delta-trim`. Both exports have `.export_complete` markers; the DVLA 25 FPS export finished in Slurm job 178252 on 2026-10-02 China time. Set `dataloader.dvla_dir=null` only when intentionally training on REASSEMBLE alone. Event-only defaults to `dataloader.homogeneous_source_batches=true`: each micro-step uses one source across all GPUs, with approximately58:42 window-weighted sampling and source-local tail padding. Epoch shuffling is deterministic and supports full-state resume. Event-only matches Piper's non-overlapping event accumulation and overlapping training clips: current1+future24, `window_stride=1`, adjacent anchors share23 future targets. Native REASSEMBLE30Hz/DVLA25Hz timing remains33.3ms/40ms per event frame. Event-only canonicalizes training channels to `[OFF, ON]`: REASSEMBLE disk polarity0/1 retains its order; DVLA disk blue/red is empirically ON/OFF and is swapped online. It converts native `count > 0` to occupancy before area resizing, pads to256x256, and uses FP32 `[0,1]` without division by15. Input conventions align; event-density distributions remain source-dependent. The polarity evidence and sampled distributions are documented in `docs/dvla_reassemble_joint_pretraining.md` and `tmp/event_only_contract_audit/REPORT.md`. See `OpenWAM/docs/ewm_rgb_action_piper.md` for the RGB-action data contract.

### Piper event representation and flow

- Use `model.event_codec_impl=leanvae16_joint` and one-bin OFF/ON voxels (`model.voxel_bins=1`, `dataloader.voxel_bins=1`), normalized with `count_clip=15` and matching one-bin normalization metadata.
- Map normalized voxels to floating RGB `[OFF, (OFF+ON)/2, ON]`, then to Lean `[-0.5,0.5]` (Wan uses `[-1,1]`). Preserve both polarities; do not use signed cancellation or byte quantization.
- Jointly encode current1+future24 frames per camera. The causal VAE produces7 latent time slices: one observation condition plus six future slices. Decode the observation latent and generated future latents together, then discard the first reconstructed frame.
- Use `model.event_patch_size=4` and `model.event_spatial_compression=32` (Lean VAE stride8 times DiT patch4), with `model.event_temporal_compression=4`. At256x256, each Lean latent slice has16x32x32 values and64 DiT event tokens:448 total for wrist-only K24. Patch size is measured on the VAE latent grid: patch4 groups16 latent cells per token; total per-axis image-to-token stride is32. Patch8 is an optional stronger spatial compression with total stride64 and112 tokens. Wan alternative stride16 times patch2 also uses total stride32 and448 tokens. Actions retain24 steps. Future horizon must be divisible by4.
- Optional `wan22_frame` requires temporal stride1 and independent per-frame decoding; do not mix its latent representation with joint-mode checkpoints.
- Add `N(0,1)` noise **after** clean VAE encoding and train the velocity in the selected codec’s native latent coordinates. The old voxel `model.event_scale=10.0`, dense voxel codecs and learned temporal down/up projections have been removed; do not reintroduce their configuration fields.

`event_camera_mode` must match in both `model` and `dataloader`. Production defaults retain dual-view RGB context and wrist-only events:

```yaml
model.event_camera_mode: wrist_only
dataloader.event_camera_mode: wrist_only
```

Use `dual` in both fields for cam0 + wrist events; this doubles event tokens at the same horizon and resolution.

### Weights and training defaults

The VAE path is `model.event_vae_path`, defaulting to `/projects/hdd/myproject/CVPR2027/pretrained/leanvae`, overridable with `EWM_VAE_PATH`. The default requires LeanVAE-dim16.ckpt. Select `model=ewm_wan22` (launcher `EVENT_CODEC_IMPL=wan22_joint`) for Wan2.2, with its corresponding pretrained/wan2.2-vae directory. DINO, T5 and VAE stay frozen and in evaluation mode.

The default training configuration is Event D768 / Action D512, H8/L12/K24, BF16, batch32 per GPU on two RTX6000 Ada, gradient checkpointing disabled and `training.compile_core=false`. `model.codec_batch_size=16` limits the number of videos (sample×camera) passed to the VAE encoder per call in joint mode; it is not the training batch size. Profile before increasing batch size or enabling compilation.

VAE encoding currently runs online; no offline VAE latent cache is implemented. The new RGB/event memory variant separately supports an optional frozen-DINO history feature cache (see the migration handoff); this is not a VAE latent cache. Ordinary training steps do not invoke the decoder. Decode only for low-frequency diagnostics or final sampled voxel outputs.

The existing Piper joint launcher `OpenWAM/scripts/train_ewm/train_ewm.sh` explicitly selects D768 / Action D512, patch4 and dual events. The base Hydra defaults use patch4 and wrist-only events. This launcher is Piper-specific; mixed event-only training uses `--config-name train_ewm_event_only`. Actual temporal compression is4. Diagnostic launches can set `SAVE_CHECKPOINTS=false SAVE_STEPS=0`.

### Metrics, checkpoints and isolated diagnostics

`loss_event`, `latent_flow_mse` and `latent_x0_mse` are latent-space metrics. Do not compare their numeric scale directly to historical dense voxel losses or apply voxel active/inactive masks to latent channels.

Decoded `voxel_x0_*`, precision/recall/F1 and occupancy metrics run at `training.diagnostic_every=200` and the final step, using `model.diagnostic_samples=1` sample per process by default. These are teacher-noised x0 diagnostics, not full forecasting scores. The occupancy threshold is1/30 in normalized voxel units.

Policy checkpoint format is `ewm_wan_latent_v2`. Frozen VAE/DINO/T5 weights are external dependencies excluded from policy checkpoints. Old dense `ewm_trainable_v1` and full-state checkpoints are incompatible. Policy checkpoints do not contain optimizer/RNG state for exact training resume.

For strict single-sample flow-overfit diagnostics, keep code in `tmp/`, fix the sample, flow time and latent/action noise, and disable checkpoint saving. The old `tmp/ewm_overfit/` scripts target the removed dense architecture and must be adapted before reuse. Historical dense implementations and profilers are archived in `tmp/ewm_vae_integration/legacy/`; do not treat them as the active production path.

LeanVAE now uses BF16 weights and compute; its previous FP32-only integration is historical. Joint/single-observation latents need not be bitwise equal in BF16. Keep this distinct from future-dependence checks. Both default Lean patch4 and alternative Wan patch2 yield448 event tokens; full training speed results are in `tmp/ewm_lean_speed/REPORT.md`.


## 独立账号迁移交接（2026-10-12，Asia/Shanghai）

### 1. 用户目标、授权范围和当前状态

- 用户的旧账号 GPU 额度已经用完。新账号仍在同一集群，但不能访问旧账号项目目录；用户明确决定在新账号重新安装环境。
- 目标：迁移 EWM 训练/推理、Piper stack/conveyor 数据采集、EVIS event camera、Isaac Sim 闭环评估；新账号部署不依赖旧账号的目录。
- 当前源目录：`/projects/hdd/myproject/CVPR2027`，物理目录 `/projects/_hdd/hdd/myproject/CVPR2027`。新目录未知；后续 agent 应取得新账号实际可写目录，并定义 `NEW_PROJECT_ROOT`。不要把源路径再次写死到新环境。
- 本次授权：编写交接说明、准备 GitHub 代码上传内容。尚未取得最终 GitHub 目标地址时，不推送到 OpenWAM 官方上游，不自行把项目公开。
- 源项目根目录不是 Git 仓库；`OpenWAM/` 和 `Event-DataCollection/` 各自有 Git 历史。上传准备采用单独的首方代码快照仓库，两个项目作为普通子目录；第三方依赖通过固定 commit 恢复。不要把已有嵌套仓库当成自动有效的根仓库 submodule。
- 源 OpenWAM HEAD：`f6d9f1059eb63a8a76dc60e07da2dad21615a161`；remote 为 `https://github.com/OpenWAM-Official/OpenWAM.git`。
- 源采集仓库 HEAD：`7b2967f72ae27a21cc7ae4c1020992905679d251`；remote 为 `https://github.com/ttandkk/Event-DataCollection.git`。
- EWM、RGB/event memory 和 conveyor 采集的大量文件尚未进入这两个源仓库的提交；必须搬运当前工作树，不能只 clone 以上 HEAD。
- 用户已要求并完成清理：15 个 `outputs/**/resume_state_step_*` 目录（59 文件，约26.98 GiB）、`OpenWAM/benchmarks/`（99 文件，约710 KiB）、`OpenWAM/openwam.egg-info/`（6 文件）。Policy checkpoint 和评测产物保留；不再要求恢复精确续训状态。
- benchmarks 删除后已调整 Makefile、CI、Docker ignore；`tests/conftest.py` 在 benchmarks 缺失时忽略其旧测试。不要为了恢复原版 benchmark 而重新引入它。
- 上述删除后的 CPU 回归/收集曾被 Ceph metadata I/O 阻塞而终止。静态检查通过，但不能声称新账号已经跑通。

### 2. 新账号目录结构与上传内容

建议保留相对布局：

```text
$NEW_PROJECT_ROOT/
  AGENTS.md
  OpenWAM/                   # 首方源码、配置、tests、docs、vendored LeanVAE
  Event-DataCollection/       # 当前采集源码、tests、tools、必要纹理资产
    third_party/             # 按 migration/sources.json 恢复
  scripts/                   # 根目录采集/转换/Slurm 启动器
  docs/
  assets/                    # 项目必要的小型资产
  migration/                 # 上传快照中的依赖、固定源码版本、恢复工具
  datasets/                  # 数据另传，不进 Git
  pretrained/                # 权重另传，不进 Git
  envs/openwam/
  envs/lerobot/
  envs/torchcodec_ffmpeg7/
  isaacsim601/env/
  isaacsim601/IsaacLab/        # 真正使用的 runtime commit
  logs/
  tmp/
```

GitHub 代码快照应包含：当前 OpenWAM、采集仓库首方源码（含未跟踪文件）、根 AGENTS、根 scripts/docs、必要小资产、migration 清单。保留 LICENSE、NOTICE、THIRD_PARTY_NOTICES 和 vendored LeanVAE 许可。

不要上传：安装好的环境、Isaac runtime、数据、checkpoint、outputs、wandb、logs、cache、`.git/` 嵌套元数据、`.orig`、字节码、个人密钥或 `.env`。不复制整个112 GiB源目录。源码快照会包含嵌套 `.gitmodules` 作为来源记录，但没有嵌套 Git 历史或 gitlink；恢复第三方依赖用 `migration/restore_sources.py`，不要在快照中直接假设 `git submodule update` 有效。

### 3. 当前数据、权重和 checkpoint 清单

实地检查源 `datasets/` 当前只有下面三个 Piper 数据父目录；正式搬运时确认具体 job 子目录和 `.export_complete`：

- `datasets/piper_stack_fronttight_v3_50/`（训练默认 anchor 数据为 `job177683`）。
- `datasets/piper_conveyor_front_wrist_v3_random_50/`（历史50集导出在 `shards/job181776`）。
- `datasets/piper_conveyor_uniform_0p04_0p08_front_wrist_v3_50/`。

**当前根目录没有** `datasets/reassemble_training_eventframes_10ms`、`datasets/dvla-place7obj-roll-25fps-events-40ms-delta-trim`。旧 guide/config 提到了它们，但不能把那些历史描述当成当前文件存在。需要 event-only 预训练时另外定位/恢复导出数据；不要运行会因不存在路径报错的默认配置，也不要未经要求重新采全量数据。

数据必须保持完整目录：`meta/`、Parquet、RGB videos、双摄像头 event-frame PNG、normalization stats、`piper_temporal_contract.json`、完成/审计标记。不能只搬视频。迁移前后保存文件数、总字节数及 SHA256；大量数据可先做清单，传输后按清单逐文件校验，不要把抽样校验描述为全量校验。

最低必要预训练权重：

- `pretrained/dinov3-vitl16/`：包括 config 和权重 shard/index。
- `pretrained/t5-small/`：包括 tokenizer 配置/词表和权重。
- `pretrained/leanvae/LeanVAE-dim16.ckpt` 及该目录配套文件。
- `pretrained/wan2.2-vae/`：只在使用 Wan codec 或旧 checkpoint 时需要。`svd-vae`、`wan2.1-vae` 不属于当前 Lean EWM 最低依赖。

当前全部 datasets 和 pretrained 合计约17 GiB；测量是源目录占用，不是已打包/已传输的证明。

Policy checkpoint 通常约0.99 GiB/文件。只挑用户需要的模型，连同该 run 的 `config.yaml`、normalization 配置和关键评测指标迁移。候选包括 stack global/wrist、conveyor uniform joint、action-only/RGB-action 对比模型、DVLA pretrain/finetune、dense `job179463`。不要替用户删除或选择性丢弃尚未指定的模型。

旧 snapshot 文件有硬链接，搬运按实际选定权重处理，防止把 `checkpoint.pt` 与同一 inode 的 `checkpoint_step_*.pt` 重复传输。Policy 不包含冻结 DINO/T5/VAE 权重，也不能替代已删除的 optimizer/RNG 状态。新训练使用新的 output_path，不设旧 `resume_ckpt_path`。

数据/权重经旧账号主动传到新账号 SSH 入口、双方可访问中转区或用户指定存储；新的安装脚本不能假设能从旧目录直接 cp/rsync。凭据由用户通过正常认证提供，不写进 Git。

### 4. 已导出的精确依赖与三个独立 Python 环境

源机器依赖导出位于 `tmp/migration_bundle/`；GitHub 快照中复制到 `migration/dependencies/`。另有 `tmp/migration_dependencies.tar.gz`，这是依赖材料，不是完整项目代码包。

- `*.installed.json`：已安装 distribution 名称/版本。
- `*.requirements.snapshot.txt`：排除 editable 包后的版本快照。
- `*.editable.json`：需在新路径重建的源码安装项。
- `*.conda-packages.json`：Python 环境中的 Conda 包构建记录。
- `ffmpeg.conda-packages.json`：独立 FFmpeg 原生库环境（127 条记录）。
- `versions.lock`：采集和第三方固定 commit。
- `manifest.json`：快照数量和关键版本；共403 distributions，OpenWAM133、LeRobot53、Isaac217。

**这些是现状快照，不是已验证的可直接 pip install 的锁文件。** Conda 原生库、不同 wheel 索引、框架内部 distributions、已知元数据冲突、源码安装项必须分别处理。没有进行新账号安装、pip resolver 或 GPU 验证。

| 环境 | Python | Torch / torchvision | 其他关键版本 |
| --- | --- | --- | --- |
| OpenWAM | 3.10 | `2.7.1+cu128` / `0.22.1+cu128` | numpy2.2.6、transformers5.17.0、diffusers0.40.0、torchcodec0.5 |
| LeRobot writer/convert | 3.12 | `2.7.1+cpu` / `0.22.1+cpu` | numpy2.2.6、lerobot0.6.1（固定源码）；现状还有torchcodec0.11.1 |
| Isaac sim | 3.12 | `2.11.0+cu128` / `0.26.0+cu128` | isaacsim6.0.1.0、isaaclab6.1.14、numpy2.3.1、transformers4.57.6 |

三个环境不能合并；尤其 OpenWAM 与 Isaac 的 transformers/Torch 不同。采集脚本通过 `--writer-python` 把写入交给独立 LeRobot 进程，不要把 LeRobot 安进 sim 环境。

LeRobot 现状 torchcodec0.11.1 与 Torch2.7.1 的完整兼容性没有在本轮验证；writer/读视频先按现有 PyAV 路径验收，不把所有 installed 包自动当成必要或已兼容组件。

### 5. 固定第三方源码版本和源码审计

`migration/sources.json` 应记录以下 URL、commit、目标相对路径；恢复脚本 clone 后 detached checkout 对应 commit，拒绝覆盖已有不匹配仓库。

| 组件 | GitHub 上游 | 固定 commit |
| --- | --- | --- |
| 实际 runtime IsaacLab | `isaac-sim/IsaacLab` | `ffff603eafc6b74264a5261cc0183d6a65390d78`（v3.0.0-beta2.patch1） |
| EVIS | `spikelab-jhu/isaac-sim-event-camera-plugin` | `d6fe47923fad09c4bfa2fd2cb13f82520114cfae` |
| Agilex/Piper | `agilexrobotics/Agilex-College` | `c2688be41e1bc99a9addd555f237a16bc839d936` |
| robot_lab | `agilexrobotics/robot_lab` | `b868140eeb1459acefef24a865587ec39a5278c3` |
| LeRobot | `huggingface/lerobot` | `7e241bd630a3719a56157a497ce5d08f244784f1` |
| legacy IsaacLab，仅历史记录 | `isaac-sim/IsaacLab` | `37ddf626871758333d6ed89cf64ad702aef127d0` |

实际 source sim 的 editable IsaacLab 指向 `isaacsim601/IsaacLab/source/...`，不是采集仓库 legacy 子模块。legacy checkout 以前有恢复 stash/脏状态，不是新环境要安装的源码。

2026-10-12 审计：runtime IsaacLab、EVIS、Agilex、robot_lab、采集 LeRobot 的 tracked working tree 与上述 commit 一致；Agilex 的未跟踪内容为 Python bytecode，不需迁移。writer LeRobot 的检查结果以 `migration/source_audit/` 中记录为准。源码审计使用临时本地 LFS clean 归一化器，绕开源 PATH 缺少 git-lfs 的问题；没有调用 LFS 上传/下载，也没有修改第三方源码。新账号仍需安装真正 git-lfs 来取得任务必要的 LFS 资产。

clone 时可用 `GIT_LFS_SKIP_SMUDGE=1` 避免无关测试数据；之后检查 Piper URDF/meshes、EVIS texture/resource 文件是否仍是 LFS pointer，对必要路径执行 LFS 获取。不要跳过资产校验后声称仿真部署成功。

### 6. 推荐安装顺序（示例命令需要新路径）

先确认新账号 Linux/NVIDIA 驱动、集群可用 GPU、磁盘空间，以及可用 Python/Conda。若公共 Miniforge 路径对新账号不可用，就在新账号安装 Miniforge；不要调用旧 `envs/*/bin/python`。使用用户自己的 shell 环境，不覆盖 HOME/CODEX_HOME。

```bash
export NEW_PROJECT_ROOT=/new/account/writable/path/CVPR2027
cd "$NEW_PROJECT_ROOT"
python3 migration/restore_sources.py --root "$NEW_PROJECT_ROOT" --dry-run
# 检查输出后，实际恢复固定源码（需要网络，非 GPU）：
python3 migration/restore_sources.py --root "$NEW_PROJECT_ROOT"
```

OpenWAM：

```bash
conda create -p "$NEW_PROJECT_ROOT/envs/openwam" python=3.10 pip -y
OWPY="$NEW_PROJECT_ROOT/envs/openwam/bin/python"
"$OWPY" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
# 根据 openwam 快照安装普通依赖并约束关键版本，避免安装最新版本。
# 完成普通依赖后再绑定首方源码，防止 pyproject 的宽版本范围升级环境：
"$OWPY" -m pip install --no-deps -e "$NEW_PROJECT_ROOT/OpenWAM"
```

LeRobot：

```bash
conda create -p "$NEW_PROJECT_ROOT/envs/lerobot" python=3.12 pip -y
LRPY="$NEW_PROJECT_ROOT/envs/lerobot/bin/python"
"$LRPY" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
# 恢复 lerobot 快照中的所需普通依赖后：
"$LRPY" -m pip install --no-deps -e "$NEW_PROJECT_ROOT/third_party/lerobot"
```

Isaac Sim：可用新的 Python3.12 创建 venv；如果没有公共 Python，先在新账号 Conda 中创建 Python3.12。官方预编译包安装方式见 https://github.com/isaac-sim/IsaacSim/blob/develop/skills/isaac-sim-installation/references/python.md 。锁定现有版本，不升级到新的 Isaac/IsaacLab。

```bash
python3.12 -m venv "$NEW_PROJECT_ROOT/isaacsim601/env"
SIMPY="$NEW_PROJECT_ROOT/isaacsim601/env/bin/python"
"$SIMPY" -m pip install 'isaacsim[all,extscache]==6.0.1.0' --extra-index-url https://pypi.nvidia.com
# 按现状恢复 Torch2.11.0+cu128、numpy2.3.1 和普通依赖；确认包索引/可用性。
```

随后将 runtime IsaacLab 包、agx_teleop、dvs_gen 重新 editable 安装到 sim。源码 package 路径：

- `isaacsim601/IsaacLab/source/isaaclab`。
- `source/isaaclab_assets`、`source/isaaclab_tasks`、`source/isaaclab_physx`、`source/isaaclab_ppisp`、`source/isaaclab_teleop`、`source/isaaclab_visualizers`（都相对 runtime IsaacLab）。
- `Event-DataCollection/third_party/agilex-college/isaac_sim/agx_arm_IsaacLab/IsaacLab_Data_Collection/source/agx_teleop`。
- `Event-DataCollection/third_party/isaac-sim-event-camera-plugin`。

当前 sim 是任务验证过的精简组合；不要求补齐 Newton/XR/dev extras。EVIS 旧 metadata 要求 numpy<2，Isaac6.0.1 已验证组合是 numpy2.3.1。先恢复已验证普通依赖，再对这些固定源码包用 `pip install --no-deps -e <path>`，避免插件解析把 NumPy 降级。不要掩盖 pip check 的冲突；把已知 optional 缺失/元数据冲突与实际 import/runtime 错误分别记录。

FFmpeg：OpenWAM TorchCodec0.5 需要匹配 Torch2.7.1 和 FFmpeg 共享库；参考 `migration/dependencies/ffmpeg.conda-packages.json` 恢复 FFmpeg7，确认 `libavutil.so.59` 等存在。使用 TorchCodec 时将新 FFmpeg lib 加入 LD_LIBRARY_PATH。最初可用 `dataloader.rgb_decode_backend=pyav`，避免把原生解码问题和 EWM 核心问题混在一起。

### 7. 路径迁移：必须修改的运行入口

源码快照保留旧路径以忠实备份；新账号正式运行前必须完成路径改写并检查，不能仅 export NEW_PROJECT_ROOT 就认为所有 launcher 会读取它。

优先审计：

- `OpenWAM/scripts/train_ewm/train_ewm.sh` 的 ROOT、输出、日志、DINO/T5/VAE/data 路径；此 launcher 默认仍是 dense joint/action-only，不会自动选择新 memory preset。
- `OpenWAM/scripts/eval_ewm/run_*.sh`、`run_*.sbatch` 的 project_root、checkpoint、日志、Isaac cache/config。
- 根 `scripts/collect_piper*.sbatch`、`eval_piper*.sbatch`、`test_piper*.sbatch`、导出脚本。
- `OpenWAM/configs/train_ewm*.yaml`、model/dataloader 默认路径。
- Slurm `#SBATCH --output/--error` 必须写新账号的绝对 logs 路径；shell变量不会自动展开 Slurm 头。
- collector `--writer-python`、DINO cache manifest、复制的 checkpoint config 中旧权重/数据路径。
- 新账号的 W&B 登陆/entity、Slurm account/partition/gres 使用新账号实际可用配置；不要复制旧凭据。

```bash
export EWM_DATASET_DIR="$NEW_PROJECT_ROOT/datasets/piper_stack_fronttight_v3_50/job177683"
export EWM_DINO_PATH="$NEW_PROJECT_ROOT/pretrained/dinov3-vitl16"
export EWM_T5_PATH="$NEW_PROJECT_ROOT/pretrained/t5-small"
export EWM_VAE_PATH="$NEW_PROJECT_ROOT/pretrained/leanvae"
export XDG_CACHE_HOME="$NEW_PROJECT_ROOT/isaacsim601/cache"
export OMNI_USER_CONFIG_PATH="$NEW_PROJECT_ROOT/isaacsim601/config"
mkdir -p "$NEW_PROJECT_ROOT/logs" "$XDG_CACHE_HOME" "$OMNI_USER_CONFIG_PATH"
```

审计 `rg -n '/projects/(hdd|_hdd)|/home/haotian' OpenWAM/scripts OpenWAM/configs scripts Event-DataCollection/scripts`。文档中的源路径可以保留；实际入口、editable importer 和目标配置不得依赖旧目录。必要时修改复制的 checkpoint config；不改变相机数、codec、patch/horizon 或 normalization 的语义。

### 8. 当前 EWM / 分层记忆实现，迁移时不能漏掉

新配置 `OpenWAM/configs/train_ewm_rgb_event_memory_piper.yaml`，model type `ewm_rgb_event_memory`，checkpoint format `ewm_rgb_event_memory_latent_v1`。

- RGB4 时间点 [-9,-6,-3,0]，不做 RGB 差分，冻结 DINO 提取空间语义。
- event10 时间点 [-9..0]，旧 CNN 编码帧及相邻9个差分补充运动；双摄像头。
- RGB/event 是两层模态记忆，不是另加3个时间层；融合后作为只读 memory K/V 输入 DiT，不写入 noisy/future target。
- 新文件包括 `hierarchical_memory.py`、`hierarchical_ewm.py`、`hierarchical_dataset.py`、`scripts/train_ewm/cache_dino_memory.py`、`scripts/eval_ewm/ewm_memory_history.py`；还依赖旧 `ewm_memory.py`/`event_memory.py` 的共享代码。
- 历史 DINO cache 为 BF16 bits/uint16 mmap、8x8 pooled features，可选，不是离线 VAE cache；当前全数据 cache 尚未生成。搬迁后旧 cache 路径/源视频指纹可能无效，应重新校验或重建。
- 当前 dense、joint、event-only、RGB-action、action-only 对比路径保留；sparse 新训练已退役，旧 `sparse_token_ewm.py` 和部分测试仍用于旧 checkpoint 兼容。不要盲目删它们或通用导入链。
- 先前实现验证：204 CPU regression tests passed；新 memory 专项16项；GPU smoke job200343完成。在 PRO6000、B32/K24/dual 下 dense约0.674s/step，cached memory约0.764s，online memory约1.005s；peak约33.41/34.71GiB。是合成输入训练速度，不是新账号验收或闭环成功率。
- 尚未做 memory 的完整训练、全量 cache 和闭环效果验证。
- 迁移快照应带 `docs/ewm_rgb_event_memory.md` 及选定 migration/evidence 报告；不要把整个 tmp 实验目录塞进 Git。

### 9. 分阶段验收及失败处理

全部新账号验证都写入它自己的 logs/tmp，记录命令、环境版本、模型/数据 SHA、耗时、退出码；源账号不再申请 GPU。

1. 静态：确认代码含全部 EWM/collector 未跟踪文件；固定源码 commit 正确；必要资产不是 LFS pointer；各 JSON/requirements 可解析；源码语法检查。
2. CPU import：分别检查 Python版本、实际模块 __file__（必须新路径）、torch/transformers版本；sim用 `Event-DataCollection/tools/check_install.py --mode sim`，writer用 `--mode convert`。这些不是完整渲染验证。
3. 数据：读取真实 Piper anchor0和非起始窗口；检查双RGB/event、state/action、mask、normalization和时间契约；不要只检查目录存在。
4. 权重：加载冻结DINO/T5/LeanVAE以及选定policy，确认policy format和配置匹配。
5. 新账号GPU小测：先 `sinfo` 检查资源，几步双摄像头RGB/HDR/EVIS静态与运动探测；保留旧renderer和OFF/ON语义，避免renderer噪声变成假events。
6. 采集：一次成功episode，独立writer写LeRobot，执行对应validate脚本并解码两路RGB视频；再测conveyor运动/接触任务。
7. EWM：短训练（禁止checkpoint保存），再启动policy server+Isaac client短闭环；确认输出/日志/缓存均为新账号路径。
8. 所有必要小测通过后才正式训练或批量采数据。可选编译、TorchCodec、离线DINO cache放到基础路径验证之后。

最低训练 smoke 参考（需真实GPU和完整数据，非本轮已运行）：

```bash
cd "$NEW_PROJECT_ROOT/OpenWAM"
"$NEW_PROJECT_ROOT/envs/openwam/bin/python" scripts/train.py \
  --config-name train_ewm_rgb_event_memory_piper \
  training.max_steps=2 training.batch_size=1 dataloader.num_workers=0 \
  training.save_checkpoints=false training.save_steps=0 \
  training.compile_core=false training.output_path="$NEW_PROJECT_ROOT/tmp/migration_smoke"
```

如果 Ceph 元数据 I/O 持续阻塞，用 ps/wchan 区分 I/O 等待和模型错误；只停止本次 agent 启动的进程/Slurm job。不要把未完成测试标为通过。

### 10. GitHub 上传准备与后续 agent 的第一步

源根 AGENTS 和运行代码已经与迁移快照区分。上传准备位置以 `tmp/github_upload/` 内说明/清单为准，准备的是一个新的首方源码快照仓库，不更改 OpenWAM 官方 remote，不强推任何已有仓库。

上传前检查：

- 目标仓库是用户指定的仓库，建议私有。
- staged 清单包含 root AGENTS、OpenWAM EWM/memory、collector/conveyor/event兼容代码、根启动脚本、小资产、migration。
- 没有数据、权重、环境、checkpoint、视频、凭据、备份或嵌套 `.git`。
- 大于25MiB的首方文件必须解释并移到大文件传输清单；当前代码准备不依赖 Git LFS 上传大训练产物。
- `git diff --cached --check`、内容/体积扫描、第三方恢复dry-run和CPU轻量验证完成；GPU验证按第9节另做。
- 若目标URL或认证缺失，完成本地可审阅准备后明确报告尚未push，不能声称已上传。

下一位 agent 从这里接手时：先读本节和 migration 清单，确认新账号路径/目标仓库；取得代码快照和另传数据/权重；恢复固定源码并重建三个环境；修正路径；逐项验收。最终报告必须包含“已完成 / 未完成 / 下一步建议”，给出真实检查数值和阻塞，区分历史源环境验证与新环境验证。
