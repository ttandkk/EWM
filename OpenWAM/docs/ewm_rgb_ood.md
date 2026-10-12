# Piper RGB-only OOD 评测

Piper 仿真评测支持可选的光照 OOD 和运动模糊 OOD。默认均关闭，原有评测命令和 event-guided replan 默认行为不变。

## 一键启动

在项目目录执行以下命令，会自动提交 Slurm 任务。默认仍是 50 trials、500 steps、固定 execution horizon 16；这里只额外启用双路 RGB OOD。

运行前需由使用者确认接受 Isaac Sim 要求的 NVIDIA Omniverse EULA。现有启动脚本包含 `OMNI_KIT_ACCEPT_EULA=YES`，不要在尚未接受许可时运行该脚本。

```bash
bash OpenWAM/scripts/eval_ewm/run_ewm_piper_eval.sh \
  --checkpoint-dir /projects/hdd/myproject/CVPR2027/OpenWAM/outputs/ewm_lean16_p4_d768_a512_dual_ada2_80k/job179463 \
  --checkpoint-step 80000 \
  --rgb-ood
```

`--rgb-ood` 同时开启光照和运动模糊，默认强度 `moderate`。两项也可独立使用：

```bash
# 用下面任意一项替换上面的 --rgb-ood：
--rgb-lighting-ood                       # 只有光照
--rgb-motion-blur-ood                    # 只有运动模糊
--rgb-ood --rgb-ood-severity strong       # 两项强扰动
--rgb-ood --no-rgb-motion-blur-ood        # 总开关开启，但明确关闭模糊
```

组件开关优先于总开关，和参数顺序无关。`--no-rgb-ood` 关闭总开关，但不会覆盖明确的 `--rgb-lighting-ood` 或 `--rgb-motion-blur-ood`。

如需指定 GPU，可用 `sbatch --gres=gpu:rtx5090:1 OpenWAM/scripts/eval_ewm/run_ewm_piper_eval.sh ...`。申请硬件前先检查 `sinfo -o '%P %G %D %N'`。`--event-guided-replan` 可以与 RGB OOD 同时使用，OOD 开关本身不会启用 event-guided replan。

## 固定、单因素模式

event-action、action-only、RGB-action 三套 stack 测评入口均支持新增的
`--rgb-ood-mode`。不传此参数时默认 `legacy`，原有随机 OOD 开关、强度和历史结果全部保留。

| 参数值 | 唯一施加的处理 |
| --- | --- |
| `clean` | 无扰动，RGB 直接透传 |
| `blur-3` | 3 px 水平均匀线性卷积 |
| `blur-5` | 5 px 水平均匀线性卷积 |
| `exposure-0.8` | 线性 RGB 乘 0.8 |
| `exposure-0.6` | 线性 RGB 乘 0.6 |
| `exposure-0.4` | 线性 RGB 乘 0.4 |

例如把启动命令中的 `--rgb-ood` 替换为 `--rgb-ood-mode blur-3`，
或 `--rgb-ood-mode exposure-0.8`。这些选项会由现有启动脚本透传，不改变
checkpoint、trial 数、执行 horizon、视频保存或 replan 设置。

固定模式两路相机、全部 trial、全部帧使用相同参数，不受 OOD seed 或 severity
影响；不叠加偏色、空间亮度渐变、噪声或其他扰动。模糊在原始 256×256 RGB
网格上操作，水平方向，不做垂直扩散，边界使用 `reflect101`。曝光和卷积均在线性
RGB 中计算，再转回 uint8 sRGB；曝光增益不是 lux，也不是直接乘 sRGB 像素值。
这是单因素传感器退化测试，不是实际场景变暗或真实运动快门模拟。

非 `legacy` 模式不能混用 `--rgb-ood`、`--[no-]rgb-lighting-ood` 或
`--[no-]rgb-motion-blur-ood`，避免配置歧义；单独 `--no-rgb-ood` 可保留。
固定模式的配置仍记录 seed 以便审计，但不会从它抽样。summary 新增
`rgb_ood_mode`、固定曝光增益、核长度和方向；manifest 和 trial JSON
记录同样的信息，视频顶行标注模式。恢复时会拒绝不同模式混入同一测评目录。

## RGB 和 event 的隔离边界

以下采集隔离边界同样适用于固定模式。

采集顺序如下：

1. EVIS 从未修改的仿真相机 HDR/depth/motion 数据取前后快照、插帧、生成事件。
2. 按原有 `(start, end]` 时间窗体素化事件。
3. 复制 `cam0` 和 `cam_wrist` 的 RGB，然后仅在该副本上做 OOD。
4. 把扰动后的 RGB、原始 event 和原始 proprio 发送给策略。视频顶行显示同一份扰动 RGB。

不会改变场景灯光、渲染参数、原始相机缓冲区、EVIS 输入、event 的幅值/极性/时间戳/计数，或 state。双路 RGB 均受扰动，event 模式仍可为 `dual` 或 `wrist_only`。真实 event 行和预测 event 行不再做 OOD 图像处理；预测 event 是策略基于扰动输入的实际预测。

这是**传感器侧 RGB 后处理 OOD**：光照以曝光、色温和空间亮度渐变近似，不是改动 USD 灯光；运动模糊使用有方向的直线 PSF 卷积，不是高斯模糊，也不是根据实际速度积分的物理快门模拟。这样才能保证 event 生成仍使用干净源数据。

“不影响 event”指直接采集链路隔离。在相同场景和相同行为序列下，event 保持不变；闭环策略看到不同 RGB 后可能输出不同动作，从而改变之后的场景和事件，因此不能保证 clean/OOD 两次完整闭环评测的 event 序列完全一致。

## 强度与可复现性

以下描述 `legacy` 随机模式：先将 uint8 sRGB 转为线性 RGB，依次施加曝光、色温/空间渐变、方向性模糊，最后截断并转换回 `[camera, 3, 256, 256]` uint8。每个 trial、每个相机固定一组参数，下一 trial 重新采样；不会逐帧闪烁。

| `--rgb-ood-severity` | 暗曝光增益 | 亮曝光增益 | RGB 暖/冷偏移上限 | x/y 渐变系数上限 | 模糊核长度 |
| --- | --- | --- | --- | --- | --- |
| `mild` | 0.65–0.80 | 1.20–1.40 | 0.08 | 0.10 | 5 或 7 px |
| `moderate`（默认） | 0.40–0.60 | 1.65–2.00 | 0.15 | 0.25 | 9 或 11 px |
| `strong` | 0.20–0.35 | 2.50–3.00 | 0.25 | 0.40 | 15 或 19 px |

每个相机等概率选暗/亮曝光；暖冷系数 `w` 对应线性 RGB 通道增益 `[1+w, 1, 1-w]`；空间亮度乘数为 `1 + shade_x*x + shade_y*y`，其中坐标在 `[-1, 1]`。模糊方向在 `[0, 180)` 度采样，核归一化，边缘为 `reflect101`。

`--rgb-ood-seed N` 可指定独立种子，默认取 `--seed`（启动脚本默认 10000）。局部 RNG 从 OOD seed、trial index、camera index、组件编号派生，不消耗 Python/NumPy/Torch 的全局随机数；单独开关模糊不会改变光照抽样。相同配置和 trial index 可重现相同参数，watchdog 恢复时也一致。

恢复运行会检查 OOD 配置，不允许混合 clean/OOD、不同强度或不同种子。没有 OOD 字段的历史 clean manifest 仍可按 clean 设置恢复。

## 输出

默认输出仍在 checkpoint 目录的 `eval_piper_sim/job<job_id>/`：

- `manifest.json` / `complete.json`：开关、强度、种子、实现版本及作用范围。
- `trial_XXX_initial_state.json` / `trial_XXX_result.json`：上述配置和每个相机实际曝光、色温、渐变、模糊长度/方向。
- `summary.txt`：`rgb_ood_enabled`、两个组件开关、强度和种子。
- `trial_XXX_h264_pred_events_*.mp4`：原有六宫格布局，RGB 顶行显示扰动效果及 OOD 标签，event 两行保持原有显示流程。
- 开启 `--profile-timing` 时，`step_timing.rgb_ood` 统计后处理耗时；它已经包含在 `capture_total` 中，不重复计入总耗时。

实现：`scripts/eval_ewm/ewm_rgb_ood.py`，采集接入：`scripts/eval_ewm/eval_ewm_piper_sim.py`。回归测试：`tests/test_ewm_rgb_ood.py`，使用真实采集/体素化函数和受控场景替身逐步比较 clean/OOD，检查 event、快照、时间戳、计数和 state 不变；另有开关、可复现、视频和 summary 测试。

## 验证记录

新增固定模式已通过 129 项 CPU 回归测试，覆盖三套入口的参数解析及冲突检查、
线性 RGB 的独立像素参考计算、固定水平卷积、全局 RNG 隔离、真实采集函数的
event/state 隔离、模式恢复检查、summary，以及原有 action-only/RGB-action
闭环和完整视频保存流程。另在 `tmp/ewm_pure_rgb_ood/` 晋升前验证了 18 项：
旧模式 12 种开关/强度组合的图像与元数据和修改前完全一致，6 个固定模式的
像素计算通过。新增模式尚未进行 Isaac/GPU 实跑，本次没有提交测评任务。

42 项相关回归测试通过。真实 RTX Pro 6000 / Isaac 短验证任务 `180762` 也已成功完成：使用 80k checkpoint，`--rgb-ood`，1 trial / 8 steps；配置记录一致，六宫格 H.264 视频正常（512×768、9 帧、30 FPS），双路 RGB 后处理平均耗时 5.34 ms。输出在项目的 `tmp/piper_rgb_ood/results/job180762/`，详细记录见 `tmp/piper_rgb_ood/VALIDATION.md`。该短任务只验证功能链路，不用于比较成功率。
