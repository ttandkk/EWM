# EWM v1 实施方案与运行约定

架构以 `ewm_v1_architecture_spec.md` 为准。EWM 专属模型逻辑集中在
`OpenWAM/openwam/model/architectures/ewm.py`，复用 OpenWAM 组件；不继承绑定
Wan/video latent 语义的 DualSystemMoTDriver。每完成一个模块，先补齐并通过对应测试，再推进下一模块。

## 文件与复用边界

- `openwam/model/architectures/ewm.py`：3D event codec、布局转换、共享 RoPE、双专家 joint block、冻结 context、flow loss、采样、精简 checkpoint 接口。
- `configs/model/ewm.yaml`：D=512、8 heads、8 joint layers，codec base_dim=32。
- `configs/train_ewm.yaml`：通过既有 EWMDataset 配置使用 K=8。
- `scripts/train_ewm.py`：Hydra/Accelerate 训练，复用 dataset registry、seeding、cosine scheduler、CSV/W&B 工具；既有 `scripts/train.py` 自动分派 EWM。
- `scripts/eval_ewm_policy.py`：独立 policy 推理，支持离线 dataset 采样和外部环境工厂。
- `tests/test_ewm_model.py` 及已有 codec/attention/dataset 测试：模块门禁、集成测试和小规模诊断实验。

直接复用 `DinoV3VideoEncoder`、`T5SmallTextEncoder`、`RMSNorm`、
`TimestepEmbedding`、`TimestepModulation`、`BridgeCrossAttention`、
`CosmosFlowSchedulerAdapter`、`EWMDataset` 和 `EWMNormalizer`。

## 不变的模型语义

1. p-major 8 通道恢复成负/正极性 × 四个时间 bin，每个 interval/camera 独立经过 3D codec。
2. Event 为 `[V_k clean; noisy V_{k+1:k+K}]`；Action 为 `[s_k clean; noisy a_{k:k+K-1}]`。
3. 两条流时间坐标都是 `0..K`；`a_k` 与 `V_{k+1}` 的 RoPE 时间都是 1。
4. head_dim=64 时 RoPE 使用 16/24/24 的 t/y/x 子空间；Action 的 y/x 是恒等变换，camera 只用 learned embedding。
5. 每层独立 Event/Action AdaLN/QKV/output/FFN，共享 full mixed attention 的可见性。将两条 Q 拼接，一次 SDPA 后按流拆分，与分别调用两次 SDPA 等价。
6. RGB/T5 只作为外部 cross-attention context，不进入 mixed self-attention 序列。
7. 一个样本共享一个 `t ~ Uniform[0,1]`；`x_t=(1-t)x_0+t*epsilon`，velocity target 为 `epsilon-x_0`。
8. 全局均值 `L_event + L_action`，权重均为 1；仅解码未来 tokens。clean prefix 内容不加噪，但 hidden representation 可以受共享 flow time 的 AdaLN 调制。
9. 推理使用 shift=1 的 Cosmos scheduler，从 t=1 向 0 联合 Euler 积分，最后才 clamp event/action 并反归一化 action。

## Review 后的工程调整

### RGB 预处理

在 DINO 参数所在 device/dtype 上批量完成 uint8→float、除以 255、ImageNet mean/std 标准化，得到 `[2B,3,1,H,W]`。使用 OpenWAM DINO 的同一组归一化常数，训练 forward 中不经过 `.cpu()`、NumPy 或 PIL。FP32/BF16 下与原 PIL 预处理逐元素一致；输出 token 数从实际 DINO feature grid 推导。冻结编码器始终保持 eval 和无梯度，projection/modality/camera 参数参与训练。

### Codec microbatch 与 decoder

生产配置以 `codec_batch_size=16` 起步，GPU 上需要实测 16/32/64 的显存和吞吐；这不是已证明最优的值。CPU 小模型诊断显式使用 2。不同 microbatch 划分的输出/梯度有一致性测试。

保留当前 `ConvTranspose3d(kernel_size=stride)` 上采样后接残差 Conv3D refinement 的实现。它符合冻结的时空输出几何，非因果、无缓存、输出无界 velocity。原 spec 的 ResBlock→upsample 是推荐几何，不应写成已经实现了 interpolation upsample。若真实重建出现网格/时间 bin 伪影，再将 interpolation+Conv3D 作为明确的后续消融。

### 稀疏区域指标

不改变 loss。使用 clean future voxel `>0` 作为 active mask，记录 detached velocity residual 的 `event_active_mse`、`event_inactive_mse` 与 `event_active_fraction`。空区域的 MSE 记 0，必须结合 occupancy 解读。训练输出 `event_metrics.csv`，并在日志和 W&B 中同步记录；跨 microbatch/rank 取这些 batch 指标的平均值，不冒充按全数据集区域像素数加权的指标。重建诊断另报告 voxel reconstruction MSE 和全零预测基线。

### Checkpoint 约定

新 checkpoint 标记 `format=ewm_trainable_v1`，保存 EWM 权重/buffers、context projection 与 embedding、模型配置、step 和 horizon，排除 `context_encoder.dino.*`、`context_encoder.text.*`。冻结编码器保留在外部 pretrained 目录，恢复依赖相同权重和 tokenizer。加载时严格检查所有 EWM key，只允许预期的冻结编码器键缺失。

兼容读取此前不带 format 的完整 state_dict checkpoint；新 checkpoint 不再重复保存冻结权重。两种格式都不是 optimizer/RNG 完整训练恢复点。

## 本地权重与运行

已下载并离线验证：

- `pretrained/dinov3-vitl16`：用户指定 `camenduru/dinov3-vitl16-pretrain-lvd1689m`，303,129,600 参数，通常称约 300M。
- `pretrained/t5-small`：`google-t5/t5-small`，完整约 60M，EWM 仅加载其中 35,330,816 参数的 encoder。

每个目录的 `download_source.json` 记录仓库和 revision。
从 `OpenWAM` 启动：

```bash
EWM_DINO_PATH=/projects/hdd/myproject/CVPR2027/pretrained/dinov3-vitl16 \
EWM_T5_PATH=/projects/hdd/myproject/CVPR2027/pretrained/t5-small \
../envs/openwam/bin/python scripts/train.py --config-name train_ewm
```

CPU 小规模运行需要 `training.mixed_precision=no model.codec_batch_size=2` 并降低模型规模。
生产 D512/L8/K8 训练应在目标 GPU 完成验证后运行。

推理脚本从 checkpoint 和 `normalization.json` 加载模型与统计量；环境每次提供双视角 RGB、当前 causal voxel、raw proprio 和 prompt。预测后仅执行首个物理 action，再观测、重新规划。采集脚本不依赖模型，训练脚本不创建环境。环境工厂必须自行接好 RGB/event sensors；generic policy loop 不等同于已完成 Isaac Sim 接线或闭环成功率评测。

## 验证顺序与完成边界

1. codec 几何/轴分离/梯度，mixed attention 双向可见性与合并 SDPA 等价性。
2. RoPE 的 t/y/x 相位与 action/event transition 对齐。
3. tensor RGB 预处理等价性、冻结状态、真实 DINO/T5 context 编码。
4. oracle velocity 从 epsilon 出发，经完整 scheduler 在 clamp 前恢复 x0，覆盖 1/7/20 步。
5. synthetic moving-edge 和真实双视角四-bin event 重建，检查 active/inactive 指标、全零基线和图像。
6. D512/L8/K8 GPU smoke/overfit；通过后对 K=16/32 和 codec microbatch=16/32/64 测显存与吞吐。

当前 CPU 测试和 GPU 门禁的实际结果、诊断图路径及限制见 `ewm_v1_implementation_status.md`。没有 GPU 时，GPU tests 必须明确 skip，不能把小模型 CPU 测试写成生产训练通过。
