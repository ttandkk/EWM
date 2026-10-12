**EWM / Sparse 推理与相关训练实验总表**（整理日期：2026-10-06）

覆盖本轮对话中 Sparse 首版测速、双摄像头编译排查、请求内缓存/恢复消融、单摄像头对照、瓶颈分解，最后两项优化，双事件历史4帧eager探索，以及双bin形状测速的全部有记录实验；包括未合入方案、复测和失败启动。数值由原始 JSON 提取，通常保留三位小数；“—”表示未测或不适用，不表示零。表中每个链接可追溯原始记录；同目录保留另一模型的对应结果及完整逐次数据。

共同配置为 NVIDIA L40S 48GB、PyTorch 2.7.1+cu128、BF16 autocast、Event D768 / Action D512、12层/8头、预测24帧，冻结本地预训练 LeanVAE/DINO/T5；推理 B1，训练 B32。测速采用随机 policy 权重；除明确标注的双bin占位实验外，输入为真实观测，均不构成训练收敛或任务精度结论。RGB条件始终为双摄像头；“单/双事件”仅指 event 摄像头数量。单历史帧时，双事件 Dense/Sparse 为896/608个 event tokens（joint921/633）；单事件为448/304（joint473/329）。joint计数包含action24和state1，不含cross-attention条件K/V。

“完整采样”默认包含在线观测VAE、条件编码、K/V准备、layout和全部去噪步骤，排除数据读取/H2D、首次编译/capture、最终事件解码；含解码的初测单列。配对值 `a / b` 的顺序在单位列注明，通常是4步 / 20步。不同作业的host状态、预热和测量方法不同，**只在同批次同口径内比较加速比例**；Profiler累计kernel、异步span、独立GPU微测和完整请求wall time不能混用或直接相加。

| 序号 | Slurm job | 测量口径 | Event视角 | 方案 / 指标 | 单位与顺序 | Dense EWM | Sparse EWM | 结论 / 状态 / 采样方法 | 证据 |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| 1 | 187215 | 完整采样 | 双事件 | 初测 eager | ms/请求，4步 / 20步 | 162.267 / 411.089 | 171.365 / 428.418 | 3次预热+10次计时；初测波动大，不用于可靠加速结论 | [原始记录](../tmp/ewm_sparse_token_speed/results/job187215/dense_inference.json) |
| 2 | 187215 | 完整采样 | 双事件 | 初测 eager，含最终 VAE 解码 | ms/请求，4步 | 115.488 | 119.733 | 3次预热+10次计时；初测波动大，不用于可靠加速结论 | [原始记录](../tmp/ewm_sparse_token_speed/results/job187215/dense_inference.json) |
| 3 | 187215 | 完整采样 | 双事件 | 复测 eager | ms/请求，4步 / 20步 | 100.178 / 415.601 | 103.901 / 431.475 | 10次预热+30次计时；确认 Sparse 未加速 | [原始记录](../tmp/ewm_sparse_token_speed/results/job187215/confirmed/dense_inference.json) |
| 4 | 187215 | 完整采样 | 双事件 | 复测 eager，含最终 VAE 解码 | ms/请求，4步 | 117.489 | 120.996 | 10次预热+30次计时；确认 Sparse 未加速 | [原始记录](../tmp/ewm_sparse_token_speed/results/job187215/confirmed/dense_inference.json) |
| 5 | 187215 | 训练 | 双事件 | compile default 稳态训练计算，中位数 | ms/step | 663.551 | 587.637 | B32，预热15步、统计20步；含在线编码、前后向和优化器；Sparse计算约减少11.4% | [原始记录](../tmp/ewm_sparse_token_speed/results/job187215/dense_train.json) |
| 6 | 187215 | 训练 | 双事件 | compile default 稳态训练计算，均值 | ms/step | 663.534 | 587.819 | 与中位数一致，表明计算收益稳定 | [原始记录](../tmp/ewm_sparse_token_speed/results/job187215/dense_train.json) |
| 7 | 187215 | 训练 | 双事件 | 含加载/H2D完整训练 step，中位数 | ms/step | 698.577 | 621.998 | I/O分布不同，不能据此声称完整训练稳定加速 | [原始记录](../tmp/ewm_sparse_token_speed/results/job187215/dense_train.json) |
| 8 | 187215 | 训练 | 双事件 | 含加载/H2D完整训练 step，均值 | ms/step | 739.885 | 736.135 | 约0.740 vs0.736s；I/O抵消大部分计算收益 | [原始记录](../tmp/ewm_sparse_token_speed/results/job187215/dense_train.json) |
| 9 | 187215 | 训练 | 双事件 | 峰值 allocated 显存 | GiB | 21.546 | 17.065 | B32；Sparse约减少20.8%，不是推理权重缓存占用 | [原始记录](../tmp/ewm_sparse_token_speed/results/job187215/sparse_train.json) |
| 10 | 187215 | 训练冷启动 | 双事件 | 首个完整 step，含编译 | s/step | 136.814 | 266.506 | 不能当稳态训练时间；未保存checkpoint | [原始记录](../tmp/ewm_sparse_token_speed/results/job187215/sparse_train.json) |
| 11 | 187400 | 完整采样 | 双事件 | eager | ms/请求，4步 / 20步 | 136.575 / 586.028 | 140.522 / 597.876 | 3轮×12次计时，每组8次预热；Sparse更慢；host dispatch和小kernel开销显著 | [原始记录](../tmp/ewm_sparse_inference_audit/results/job187400/summary.json) |
| 12 | 187400 | 完整采样 | 双事件 | default | ms/请求，4步 / 20步 | 64.440 / 223.101 | 66.579 / 229.904 | 3轮×12次计时，每组8次预热；融合已加速两者，但Sparse仍慢 | [原始记录](../tmp/ewm_sparse_inference_audit/results/job187400/summary.json) |
| 13 | 187400 | 完整采样 | 双事件 | reduce-overhead | ms/请求，4步 / 20步 | 51.340 / 152.361 | 49.167 / 144.823 | 3轮×12次计时，每组8次预热；Sparse耗时减少4.23% / 4.95%；CUDA Graph有效，无graph break | [原始记录](../tmp/ewm_sparse_inference_audit/results/job187400/summary.json) |
| 14 | 187400 | Profiler | 双事件 | eager DiT累计kernel时间 | ms/去噪步 | 8.805 | 8.128 | Profiler结果，非真实请求延迟 | [原始记录](../tmp/ewm_sparse_inference_audit/results/job187400/REPORT.txt) |
| 15 | 187400 | Profiler | 双事件 | eager DiT GPU执行span | ms/去噪步 | 44.720 | 44.760 | 含launch空隙和Profiler开销 | [原始记录](../tmp/ewm_sparse_inference_audit/results/job187400/REPORT.txt) |
| 16 | 187400 | Profiler | 双事件 | eager输入准备累计kernel | ms/去噪步 | 0.239 | 0.328 | Sparse重排额外成本 | [原始记录](../tmp/ewm_sparse_inference_audit/results/job187400/REPORT.txt) |
| 17 | 187400 | Profiler | 双事件 | eager恢复累计kernel | ms/去噪步 | 0.028 | 0.056 | Sparse恢复成本增加 | [原始记录](../tmp/ewm_sparse_inference_audit/results/job187400/REPORT.txt) |
| 18 | 187639 | 完整采样 | 双事件 | eager基线 | ms/请求，4步 / 20步 | 98.564 / 411.373 | 103.310 / 430.572 | 4轮×10次计时，每组6次预热；对照 | [原始记录](../tmp/ewm_inference_opt/results/summary.json) |
| 19 | 187639 | 完整采样 | 双事件 | reduce-overhead | ms/请求，4步 / 20步 | 45.073 / 142.780 | 43.878 / 134.524 | 4轮×10次计时，每组6次预热；已合入共享compile_inference接口 | [原始记录](../tmp/ewm_inference_opt/results/summary.json) |
| 20 | 187639 | 完整采样 | 双事件 | +请求内不变量缓存 | ms/请求，4步 / 20步 | 44.866 / 139.994 | 44.041 / 132.916 | 4轮×10次计时，每组6次预热；缓存观测/state投影、坐标、RoPE；Sparse20步仅约1.2%收益，未合入 | [原始记录](../tmp/ewm_inference_opt/results/summary.json) |
| 21 | 187639 | 完整采样 | 双事件 | +逆置换gather恢复 | ms/请求，4步 / 20步 | — | 44.135 / 132.961 | 4轮×10次计时，每组6次预热；仅Sparse；相对cache无额外收益，未合入 | [原始记录](../tmp/ewm_inference_opt/results/summary.json) |
| 22 | 187681 | 正式代码复核 | 双事件 | reduce-overhead | ms/请求，4步 / 20步 | 46.137 / 145.141 | 43.922 / 134.261 | 16次计时；换观测、输出保留、eager回退、state_dict检查通过；71项正式回归通过 | [原始记录](../tmp/ewm_inference_opt/results/job187681/sparse_production.json) |
| 23 | 188045 | 完整采样 | 单事件 | eager | ms/请求，4步 / 20步 | 136.564 / 574.525 | 139.997 / 586.287 | 4轮×10次计时，每组6次预热；Sparse更慢 | [原始记录](../tmp/ewm_single_event_speed/results/job188045/results.json) |
| 24 | 188045 | 完整采样 | 单事件 | graph | ms/请求，4步 / 20步 | 47.570 / 137.285 | 47.393 / 134.832 | 4轮×10次计时，每组6次预热；Sparse仅快0.37% / 1.79%；4步差异接近噪声 | [原始记录](../tmp/ewm_single_event_speed/results/job188045/results.json) |
| 25 | 188069 | 失败启动 | 单事件 | 无有效测速数据 | — | — | — | Slurm脚本指向错误路径，修正后重跑 | [原始记录](../logs/ewm_single_profile_188069.log) |
| 26 | 188076 | 失败启动 | 单事件 | 无有效测速数据 | — | — | — | 诊断脚本profile.py遮蔽标准库，改名benchmark_profile.py后重跑 | [原始记录](../logs/ewm_single_profile_188076.log) |
| 27 | 188087 | 分段诊断 | 单事件 | 带分段计时的完整请求 | ms/请求，4步 / 20步 | 47.087 / 136.402 | 55.241 / 146.595 | 顺序测量受host状态影响，不能替代配对测速 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/profile.json) |
| 28 | 188087 | 分段诊断 | 单事件 | 条件编码及layout | ms/调用，4步请求 / 20步请求 | 10.078 / 10.779 | 13.529 / 14.022 | 异步CUDA event span含CPU供给空隙；每个请求调用一次 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/profile.json) |
| 29 | 188087 | 分段诊断 | 单事件 | 条件K/V准备 | ms/调用，4步请求 / 20步请求 | 2.585 / 2.696 | 3.313 / 3.377 | 异步CUDA event span含CPU供给空隙；每个请求调用一次 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/profile.json) |
| 30 | 188087 | 分段诊断 | 单事件 | 观测VAE编码 | ms/调用，4步请求 / 20步请求 | 1.894 / 2.023 | 2.491 / 2.548 | 异步CUDA event span含CPU供给空隙；每个请求调用一次 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/profile.json) |
| 31 | 188087 | 分段诊断 | 单事件 | 编译velocity核心span | ms/调用，4步请求 / 20步请求 | 5.441 / 5.430 | 5.846 / 5.654 | 异步CUDA event span含CPU供给空隙；每个去噪步调用一次 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/profile.json) |
| 32 | 188087 | 静态GPU微测 | 单事件 | 完整velocity核心 | ms/去噪步 | 4.639 | 4.480 | 固定输入、独立compile/capture；各段不可相加。完整核心仅快约3.4% | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/profile.json) |
| 33 | 188087 | 静态GPU微测 | 单事件 | 准备/压缩 | ms/去噪步 | 0.054 | 0.095 | 固定输入、独立compile/capture；各段不可相加。 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/profile.json) |
| 34 | 188087 | 静态GPU微测 | 单事件 | 12层DiT | ms/去噪步 | 4.446 | 4.239 | 固定输入、独立compile/capture；各段不可相加。DiT仅快约4.6% | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/profile.json) |
| 35 | 188087 | 静态GPU微测 | 单事件 | 输出头/恢复 | ms/去噪步 | 0.018 | 0.033 | 固定输入、独立compile/capture；各段不可相加。 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/profile.json) |
| 36 | 188087 | 固定输入入口 | 单事件 | 普通compiled callable入口 | ms/调用 | 4.907 | 4.747 | 包含入口处理，非完整采样；5轮中位数 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/profile.json) |
| 37 | 188087 | 首层独立微测 | 单事件 | 时间MLP/modulation | ms/组件调用 | 0.042 | 0.042 | 独立capture，权重缓存/融合边界不同；不可求和或乘12解释整网占比 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/profile.json) |
| 38 | 188087 | 首层独立微测 | 单事件 | event prepare | ms/组件调用 | 0.044 | 0.042 | 独立capture，权重缓存/融合边界不同；不可求和或乘12解释整网占比 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/profile.json) |
| 39 | 188087 | 首层独立微测 | 单事件 | action prepare | ms/组件调用 | 0.031 | 0.030 | 独立capture，权重缓存/融合边界不同；不可求和或乘12解释整网占比 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/profile.json) |
| 40 | 188087 | 首层独立微测 | 单事件 | mixed attention | ms/组件调用 | 0.026 | 0.022 | 独立capture，权重缓存/融合边界不同；不可求和或乘12解释整网占比 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/profile.json) |
| 41 | 188087 | 首层独立微测 | 单事件 | event finish | ms/组件调用 | 0.103 | 0.093 | 独立capture，权重缓存/融合边界不同；不可求和或乘12解释整网占比 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/profile.json) |
| 42 | 188087 | 首层独立微测 | 单事件 | action finish | ms/组件调用 | 0.062 | 0.062 | 独立capture，权重缓存/融合边界不同；不可求和或乘12解释整网占比 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/profile.json) |
| 43 | 188087 | 首层独立微测 | 单事件 | event FFN | ms/组件调用 | 0.054 | 0.039 | 独立capture，权重缓存/融合边界不同；不可求和或乘12解释整网占比 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/profile.json) |
| 44 | 188087 | 首层独立微测 | 单事件 | action FFN | ms/组件调用 | 0.016 | 0.016 | 独立capture，权重缓存/融合边界不同；不可求和或乘12解释整网占比 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/profile.json) |
| 45 | 188087 | Profiler累计 | 单事件 | 纯转换/拷贝kernel | ms/去噪步 | 1.753 | 1.748 | 仅velocity范围；纯转换占累计kernel约37–39%，其中确认包含大矩阵权重转换 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/kernel_summary.json) |
| 46 | 188087 | Profiler累计 | 单事件 | GEMM/GEMV | ms/去噪步 | 1.267 | 1.147 | 仅velocity范围；纯转换占累计kernel约37–39%，其中确认包含大矩阵权重转换 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/kernel_summary.json) |
| 47 | 188087 | Profiler累计 | 单事件 | Attention | ms/去噪步 | 0.835 | 0.798 | 仅velocity范围；纯转换占累计kernel约37–39%，其中确认包含大矩阵权重转换 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/kernel_summary.json) |
| 48 | 188087 | Profiler累计 | 单事件 | 其他kernel | ms/去噪步 | 0.841 | 0.839 | 仅velocity范围；纯转换占累计kernel约37–39%，其中确认包含大矩阵权重转换 | [原始记录](../tmp/ewm_single_bottleneck/results/job188087/kernel_summary.json) |
| 49 | 188094 | CPU开销诊断 | 单事件 | 带包装的完整请求 | ms/请求，4步 / 20步 | 48.467 / 136.827 | 49.918 / 137.091 | 16请求/配置；诊断结果，不替代官方配对速度 | [原始记录](../tmp/ewm_single_bottleneck/results/job188094/sampler_overhead.json) |
| 50 | 188094 | CPU开销诊断 | 单事件 | 重复eval/train遍历 | CPU ms/请求，4步 / 20步 | 11.485 / 11.709 | 11.696 / 12.283 | 2次/请求，与token数无关 | [原始记录](../tmp/ewm_single_bottleneck/results/job188094/sampler_overhead.json) |
| 51 | 188094 | CPU开销诊断 | 单事件 | Euler调度器更新 | CPU ms/请求，4步 / 20步 | 0.418 / 2.126 | 0.450 / 2.218 | 4步8次、20步40次，非首要瓶颈 | [原始记录](../tmp/ewm_single_bottleneck/results/job188094/sampler_overhead.json) |
| 52 | 188119 | 完整采样 | 单事件 | 原编译基线 | ms/请求，4步 / 20步 | 47.270 / 135.650 | 47.137 / 133.647 | 4轮×10次计时，每组4次预热；该批次对照，禁止用跨批次绝对值计算收益 | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188119/results.json) |
| 53 | 188119 | 完整采样 | 单事件 | 仅跳过模式切换 | ms/请求，4步 / 20步 | 36.919 / 125.215 | 36.584 / 122.775 | 4轮×10次计时，每组4次预热；已合入；输出与原路径完全一致 | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188119/results.json) |
| 54 | 188119 | 完整采样 | 单事件 | 跳过模式切换 + Inductor freezing | ms/请求，4步 / 20步 | 51.881 / 195.760 | 51.583 / 193.993 | 4轮×10次计时，每组4次预热；转换未消除且拷贝增多，明显变慢；未合入 | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188119/results.json) |
| 55 | 188142 | 完整采样 | 单事件 | 原编译基线 | ms/请求，4步 / 20步 | 50.416 / 138.546 | 50.547 / 137.643 | 3轮×10次计时，每组4次预热；该批次对照，禁止用跨批次绝对值计算收益 | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188142/results.json) |
| 56 | 188142 | 完整采样 | 单事件 | 仅跳过模式切换 | ms/请求，4步 / 20步 | 39.688 / 127.152 | 39.633 / 126.263 | 3轮×10次计时，每组4次预热；已合入；输出与原路径完全一致 | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188142/results.json) |
| 57 | 188142 | 完整采样 | 单事件 | 跳过模式切换 + 普通BF16权重缓存 | ms/请求，4步 / 20步 | 49.253 / 162.132 | 48.493 / 161.903 | 3轮×10次计时，每组4次预热；转换消除，但缓存被当普通输入逐步拷贝；未合入 | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188142/results.json) |
| 58 | 188155 | 完整采样 | 单事件 | 原编译基线 | ms/请求，4步 / 20步 | 50.715 / 138.337 | 50.170 / 135.929 | 3轮×10次计时，每组4次预热；该批次对照，禁止用跨批次绝对值计算收益 | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188155/results.json) |
| 59 | 188155 | 完整采样 | 单事件 | 仅跳过模式切换 | ms/请求，4步 / 20步 | 38.578 / 126.835 | 39.016 / 124.651 | 3轮×10次计时，每组4次预热；已合入；输出与原路径完全一致 | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188155/results.json) |
| 60 | 188155 | 完整采样 | 单事件 | 跳过模式切换 + 固定地址BF16权重缓存 | ms/请求，4步 / 20步 | 35.876 / 109.336 | 35.566 / 106.771 | 3轮×10次计时，每组4次预热；已合入；相对本轮基线Dense减少29.26% / 20.96%，Sparse减少29.11% / 21.45% | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188155/results.json) |
| 61 | 188119 | Profiler累计 | 单事件 | 仅模式优化：纯转换kernel | ms/4步请求 | 7.051 | 6.981 | 包含整请求固定编码器；非请求wall time；每个trace均4次graph replay | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188119/dense_mode_only_trace.json) |
| 62 | 188119 | Profiler计数 | 单事件 | 仅模式优化：cudaMemcpyAsync | 次/4步请求 | 227 | 247 | 188119=freezing；188142=普通缓存；188155=固定地址缓存 | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188119/dense_mode_only_trace.json) |
| 63 | 188119 | Profiler累计 | 单事件 | 本轮权重优化：纯转换kernel | ms/4步请求 | 7.065 | 6.944 | 包含整请求固定编码器；非请求wall time；每个trace均4次graph replay | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188119/dense_both_trace.json) |
| 64 | 188119 | Profiler计数 | 单事件 | 本轮权重优化：cudaMemcpyAsync | 次/4步请求 | 2511 | 2559 | 188119=freezing；188142=普通缓存；188155=固定地址缓存 | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188119/dense_both_trace.json) |
| 65 | 188142 | Profiler累计 | 单事件 | 仅模式优化：纯转换kernel | ms/4步请求 | 7.020 | 6.973 | 包含整请求固定编码器；非请求wall time；每个trace均4次graph replay | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188142/dense_mode_only_trace.json) |
| 66 | 188142 | Profiler计数 | 单事件 | 仅模式优化：cudaMemcpyAsync | 次/4步请求 | 227 | 247 | 188119=freezing；188142=普通缓存；188155=固定地址缓存 | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188142/dense_mode_only_trace.json) |
| 67 | 188142 | Profiler累计 | 单事件 | 本轮权重优化：纯转换kernel | ms/4步请求 | 0.017 | 0.011 | 包含整请求固定编码器；非请求wall time；每个trace均4次graph replay | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188142/dense_both_trace.json) |
| 68 | 188142 | Profiler计数 | 单事件 | 本轮权重优化：cudaMemcpyAsync | 次/4步请求 | 2003 | 2031 | 188119=freezing；188142=普通缓存；188155=固定地址缓存 | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188142/dense_both_trace.json) |
| 69 | 188155 | Profiler累计 | 单事件 | 仅模式优化：纯转换kernel | ms/4步请求 | 7.026 | 6.952 | 包含整请求固定编码器；非请求wall time；每个trace均4次graph replay | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188155/dense_mode_only_trace.json) |
| 70 | 188155 | Profiler计数 | 单事件 | 仅模式优化：cudaMemcpyAsync | 次/4步请求 | 227 | 247 | 188119=freezing；188142=普通缓存；188155=固定地址缓存 | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188155/dense_mode_only_trace.json) |
| 71 | 188155 | Profiler累计 | 单事件 | 本轮权重优化：纯转换kernel | ms/4步请求 | 0.015 | 0.010 | 包含整请求固定编码器；非请求wall time；每个trace均4次graph replay | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188155/dense_both_trace.json) |
| 72 | 188155 | Profiler计数 | 单事件 | 本轮权重优化：cudaMemcpyAsync | 次/4步请求 | 227 | 247 | 188119=freezing；188142=普通缓存；188155=固定地址缓存 | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188155/dense_both_trace.json) |
| 73 | 188177 | 正式代码复核 | 单事件 | 两项优化后，4步请求 | ms/请求 | 34.009 | 34.374 | 10次计时，含输出clone；用于功能/速度复核，主速度结论使用188155配对实验 | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188177/validation.json) |
| 74 | 188177 | 缓存显存 | 单事件 | BF16 Linear权重副本 | MiB | 503.051 | 504.553 | 额外约0.5GiB；不修改FP32训练参数，不进入policy checkpoint | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188177/validation.json) |
| 75 | 188177 | 数值验证 | 单事件 | 修改checkpoint后的缓存输出 vs eager | latent最大绝对差 | 0.016 | 0.015 | 通过BF16容差rtol=0.015、atol=0.02；非bitwise一致 | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188177/validation.json) |
| 76 | 188177 | 数值验证 | 单事件 | 恢复checkpoint后的输出 vs原输出 | 最大绝对差 | 0.000 | 0.000 | 两者均完全恢复；修改权重时输出均显著改变，未用旧权重 | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188177/validation.json) |
| 77 | 188177 | 生命周期验证 | 单事件 | load后旧缓存释放 | 是否通过 | 是 | 是 | 训练、加载、设备/dtype迁移均失效缓存；server在切checkpoint后重建 | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188177/validation.json) |
| 78 | 188177 | 回归测试 | 单/双事件 | 正式代码测试集 | 通过数 | — | — | 合计96项通过：推理编译/模式恢复/缓存生命周期、Sparse、EWM、action-only；GPU作业exit 0 | [原始记录](../tmp/ewm_inference_fixed_cost/results/job188177/tests.txt) |

| 79 | 188245 | 序列检查 | 双事件 | 历史1帧：真实进入DiT的token数 | event / joint | 896 / 921 | 608 / 633 | 历史4帧含当前及此前3帧，独立VAE编码；历史不做Sparse压缩，未来仍6个latent片 | [原始记录](../tmp/ewm_history4/results/job188245/results.json) |
| 80 | 188245 | 完整采样 | 双事件 | 历史1帧 eager | ms/请求，4步 / 20步 | 129.172 / 574.624 | 135.060 / 587.638 | 3轮×10次，每组预热4次；无compile及BF16权重缓存；轮次有漂移，未见Sparse稳定加速 | [原始记录](../tmp/ewm_history4/results/job188245/results.json) |
| 81 | 188245 | eager组件wall | 双事件 | 历史1帧：固定输入去噪核心 | ms/调用 | 28.558 | 29.121 | 3轮×20次；core每去噪步调用，VAE每请求调用；非独立纯GPU时间 | [原始记录](../tmp/ewm_history4/results/job188245/results.json) |
| 82 | 188245 | eager组件wall | 双事件 | 历史1帧：独立历史VAE编码 | ms/调用 | 1.801 | 1.798 | 3轮×20次；core每去噪步调用，VAE每请求调用；非独立纯GPU时间 | [原始记录](../tmp/ewm_history4/results/job188245/results.json) |
| 83 | 188254 | eager Profiler | 双事件 | 历史1帧：累计全部kernel | GPU ms/去噪步 | 9.010 | 8.472 | 5次预热、3次核心调用；GPU累计非wall time；历史4帧Sparse GPU累计少约9.5%但未转化为请求收益 | [原始记录](../tmp/ewm_history4/results/job188254/kernel_summary.json) |
| 84 | 188254 | eager Profiler | 双事件 | 历史1帧：累计attention kernel | GPU ms/去噪步 | 0.938 | 0.833 | 5次预热、3次核心调用；GPU累计非wall time；历史4帧Sparse GPU累计少约9.5%但未转化为请求收益 | [原始记录](../tmp/ewm_history4/results/job188254/kernel_summary.json) |
| 85 | 188254 | eager Profiler | 双事件 | 历史1帧：kernel数 | 次/去噪步 | 3031.000 | 3088.000 | 5次预热、3次核心调用；GPU累计非wall time；历史4帧Sparse GPU累计少约9.5%但未转化为请求收益 | [原始记录](../tmp/ewm_history4/results/job188254/kernel_summary.json) |
| 86 | 188245 | 序列检查 | 双事件 | 历史4帧：真实进入DiT的token数 | event / joint | 1280 / 1305 | 992 / 1017 | 历史4帧含当前及此前3帧，独立VAE编码；历史不做Sparse压缩，未来仍6个latent片 | [原始记录](../tmp/ewm_history4/results/job188245/results.json) |
| 87 | 188245 | 完整采样 | 双事件 | 历史4帧 eager | ms/请求，4步 / 20步 | 132.716 / 582.279 | 133.236 / 588.572 | 3轮×10次，每组预热4次；无compile及BF16权重缓存；轮次有漂移，未见Sparse稳定加速 | [原始记录](../tmp/ewm_history4/results/job188245/results.json) |
| 88 | 188245 | eager组件wall | 双事件 | 历史4帧：固定输入去噪核心 | ms/调用 | 28.524 | 29.261 | 3轮×20次；core每去噪步调用，VAE每请求调用；非独立纯GPU时间 | [原始记录](../tmp/ewm_history4/results/job188245/results.json) |
| 89 | 188245 | eager组件wall | 双事件 | 历史4帧：独立历史VAE编码 | ms/调用 | 4.076 | 4.080 | 3轮×20次；core每去噪步调用，VAE每请求调用；非独立纯GPU时间 | [原始记录](../tmp/ewm_history4/results/job188245/results.json) |
| 90 | 188254 | eager Profiler | 双事件 | 历史4帧：累计全部kernel | GPU ms/去噪步 | 10.425 | 9.438 | 5次预热、3次核心调用；GPU累计非wall time；历史4帧Sparse GPU累计少约9.5%但未转化为请求收益 | [原始记录](../tmp/ewm_history4/results/job188254/kernel_summary.json) |
| 91 | 188254 | eager Profiler | 双事件 | 历史4帧：累计attention kernel | GPU ms/去噪步 | 1.549 | 0.959 | 5次预热、3次核心调用；GPU累计非wall time；历史4帧Sparse GPU累计少约9.5%但未转化为请求收益 | [原始记录](../tmp/ewm_history4/results/job188254/kernel_summary.json) |
| 92 | 188254 | eager Profiler | 双事件 | 历史4帧：kernel数 | 次/去噪步 | 3031.000 | 3088.000 | 5次预热、3次核心调用；GPU累计非wall time；历史4帧Sparse GPU累计少约9.5%但未转化为请求收益 | [原始记录](../tmp/ewm_history4/results/job188254/kernel_summary.json) |
| 93 | 188245 | 功能验证 | 双事件 | 历史输入与原路径一致性 | 结果 | 通过 | 通过 | 历史1帧逐元素匹配正式核心；改动较早3帧影响输出；输入不含未来帧；仅tmp实验，未改正式架构 | [原始记录](../tmp/ewm_history4/results/job188245/results.json) |
| 94 | 188286 | 序列检查 | 双事件 | 历史4帧，bin=1：DiT tokens | event / joint | 1280 / 1305 | 992 / 1017 | bin2为单bin重复占位的shape-only测试，非真实分bin；不用于精度结论；actions仍24 | [原始记录](../tmp/ewm_bins2/results/job188286/results.json) |
| 95 | 188286 | eager计算量测速 | 双事件 | 历史4帧，bin=1 | ms/请求，4步 / 20步 | 127.831 / 569.590 | 131.797 / 581.790 | 3轮×10次；不开compile/权重缓存；bin2为单bin重复占位的shape-only测试，非真实分bin；不用于精度结论 | [原始记录](../tmp/ewm_bins2/results/job188286/results.json) |
| 96 | 188286 | eager组件wall | 双事件 | 历史4帧，bin=1：固定输入core | ms/调用 | 26.799 | 27.506 | 3轮×20次；bin2为单bin重复占位的shape-only测试，非真实分bin；不用于精度结论 | [原始记录](../tmp/ewm_bins2/results/job188286/results.json) |
| 97 | 188286 | eager组件wall | 双事件 | 历史4帧，bin=1：历史VAE | ms/调用 | 4.021 | 4.028 | 3轮×20次；bin2为单bin重复占位的shape-only测试，非真实分bin；不用于精度结论 | [原始记录](../tmp/ewm_bins2/results/job188286/results.json) |
| 98 | 188286 | eager Profiler | 双事件 | 历史4帧，bin=1：累计kernel | GPU ms/去噪步 | 10.512 | 9.531 | 5次预热，3次core调用；累计GPU时间不是wall time；bin2为单bin重复占位的shape-only测试，非真实分bin；不用于精度结论 | [原始记录](../tmp/ewm_bins2/results/job188286/kernel_summary.json) |
| 99 | 188286 | eager Profiler | 双事件 | 历史4帧，bin=1：attention累计 | GPU ms/去噪步 | 1.562 | 0.962 | 5次预热，3次core调用；累计GPU时间不是wall time；bin2为单bin重复占位的shape-only测试，非真实分bin；不用于精度结论 | [原始记录](../tmp/ewm_bins2/results/job188286/kernel_summary.json) |
| 100 | 188286 | eager Profiler | 双事件 | 历史4帧，bin=1：kernel数 | 次/去噪步 | 3032.000 | 3091.000 | 5次预热，3次core调用；累计GPU时间不是wall time；bin2为单bin重复占位的shape-only测试，非真实分bin；不用于精度结论 | [原始记录](../tmp/ewm_bins2/results/job188286/kernel_summary.json) |
| 101 | 188286 | 序列检查 | 双事件 | 历史4帧，bin=2：DiT tokens | event / joint | 2560 / 2585 | 1984 / 2009 | bin2为单bin重复占位的shape-only测试，非真实分bin；不用于精度结论；actions仍24 | [原始记录](../tmp/ewm_bins2/results/job188286/results.json) |
| 102 | 188286 | eager计算量测速 | 双事件 | 历史4帧，bin=2 | ms/请求，4步 / 20步 | 132.919 / 562.566 | 134.117 / 589.193 | 3轮×10次；不开compile/权重缓存；bin2为单bin重复占位的shape-only测试，非真实分bin；不用于精度结论 | [原始记录](../tmp/ewm_bins2/results/job188286/results.json) |
| 103 | 188286 | eager组件wall | 双事件 | 历史4帧，bin=2：固定输入core | ms/调用 | 26.518 | 27.166 | 3轮×20次；bin2为单bin重复占位的shape-only测试，非真实分bin；不用于精度结论 | [原始记录](../tmp/ewm_bins2/results/job188286/results.json) |
| 104 | 188286 | eager组件wall | 双事件 | 历史4帧，bin=2：历史VAE | ms/调用 | 8.129 | 8.124 | 3轮×20次；bin2为单bin重复占位的shape-only测试，非真实分bin；不用于精度结论 | [原始记录](../tmp/ewm_bins2/results/job188286/results.json) |
| 105 | 188286 | eager Profiler | 双事件 | 历史4帧，bin=2：累计kernel | GPU ms/去噪步 | 15.101 | 12.790 | 5次预热，3次core调用；累计GPU时间不是wall time；bin2为单bin重复占位的shape-only测试，非真实分bin；不用于精度结论 | [原始记录](../tmp/ewm_bins2/results/job188286/kernel_summary.json) |
| 106 | 188286 | eager Profiler | 双事件 | 历史4帧，bin=2：attention累计 | GPU ms/去噪步 | 3.005 | 1.789 | 5次预热，3次core调用；累计GPU时间不是wall time；bin2为单bin重复占位的shape-only测试，非真实分bin；不用于精度结论 | [原始记录](../tmp/ewm_bins2/results/job188286/kernel_summary.json) |
| 107 | 188286 | eager Profiler | 双事件 | 历史4帧，bin=2：kernel数 | 次/去噪步 | 3020.000 | 3079.000 | 5次预热，3次core调用；累计GPU时间不是wall time；bin2为单bin重复占位的shape-only测试，非真实分bin；不用于精度结论 | [原始记录](../tmp/ewm_bins2/results/job188286/kernel_summary.json) |
| 108 | 188286 | 功能验证 | 双事件 | bin1基线与bin2输出形状 | 结果 | 通过 | 通过 | bin1与历史4帧参考输出完全一致；bin2未来12latent片、action24步；原始数据没有真实双bin，未改正式模型 | [原始记录](../tmp/ewm_bins2/results/job188286/results.json) |

当前采用的两项优化是：已处于eval时跳过重复模式遍历，以及固定地址BF16 Linear权重缓存。主配对实验188155中，4步推理耗时减少约29%，20步减少约21%；两者均受益，优化后Sparse相对Dense仍只有约0.9% / 2.3%的小幅优势。请求内坐标/RoPE缓存、逆gather恢复、Inductor freezing以及未固定地址的权重缓存均未合入。双事件训练计算/显存收益已测；单事件训练收益及这两项推理优化后的完整训练速度未测，不作外推。

正式实现见[采样及编译入口](../OpenWAM/openwam/model/architectures/ewm.py)、[权重缓存](../OpenWAM/openwam/model/architectures/ewm_inference.py)、[推理服务器](../OpenWAM/scripts/eval_ewm/ewm_eval_server.py)。服务器启用 `--compile-core` 后默认使用权重缓存，`--no-cache-bf16-weights` 可关闭；直接调用使用 `model.eval(); model.compile_inference(cache_bf16_weights=True)`，在BF16 autocast下采样并预热实际去噪步数。缓存额外占约0.5GiB；切回训练、加载state dict、迁移设备/dtype后会清理缓存，直接原地修改参数后也必须重新编译。checkpoint切换的冷启动成本不计入表内稳态推理时间。

历史4帧eager探索（188245/188254）未启用编译或权重缓存，4帧含当前和此前3帧，未来仍压缩为6个时间片。实测Sparse减少了约9.5%的GPU累计kernel时间，但没有稳定降低完整请求延迟；只增加未压缩历史token尚不足以改变eager的主要开销。轮次漂移大于约1%的模型差异，详细配置与范围见[历史4帧报告](../tmp/ewm_history4/results/REPORT.txt)。

双bin实验188286是经确认的计算量测试：保持4个历史窗口和24个未来窗口，bin在时间轴展开，action不加倍；输入由原单bin图重复构造，不能还原或替代真实事件时间信息。完整结果见[双bin计算量报告](../tmp/ewm_bins2/results/REPORT.txt)。

## 原版 EWM patch4 / patch8 配对测速（188313）

2026-10-06，单张 NVIDIA L40S，B1、BF16、双 event / 双 RGB、当前1帧、单bin、未来24帧，Event D768 / Action D512、L12/H8。未使用 Sparse 结构。VAE 空间stride8、时间stride4不变；patch4/8对应总空间stride32/64。event tokens 896→224，含24 action与1 state的联合序列921→249。

同一进程交替测两种patch与两种执行模式，4轮，每组每轮6次预热、10次计时；下表为40次请求的中位数。包含条件编码、当前VAE编码和采样，排除数据读取/H2D、最终event解码及首次编译。graph为reduce-overhead/fullgraph编译，双方均开启固定地址BF16权重缓存。所有模型已eval，均沿用正式代码跳过重复模式遍历的优化。

| 模式 | 去噪步数 | patch4（ms/请求） | patch8（ms/请求） | patch8耗时下降 |
| --- | --- | --- | --- | --- |
| eager | 4 | 126.363 | 126.966 | -0.48% |
| eager | 20 | 570.790 | 564.214 | 1.15% |
| graph | 4 | 37.327 | 32.907 | 11.84% |
| graph | 20 | 121.909 | 101.030 | 17.13% |

使用随机策略权重及本地预训练冻结编码器；两模型相同形状参数一致，patch输入投影及输出头形状不同。只验证计算速度，未验证预测/动作质量，未改变正式配置。实际DiT token数和未来latent形状检查通过，两组编译相对各自eager均通过rtol=0.015/atol=0.02数值检查，4步采样均观测到4次CUDA Graph replay。

[原始结果](../tmp/ewm_patch8_speed/results/job188313/results.json) · [运行日志](../logs/ewm_patch8_188313.log) · [测速脚本](../tmp/ewm_patch8_speed/benchmark.py)

本轮结论：eager的4步/20步差异为-0.48%/+1.15%，轮次间方向不一致且漂移明显，不认定稳定加速。graph下4步/20步耗时分别减少11.84%/17.13%，四轮均更快。收益来自更强的均匀patch压缩；未证明event自适应分配的价值，也未证明patch8质量无损。



## 已完成

已核对历史测评和测速记录，整理如下。

**表 1：Sparse / patch8 的关键效果**

均为静态 Stack、50 trials、4 步采样、执行周期 16。Token 数为双 event 摄像头进入 DiT 的 event tokens，不含 24 action＋1 state。

| 模型 | 训练步数 | Event tokens | 成功次数 | 成功率 | 同步数原版 EWM | 差值 | 测评 Job |
|---|---:|---:|---:|---:|---:|---:|---|
| 原版 EWM patch4 | 80k | 896 | 31/50 | **62%** | — | — | 180658 |
| Sparse40 | 50k | 608 | 23/50 | **46%** | 36% | +10 个百分点 | 190078 |
| Sparse40 | 70k | 608 | 28/50 | **56%** | 60% | −4 个百分点 | 190622 |
| Sparse40 | 80k | 608 | 24/50 | **48%** | 62% | −14 个百分点 | 190817 |
| 原版 EWM patch8 | 80k | 224 | 17/50 | **34%** | 62% | −28 个百分点 | 190280 |

Sparse40 的计数是 **2×(64＋6×40)＝608**，当前观测仍保留每摄像头 64 tokens；相较 patch4 减少 **32.1%**。patch8 为 **2×7×16＝224**，减少 **75%**，所以两者不是相同压缩预算。

此外，Sparse 的配对训练计算测速为 **663.6→587.6 ms/step，减少 11.4%**；峰值 allocated 显存 **21.55→17.07 GiB，减少 20.8%**。包含数据加载的平均 step 为 **739.9→736.1 ms**，没有明显整体省时。

来源：[Sparse 测评目录](/projects/hdd/myproject/CVPR2027/OpenWAM/outputs/ewm_sparse40_d768_a512_dual_l40x2_80k/job188825/eval_piper_sim)、[patch8 结果](/projects/hdd/myproject/CVPR2027/OpenWAM/outputs/stack_event_action_p8_compile_20261006_094241/eval_piper_sim/job190280/summary.txt)。

**表 2：各种模型的推理延迟**

单位均为 **ms/请求**。独立测速为 B1、BF16，不含最终画面解码；仿真实测包含仿真运行期间的调用开销，且部分开启预测画面解码，**不能与独立测速直接比较**。

| 测量口径 / Job | 模型 | Event 视角 | GPU | 执行方式 | 4 步延迟 | 20 步延迟 |
|---|---|---|---|---|---:|---:|
| 独立测速 188313 | EWM patch4 | 双 | L40S | eager | 126.36 | 570.79 |
| 同上 | EWM patch8 | 双 | L40S | eager | 126.97 | 564.21 |
| 同上 | EWM patch4 | 双 | L40S | compile＋权重缓存 | **37.33** | **121.91** |
| 同上 | EWM patch8 | 双 | L40S | compile＋权重缓存 | **32.91** | **101.03** |
| 独立测速 187681 | EWM patch4 | 双 | L40S | compile，早期优化版本 | 46.14 | 145.14 |
| 同上 | Sparse40 | 双 | L40S | compile，早期优化版本 | 43.92 | 134.26 |
| 独立测速 188155 | EWM patch4 | 单 | L40S | compile＋两项固定开销优化 | **35.88** | **109.34** |
| 同上 | Sparse40 | 单 | L40S | compile＋两项固定开销优化 | **35.57** | **106.77** |
| 仿真实测 181012 | Action-only 80k | 无 | Pro 6000 | 编译状态未核实，无画面解码 | 55.01 | — |
| 仿真实测 183008 | RGB-action 80k | 无，预测双 RGB | RTX 5090 | 编译状态未核实，含 RGB 解码 | 169.95 | — |
| 仿真实测 180658 | EWM patch4 80k | 双 | RTX 5090 | 编译状态未核实，含 event 解码 | 178.49 | — |
| 仿真实测 190280 | EWM patch8 80k | 双 | RTX 5090 | eager，含 event 解码 | 166.10 | — |
| 仿真实测 190622 | Sparse40 70k | 双 | RTX 5090 | eager，含 event 解码 | 175.91 | — |
| 仿真实测 190817 | Sparse40 80k | 双 | L40S | eager，含 event 解码 | 162.96 | — |

独立测速及原始记录链接见：[inference_time.md](/projects/hdd/myproject/CVPR2027/docs/inference_time.md)。

**目前结果：patch8 编译后确实更快，4 步延迟下降 11.8%，但成功率明显下降；Sparse 有训练计算和显存收益，推理收益较小，最终成功率也未超过原版。** 50k 的优势没有延续到 70k／80k。

## 未完成

* Action-only、RGB-action、原版 EWM 的部分历史仿真记录未核实编译状态；目前也没有这五种模型统一硬件、统一解码设置的完整配对测速。

## 下一步建议

* 当前保留 patch4 作为主基线。若用于论文速度对比，统一硬件、compile、4 步采样和关闭画面解码后再比较，避免使用仿真日志直接计算模型加速比。