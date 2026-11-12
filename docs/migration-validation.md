# Canonical package migration validation

验证日期：2026-11-11。本文区分源码/CPU 验证与实际 GPU 验收，不将终态复判当作物理回放。

用户随后明确只需要命令行安装互相隔离的原生环境，取消容器交付，并将 EWM/OpenWAM、FastWAM 模型环境安装及相关新增验收移出本次范围。本次部署验收仅覆盖 simulation 与采集 writer；已完成的模型适配代码和历史验证证据保留。取消或未运行的检查不算通过，也不再作为本次完成条件。

## 已验证范围

任务、传感器、机器人、场景与采集实现已从旧脚本迁入 `piper_bench`，旧入口只转发。完整模块对应关系见 [`benchmark/migration-map.json`](../benchmark/migration-map.json)。源布局标识为 `piper_bench_package_v1`；新产物的源码哈希包含 canonical 包，历史结果未重写。

- 11 个注册任务均可在没有 Isaac 导入、没有旧脚本路径的 Python 进程中生成配置、采样计划和 episode spec。
- 从重构前完整源码快照 `j14qdy_k` 独立生成 44 份 golden 指纹：11 任务 × train/evaluation × 两个 seed。每份覆盖五槽计划、三个 slot × 三种 retry、默认/协议配置、相机标定和有定义的桌面外观。在模型环境和仿真环境内分别比较旧、新实现，均为零字段差异。跨 NumPy 2.2.6/2.3.1 环境有 272 个相机浮点值存在舍入差异，最大 `1.1102230246251565e-16`；没有结构或非浮点差异。fixture 保留原 SHA 和独立旧源码完整 payload，对迁移后结果使用 `1e-14` 数值容差并严格检查其余结构和值。
- USD 与 FastWAM 修复阶段，两个独立 wheel 在模型/仿真解释器中均通过仓库外导入全部 11 任务、八任务 v5 场景摘要和 800 回合规划；安装包的 239 个文件与该次构建源码逐 SHA 一致。43 个第一方 USD/纹理/目录文件已逐文件哈希核对。测试禁止导入 Isaac/pxr/旧脚本，证明 CPU 计划与打包资源不依赖仿真启动。后续跨环境启动与 writer 安装修复不包含在这两个历史 wheel 内。
- 新增任务迁移及 replay CPU 测试共 62 项通过。测试使用实际正式结果中的成功/失败终态，验证重算判据而非信任保存标签；动作回放输入会检查动作完整性、连续步号、七维有限值、30 Hz 时钟和初态一致性。
- 公共 CLI 集成测试 15 项通过：8 个命令帮助无需配置模拟器、两种单任务参数写法、8 任务 × ID/背景 OOD 的 800 回合规划、正式采样步数/场景 seed/策略 seed 只能在诊断模式覆盖，以及真实本地 WebSocket 假策略服务的 preflight 和原始观测/动作往返。假服务使用明确的测试元数据，没有加载真实模型；推理请求在服务拥有者线程执行，测试结束关闭服务。
- 旧 suite 测试迁移后 17 项通过：保留启动失败继续下一格、部分结果、完整结果、缺完成标记、用户中断、恢复仅补缺失格，以及光照/速度分层的固定请求分母覆盖。checkpoint 与训练元数据使用模型侧 descriptor/provider；未读取 Hydra 运行时或启动真实仿真。
- 采集冻结聚焦测试 11 项通过：新 snapshot 包含 canonical 包及资源，采集和独立 writer 进程优先使用冻结 `PYTHONPATH`；已有 snapshot 的 resume 只校验原哈希，文件不被升级或改写。验证含完整冻结包的仓库外导入、兼容采集入口 `--plan-only`、资源篡改哈希检测及旧调度 mock。修改的三个 shell 文件通过 `bash -n`，两段内嵌 Python 也通过 AST 语法检查；这组 CPU 测试本身没有提交采集或训练任务，后续真实采集验收单独记录。
- 使用现有 simulation Python 运行旧采集仓库 CPU 测试（包括 USD 资产、Gym 注册、teleop），执行时结果为 **444 passed、6 skipped、10 failed**。其中八项在原始快照独立复现，一项是用户指定的背景默认 v4 → v5 变化，一项是旧路径位置断言；路径断言已改为检查 canonical 包，随后任务/采集相关 **31 项和 21 个 subtests 通过**，仅排除已确认的旧速度断言。

命令示例（`PYTHON` 应指向已有相应环境；未向共享环境安装依赖）：

```bash
PYTHONPATH=benchmark/src "$PYTHON" -B -m pytest -q -p no:cacheprovider \
  benchmark/tests/test_tasks_migration.py benchmark/tests/test_replay.py
PYTHONPATH=benchmark/src:Event-DataCollection/scripts "$SIM_PYTHON" -B -m pytest \
  -q -p no:cacheprovider Event-DataCollection/tests
```

## 联合回归与跨环境服务

联合 CPU 回归实际执行结果为 **252 passed、1 skipped**，覆盖 benchmark、v5 背景协议、旧静态/动态评估和 suite 故障处理。使用已存在的 FFmpeg 编码器完成视频编码/解码检查；没有向共享环境安装或升级依赖。此轮早于最后的 USD 派生兼容修复，后者的专门检查与 GPU 重验单独记录。

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PIPER_FFMPEG="$FFMPEG" \
  PYTHONPATH=benchmark/src:policies/src:OpenWAM:Event-DataCollection/scripts \
  "$SIM_PYTHON" -m pytest -q -p no:cacheprovider \
  benchmark/tests Event-DataCollection/tests/test_piper_ood_protocol.py \
  OpenWAM/tests/test_ewm_piper_tasks_eval.py \
  OpenWAM/tests/test_ewm_piper_dynamic_eval.py \
  OpenWAM/tests/test_ewm_eval_suite_failures.py
```

唯一跳过项是在仿真环境内启动模型侧 WebSocket 服务：该环境的 `websockets==12.0` 不支持服务端使用的 `ping_interval` 参数。策略包已明确要求 `websockets>=15`；该真实服务测试在模型环境通过。另以模型环境的服务进程与仿真环境的 12.0 客户端完成实际跨进程 `hello`/`predict` 往返，收到 `24×7` 物理动作并验证观测时间戳。仿真端仍可使用 12.0，无需为运行客户端升级共享环境。以上使用测试策略，不代表实际大权重 GPU 推理验收。

策略端最终目录迁移检查另有 **33 passed**：原生 FastWAM `checkpoints/weights/step_*.pt` 与 `dataset_stats.json`、指定 checkpoint step 和同一步歧义拒绝、旧 suite 向 describe/serve 转发配置/统计/路径映射、相对 `../pretrained/...` 重定位，以及直接推理依赖的源码哈希变化。descriptor 包含 checkpoint/config/normalization 和所列源码的哈希；DINO/T5/VAE/LoRA 基座等外部预训练文件尚未纳入内容哈希，不能据此声称完整模型资产一致。

该次打包阶段的模型/策略相关测试为 **156 passed**，仿真环境相关测试为 **122 passed、1 skipped**（同一服务端版本条件）。这些与前述检查存在重叠，不能相加为总测试数。该 wheel 的四个专用启动入口在两解释器中均通过；模型解释器还完成安装包的真实 WebSocket preflight/action 通信测试。wheel 审计报告为本地 `/tmp/piper-final-wheels-l2x1iuwt/validation-report.json`；benchmark SHA256 为 `751a5560cac047e9486d0bd7a66e942c31022812e380c8694832a79a786b2c0e`，policies 为 `08d4c881b195a1aebccc7a17311cfcb658c8a6892ba1978183f3b83ebdb8e1a4`。这些哈希是阶段证据，不代表后续所有源码修复。

控制器独立 golden 同时覆盖 serial/buffered/continuous、零延迟/固定延迟/不规则长延迟，9 组动作选择和时间轴均与旧实现一致。40 份完整正式 v5 场景清单与现有结果一致，包括固定时间预算；公开默认值为 `continuous_min8_v1` 和 `piper_eight_dynamic_background_ood_v5_grey_green`，旧 v4 需显式指定。

## 跨环境启动与 writer 安装验证

真实的普通 wheel 安装暴露了环境串用问题：把模型 Python 3.10 的整个 `site-packages` 传入仿真 Python 3.12，会让后者导入不兼容的 NumPy 扩展。公共启动器现使用临时目录，仅链接第一方纯 Python 包及其资源；目录在子进程结束后删除，历史冻结源码路径仍可使用。该路径覆盖 CLI 仿真、demo、validate、doctor、独立 writer，以及旧 suite 的仿真/describe/serve 子进程。`ewm serve` 也会按模型 family 选择本地配置中的策略解释器，显式 `PIPER_POLICY_PYTHON` 优先。

这一阶段的聚焦测试为 **68 passed**，含真实 Python 3.10 普通虚拟环境 → Python 3.12 仿真环境的 NumPy 隔离、资源读取、生命周期清理，以及服务提前失败后的进程清理。原 writer 环境另以 `PYTHONNOUSERSITE=1` 实际运行 `test_piper_collection_writer.py`：**5 tests / OK**，覆盖两回合写入与续写、Parquet 行、事件 PNG 数值和 H.264 视频解码。日志及命令/退出状态位于 `/tmp/piper-old-writer-isolated-validation-20261111.{log,json}`；此项是原环境回归，不能代表新 writer 安装成功。

包含跨环境桥接、writer 锁修复和 OpenWAM 统计兼容的阶段双 wheel，位于 `/tmp/piper-final-wheels-i3iplky2/dist`：benchmark SHA `d4d0135031d7e21b77c94a46b144f18cb6e826f62e59b00ea11c9f4c98e623f0`，policies SHA `b6a8c5a46431a224634dc113a0f52fc2caade7608b7146951daed76312fc8928`。241 个安装包文件与源码逐 SHA 匹配；两解释器均通过 11 个任务、8 个正式 v5 场景指纹及 800 回合 CPU 规划。独立采集 wheel 的 benchmark 源码完全相同，ZIP 哈希因构建元数据而不同。

上述安装包联合回归共有 **278 个独立用例**：模型环境首轮 220 passed、57 failed、1 skipped；57 个失败均是在仿真执行入口导入 OpenCV 时缺少 `libGL.so.1`。只将这 57 个仿真用例放到仿真解释器、同一已安装 wheel 目标补跑，全部通过，JUnit 逐项确认覆盖原失败集合。因此结论为 **跨两个适用环境 277 项通过、1 项跳过**，不能表述为单环境一次全过。跳过项需要 FastWAM 原生 provider；随后在现有 FastWAM 环境运行的独立原生处理器探针已通过，但没有将其计为这次 pytest 的通过项。新 policy-fastwam 环境验收已按用户要求移出本次范围。公共单任务/套件 `--plan-only` 和帮助命令已在两环境通过，不需要导入仿真 OpenCV。完整命令、首次失败、补跑日志与导入路径保存在 `results/deployment_refactor_smoke_20261111/installed-package-cpu/`；这些与各聚焦组有重叠，不相加。

随后针对真实容器暴露的空 `--no-checkout` 工作树恢复问题更新安装器，该阶段双 wheel 位于 `/tmp/piper-final-wheels-d6xnp5z3/dist`：benchmark SHA `6a3c7592c29403a08fdd1011120d22661029d0174c6a0d97bd6813c78fe53989`，policies SHA `3eadcfa2d7d29143c0b069b0f01cc8ac048f6794e18ed271d904d9c333b8f1b7`。241 个文件匹配构建时源码；从已安装 wheel 执行的 **22 项部署测试通过**，9 个命令帮助、50 回合单任务背景规划、8 个正式 v5 场景指纹和 800 回合套件规划通过。本次构建占用约 29.49 MiB，没有新建完整环境，也没有重跑未受影响的整组 277 项。历史容器恢复尝试复用这一构建产物，失败原因及原生安装器后续修复见下文。

独立目录 `piper-writer-fresh-20261111` 的首次安装发现锁中 8 组冲突版本：用户级 site 与 writer 虚拟环境同时进入了原快照。根据 `PYTHONNOUSERSITE=1` 的真实导入路径及 distribution RECORD 文件哈希，先保留环境内实际版本，消除了重复 pin。安装器现于创建 prefix 前拒绝规范化名称重复、非精确版本或通配符。

后续构建检查进一步发现仅去重还不够：锁内另有 **85 个用户 site 独有包**，包括 CUDA、Libero、EGL 和训练工具，不属于实际通过测试的 writer 环境。完整隔离审计得到 **68 个运行依赖 pin**，保留版本全部与原 writer 一致；加上独立锁定源码的 LeRobot 和三项安装工具，对应原环境的 72 个 distributions。原隔离环境 `pip check` 为 `No broken requirements found.`，不需要 writer metadata 例外。审计证据为 `results/deployment_refactor_smoke_20261111/writer-lock-audit.json`，最终 requirements SHA 为 `93b6ab51ccf8276cc2f44ff2cd21896eab474271295e08deaaffc74b9377d92a`。

失败日志与元数据保留，干净重装改用 `piper-writer-isolated-20261111`，避免先前额外安装的包影响验证；旧失败环境随后按用户存储清理要求删除。安装器也修复了构建工具时序与 PATH：提前安装锁内 CMake/Ninja，把目标虚拟环境的 `bin` 放在构建子进程 PATH 首位；部署专门测试为 **20 passed**，包括真实子进程工具选择。新锁下跨环境入口整组另为 **60 passed**（与前面测试有重叠）。

**全新 writer 安装及仓库外真实 I/O 已通过。** 安装耗时 337.86 秒，`pip check` 无错误、无豁免；68 个依赖版本、LeRobot 固定 commit `7e241bd630a3719a56157a497ce5d08f244784f1` 和 225 个安装包文件逐项匹配。以新 prefix 为工作目录，清除 `PYTHONPATH`/`PYTHONHOME`，直接运行 `site-packages` 中的 canonical writer，11 个任务 schema 均可加载；两个合成回合的首次保存与 resume 均退出 0，60 行 Parquet 顺序、120 张事件 PNG 精确值及 4 个 RGB 视频的各 30 帧解码全部通过。证据集中于 `results/deployment_refactor_smoke_20261111/fresh-writer-cpu/`。临时 writer prefix 及其合成验收数据已在 2026-11-12 按用户要求清理，报告和日志归档至 `results/storage_cleanup_20261112/evidence/piper-writer-isolated-20261111/validation/installed-writer-io/`。

该 I/O 探针明确 `simulation_executed=false`，没有声称合成帧具有任务成功含义。随后真实仿真采集→writer→公开 validate 流水线 **1734780 已通过**：2 次 Oracle 验收通过，2 次尝试导出 2 回合、170 帧、零失败尝试，8 个视频解码及双相机原始事件重建通过。独立目录 `piper-collection-smoke-20261111-uwh18_b6` 冻结了普通 Python 3.10 安装包入口、wheel、采样计划、FFmpeg 可执行文件哈希和运行配置；实际使用新仿真/新 writer 环境。两个完成标记、成功终态、源码/采样一致性均通过，`pipeline-result.json` 保留完整结果。FFmpeg 7.1.1 是显式指定的现有系统工具，不属于新 writer 的安装证明；仿真 prefix 在执行前后校验不变。两项验收结束后，用户要求控制存储，该临时仿真环境按下文生命周期记录归档回收。

## 策略统计与真实模型验证

OpenWAM 提交前的真实统计检查发现：Piper 导出文件使用 `eef`（7D）与 `state`（8D），原生 loader 只把 `eef_state` 识别为独立 proprio 统计，因此预测时会把 8D state 错用 7D action 统计。修复局限于公开 adapter：识别上述明确 schema，在临时副本中把 `state` 改名为 `eef_state`，逐项验证 shape/dtype/value 不变，并继续使用原生 normalizer。原始统计与 checkpoint 未改，canonical schema 原字节保留，未知或混合 schema 拒绝。descriptor 明确记录映射和 source/effective SHA；**49 项相关 CPU 测试通过**，含实际 8D normalize/7D unnormalize 和错误清理。

四种真实模型服务探针 **1734723 已结束**：EWM 联合/action-only/RGB-action 的 80k checkpoint 均完成真实权重加载、hello、一次带时间戳的有限物理动作输出与清理。联合 EWM 的实际 compile/default、关闭 CUDA Graph、KV/BF16 缓存、3 次 warmup/benchmark 和 4 步 flow 均已核对；优化后启动基准均值为 30.116 ms，动作最大误差为 0.007324。其余两项是没有 warmup 的首次 eager 预测，不能与这一均值直接比较。这些探针没有运行机器人动作，不代表闭环任务成功率。

同一任务中的 FastWAM 两步 checkpoint 在 ready/hello 前被 SIGKILL，Slurm 明确记录主机 cgroup `oom_kill`，整体状态为 `OUT_OF_MEMORY`。服务日志为空，不能断言具体原生加载阶段，也没有 CUDA OOM traceback。原失败记录及三个成功结果保留于 `piper-model-validation/four-case-1734723/audit_summary.json`。两步权重仅用于接口验证，不是训练完成的基线。

增加主机内存的单独重验 **1734847 已通过**：完整原生模型加载、hello、一次预测和正常关闭完成，作业退出 0、耗时 6 分 49 秒。返回有限的 `float32[1,7]` 物理动作，观测时间戳、步号和 30 Hz 合同一致；服务启动 371.46 秒，首次模型推理 8.610 秒。模型进程最大 RSS 为 37.668 GiB，超过原先 33 GiB 配额；PyTorch 记录 cuda:0 allocated/reserved 峰值约 25.028/25.631 GiB，cgroup 未记录 OOM。原 nvidia-smi 采样器误用宿主 GPU 编号，在分配命名空间中失败，其错误保留，不用于证明 GPU 峰值。实际峰值来自模型 wrapper 的独立统计。新增输出约 0.61 MiB，没有新权重；证据位于 `piper-model-validation/fastwam-retry-1734847/`。这仍是单次接口验证，不是闭环成功率测评。

本地未找到训练完成的 Piper OpenWAM checkpoint。独立任务 **1734785 按用户缩减范围的要求已取消，运行时间为零**。原计划是完整原生模型初始化、保存、strict reload 和一次预测，预计产生约 25 GB 临时检查点；随机初始化的动作模块仅用于接口验证，不代表微调模型或成功率。该计划及其小型准备材料保留，实际检查点没有生成，完整原生 OpenWAM 加载/预测尚未验证。它不属于本次部署验收；不会自动恢复或另行提交。

## 全新原生环境

已在新的外部目录 `piper-native-fresh-20261111` 实际执行固定版本安装，未复用已有仿真环境。使用新环境已安装的 `piper_bench`，在该 prefix 内运行并清除 `PYTHONPATH`/`PYTHONHOME`，独立下载与校验 **53/53** 个公开资产，CPU doctor 返回 `passed=true`。198 个包版本、3 个源码 commit 和兼容补丁检查通过；`pip check` 仅有锁中逐条声明的 15 条上游 metadata 例外，无新增例外。

这份 CPU 报告明确 `deployment_accepted=false`：生成时尚未执行 GPU 层；后续独立 GPU 任务的成功结果见下文。首次 `/tmp` 安装因配额不足失败，其日志保留，残留环境随后按用户存储要求清理。后续安装将缓存、临时目录和第一方构建副本放在新 prefix 中。

历史容器尝试提供了两个安装器回归案例：`1734742` 在 195 项依赖安装后因 Git 空工作树处理失败；修复后的恢复任务 `1734849` 又发现 setup.py-only editable 安装会通过 setuptools 重启嵌套 pip，丢失外层 `--no-build-isolation`。离线限制正确拒绝了未锁定的构建依赖下载。安装器现为 editable 源码明确指定 `--use-pep517`，保留锁和运行依赖版本不变；**23 项部署测试通过**，包括精确 pip/setuptools/wheel 版本下的真实离线 editable 安装与独立导入。这一修复继续用于原生安装。用户已取消容器交付，没有导出镜像；失败报告保留，临时 rootfs 与基础镜像已回收。

策略 profile 的后续构建检查确认，仅预装 Torch wheel 不足以导入 Torch，其 Python 依赖也必须先齐全。现在先安装锁内已审计的 wheel，实际导入并核对 Torch 版本，再通过 PEP517、关闭 build isolation 构建固定的 antlr4/DeepSpeed sdist；仿真和 writer 的依赖顺序不变，所有版本锁均未改。没有把原路径夸称为已复现的 DeepSpeed 必然构建失败。包含该修复的 benchmark wheel 位于 `/tmp/piper-benchmark-wheel-_fg6r38m/dist`，SHA `197b7b578bbeec6c8c18795342dc3796ad6e6476ebc39117f51f5b817a90ac15`；225 文件匹配源码，安装包下 **25 项部署测试通过**。策略 wheel 复用上述 d6xnp5z3 版本，16 文件仍匹配当前源码。

全新策略环境原拟按单 profile 安装、实际依赖/源码/视频解码/跨进程服务验证、归档、回收的顺序执行，临时 prefix 的观测容量上限为 16 GiB。EWM 原生 CPU 安装任务 **1734885 已按用户要求取消**：运行 6 分 38 秒，只完成第一批 wheel 安装，没有取得完整 setup 成功、Torch 构建前置检查、视频解码或跨进程服务验收结果。中断报告不是通过证明；调度终态与归档见 `results/storage_cleanup_20261111/policy-retirement-1734885/`。FastWAM 新环境从未提交，原有模型环境未改。相关准备脚本仅作为后续可选工作保留，不属于本次部署交付。

## 正式背景 v5 的 2000 回合复判

对五组正式背景 v5 结果，各 8 任务 × 50 回合，使用 `collection.runtime.terminal_valid` 重新判定保存的物理终态。**2000/2000 标签一致，零异常、零不一致**。未启动仿真、未重新推理、未修改原结果。

| 模型 | 回合数 | 原成功数 | 复判成功数 | 成功率 |
|---|---:|---:|---:|---:|
| Action-only | 400 | 63 | 63 | 15.75% |
| RGB-action | 400 | 48 | 48 | 12.00% |
| 对称 EWM，间隔 2 | 400 | 104 | 104 | 26.00% |
| 对称 EWM，间隔 1 | 400 | 83 | 83 | 20.75% |
| 非对称 EWM | 400 | 52 | 52 | 13.00% |

使用的结果根（相对仓库；不包含 smoke 或旧 background-speed v4）：

```text
results/piper_action_eval_262022_step080000_background_greygreen_min8_5090_20261109/ood
results/piper_ewm_eval_262022_step080000_background_greygreen_min8_decode_pro6000_20261109/ood
results/piper_three_background_greygreen_recovery_5090_20261110/rgb_action/ood
results/piper_three_background_greygreen_recovery_5090_20261110/ewm_stride1/ood
results/piper_three_background_greygreen_recovery_5090_20261110/ewm_asymmetric/ood
```

可对任一结果根独立复核，输出目录必须为新的外部目录：

```bash
./ewm replay --result-dir "$RESULT_DIR" --rejudge-only --output-dir "$AUDIT_DIR"
```

报告包含逐回合结果文件哈希、当前实现源码哈希、原标签、复判标签、错误与不一致列表。`simulation_executed=false` 明确表示只完成终态复判。

另外，对 40 个正式任务格各自第 0 回合运行了回放输入校验：共 40 条轨迹、10,808 个动作，通过保存结果/初态/scene manifest 一致性、完整步号、动作数值与时钟检查。该检查读取现有记录，仍不代表已执行物理回放。

## 八项已存在的测试失败

以下测试均在独立重构前源码快照中实际执行并复现，未通过削弱断言、调整采样或更改相机来使其通过。

| 测试 | 原始失败证据 |
|---|---|
| `test_piper_collection.py::test_recollected_rotation_id_and_ood_share_reduced_bounds` | 测试期望 slow 下限 0.255，已有协议为 0.24（0.85 与 0.8 系数不一致） |
| `test_piper_conveyor_camera.py::test_all_registered_tasks_share_one_wrist_calibration` | tennis_parabolic 已使用 `piper_tennis_lob_fixed_flight75_v1`，测试仍期望统一 `piper_centered_bracket_square75_v2` |
| `test_piper_narrow_collection.py::test_long_lob_camera_covers_entire_id_flight` | 主相机已为接取近景；旧完整飞行覆盖断言得到 margin −0.656985 m |
| `test_piper_tennis_camera.py::test_main_camera_contains_falling_flight_or_close_receiving_region[tennis_parabolic-piper_eight_dynamic_collection_v1-slow]` | slot 4，margin 0.004313 m，小于测试要求 0.005 m |
| `test_piper_tennis_wrist.py::test_fixed_wrist_contains_nominal_full_flight[tennis_container-piper_eight_dynamic_collection_v1-slow]` | slot 6，margin 0.004785 m，小于 0.005 m |
| `test_piper_tennis_wrist.py::test_fixed_wrist_contains_nominal_full_flight[tennis_container-piper_container_spatial_v3-slow]` | slot 19，margin 0.003511 m，小于 0.005 m |
| `test_piper_tennis_wrist.py::test_fixed_wrist_contains_nominal_full_flight[tennis_container-piper_container_spatial_v4-slow]` | slot 19，同样 margin 0.003511 m |
| `test_piper_turntable.py::test_basket_and_other_body_parameters_unchanged` | yaw_jitter 已分别为 3° 与 10°，测试仍要求一致 |

这些失败的基线验证分为：相机相关 90 passed / 6 failed；速度单项 1 failed；turntable 单项 1 failed。它们属于既有测试/协议状态，不代表本次包迁移引入新的物理变化，也不表示这些边界问题已修复。

## GPU 验收状态

固定动作 replay 已实现真实路径：恢复记录的场景、相机、背景版本和初态；逐步执行保存动作（包括等待/保持动作）；复用 canonical `capture_step` 记录新 RGB、原始事件、视频和终态。它不向策略服务请求动作。输出同时报告运行时成功、终态复判、原标签是否一致以及终态数值差异。

**现有环境的真实固定动作回放、修复后的全新环境 GPU doctor 和完整采集导出均已通过。** 首轮 doctor 的 USD 失败记录仍保留。这里的成功仅覆盖记录的仿真、采集与回放范围，不表示已完成模型环境安装或 OpenWAM 原生模型验收；后两项已被用户移出本次范围。可在合适的 GPU 环境使用新输出目录独立运行：

```bash
./ewm doctor --help
./ewm replay --result-dir "$TASK_RESULT_DIR" --trial 0 --output-dir "$REPLAY_DIR"
```

实际回放要求相机/RTX 渲染和 EVIS 可用，即使物理设备为 CPU。首次运行应核对 `replay_result.json` 中的 `simulation_executed`、动作数、视频解码、原始事件帧数、成功一致性和终态误差；出现 `failure.json` 时保留产物并排查，不改写原评测成绩。

首轮任务 `1734573` 于 2026-11-11 13:18–13:23 UTC 在 `gpu-pro6000-7` 实际运行。两个 Oracle 回合均成功（84/86 帧），关节运动、双 RGB、腕相机刚性安装与运动、两相机 EVIS 双极性/时间戳、两次 reset 后空初始帧、RGB/event 视频解码均通过。严格 USD 组合检查发现固定上游 `piper_base.usd` 中 `/piper/world/visuals` 指向 `piper_physics.usd</visuals/world>` 的悬空引用，因此总体 `passed=false`、`deployment_accepted=false`，作业退出码为 1。原始失败报告和视频保留在 `results/deployment_refactor_smoke_20261111`，不得将其表述为整体验收通过。

固定动作回放任务 `1734603` 已在 `gpu-pro6000-7` 完成，退出码 0、用时 53 秒。输入为 RGB-action 的 tennis_rolling/background_ood 第 0 回合、100 个动作，其原始标签为失败。输出 `simulation_executed=true`、`complete=true`、`actions_replayed=100`、`policy_requests=0`；运行时与终态复判仍为失败，`success_matches=true`。位置、速度、TCP、时间、稳定步数的最大终态差异均为 **0**。视频解码通过，事件归档为 101 帧/100 动作，包含空初始帧与末尾观测，两个相机分别记录 106,031 和 850,583 个原始事件。

回放使用冻结源码和新目录 `results/deployment_refactor_smoke_20261111/replay-output`，原成绩和动作未改。清除了仅依赖首轮 doctor 成功的调度条件，让独立回放完成；这没有跳过回放本身的任何验证。上述任务均不能证明全新环境的 GPU 验收；新环境自检单独执行。后续 USD 修复与重新验收保留原始证据，并继续执行组合检查。

USD 缺陷已在 CPU 上独立复现。只在 session layer 删除引用无法跨越原有组合边界，不能解决错误，因此未采用该办法。有效修复是在新缓存中派生五层副本，仅移除 `piper_base.usd` 的那一条无效 prepend reference；其他四层逐字节一致，原始五层文件哈希全部不变。用原生 USD 值精确比较全部实例代理后的 **761 个节点、20 个 Mesh、6,921 个属性、387 个关系**，包括 schema、metadata、默认值、连接、目标和时间采样集合，完全一致，无数值容差、无字段豁免。派生资产的组合错误为零。

原始 `Stage.Flatten().ExportToString()` 不适合作字节等价标准：输出包含来源目录和不稳定的 prototype 编号/排序。这里使用完整组合语义和原生值精确比较；字符串摘要仅用于追溯，不作为放宽数值检查的依据。生成器同时限定固定源哈希、精确坏引用及空节点条件，并保留运行时严格组合检查。CPU 等价证明本身不替代修复后的新环境 GPU doctor。

新环境内的最终 USD 与部署专门测试为 **20 passed**（6 项真实 USD、14 项部署）。覆盖首次派生、缓存复用、原资产不变、未知源拒绝、缓存文件与伪造 manifest 一同修改时仍拒绝、同进程旧 Sdf 缓存刷新，以及 `EEXIST`/`ENOTEMPTY` 并发生成处理。兼容代码的 SHA 固定在发布锁中，doctor 检查该 SHA，实际审计附带 `derivation.json`。独立等价证据保留在本轮结果的 `usd-independent-audit/` 中。

修复后的全新环境 GPU doctor **1734674 已完成，退出码 0，用时 10 分 11 秒**。它使用新 prefix 中安装的冻结 benchmark wheel，清除 `PYTHONPATH`/`PYTHONHOME`，从仓库外运行两个 Oracle 回合（84/86 帧）。RGB、腕相机、EVIS、reset/时间戳、视频检查均通过，两次严格 USD 组合错误列表均为空，`deployment_accepted=true`。224 个已安装 benchmark 文件与该次冻结源码逐 SHA 一致。脚本、冻结输入、实际报告与 acceptance summary 位于 `results/deployment_refactor_smoke_20261111/fresh-env-gpu/`。

## 验证临时文件的存储生命周期

用户随后要求控制存储。仅回收本轮创建的失败安装、已完成验收的临时仿真环境、重复构建和下载缓存，原项目环境、数据集、训练权重和实验成绩均保留。环境根部报告、冻结 wheel 和 Kit 日志归档到 `results/storage_cleanup_20261111/retired-native-environment/`；独立 GPU doctor 的输出保留在项目结果目录；临时采集输出后来按 2026-11-12 的用户清单删除，报告迁至下述新归档。上述临时 prefix 路径只表示历史运行位置，回收后不再是可直接运行的环境。精确删除清单及完成状态由同目录清理 JSON 记录。

本次清理已完成，按删除前 `du -sk` 分配占用减去约 10 MiB 的保留证据，净回收约 **36.86 GiB（39.58 GB）**。这不是个人配额查询结果。逐路径删除和原始资产保留检查见 `results/storage_cleanup_20261111/summary.json`。

容器交付取消后，唯一任务自建 rootfs 和基础镜像已删除，本轮另外回收约 **26.05 GiB**；8 份小型元数据归档逐 SHA 校验，原环境、权重与数据保留。退休清单及完成记录位于 `results/storage_cleanup_20261111/container-retirement/`。随后模型环境部署也移出本次范围，`1734785` 与 `1734885` 均已取消，不再安排相关安装或检查点验证。中断产生的唯一临时策略 prefix 已删除，额外回收约 **7.28 GiB**，8 份日志和元数据归档逐 SHA 校验通过。清理在调度终态、队列无任务且集群 300 秒进程清理时限结束后执行；完成记录见 `policy-retirement-1734885/reclaim-result.json`。原有模型环境、权重和数据保留。

2026-11-12（北京时间），按用户截图逐项删除项目外的 **9 个临时目录和 9 个文件**，包括临时 writer、采集验收输出、容器准备和模型验证准备目录。保留 421 份小型报告、日志和运行脚本（约 3.1 MB），归档逐 SHA 校验；不保留这些临时目录中的环境、重复源码和验收数据/视频。扣除归档后按分配占用估算净回收约 **1.51 GiB**。原项目环境、正式数据、权重与实验结果仍保留。精确清单、归档映射及完成结果见 `results/storage_cleanup_20261112/manifest.json` 和 `result.json`；历史验收报告中的原绝对路径仅代表当时运行位置。
