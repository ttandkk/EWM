AgileX 官方/社区提供的 Isaac Sim / Isaac Lab 集成资源  有没有提供自动采集一些简单任务数据的脚本，能不能结合evis的事件相机

有，而且我刚查到一个比前面更直接的：AgileX 这套 PiPER Isaac Lab Data Collection 已经提供了“自动方块堆叠并连续保存成功轨迹”的脚本，不是必须键盘 teleop。

他们新增了：

scripts/tools/record_ik_stack.py

对应任务：

Isaac-Stack-Cube-Piper-IK-Rel-v0

自动流程是：

识别 cube 位置
    ↓
生成 Cartesian trajectory
    ↓
Approach
    ↓
Descend
    ↓
Grasp
    ↓
Lift
    ↓
Move above target
    ↓
Place
    ↓
Success detection
    ↓
Isaac Lab Recorder
    ↓
HDF5

支持完全无人干预地连续生成成功 demonstration，官方示例就是：

python scripts/tools/record_ik_stack.py \
    --task Isaac-Stack-Cube-Piper-IK-Rel-v0 \
    --device cuda \
    --dataset_file ./datasets/ik_dataset.hdf5 \
    --num_demos 50

这基本就是你现在需要的东西。这个 pipeline 使用 Differential IK、PCHIP 平滑、自动 success detection，并把成功轨迹输出成 HDF5。

另外 AgileX 的 robot_lab 还已经有 PiPER 的 Reach 和 Open Drawer 环境：

Isaac-Reach-Piper-v0
Isaac-Open-Drawer-Piper-v0

但这两个目前仓库给的是 RSL-RL train/play workflow，不像 cube stack 那个已经明确给出了开箱即用的 autonomous demonstration recorder。因此如果你只是要尽快得到 50 条数据，我会先用 Cube Stack。

而且 EVIS 非常适合直接接进去

这点比我之前预想的还简单。

现在 EVIS 已经专门提供了：

DVSCameraCfg
DVSCamera

其中 DVSCameraCfg 本身就是 Isaac Lab CameraCfg 的 drop-in 替代品。官方示例明确说，可以把：

CameraCfg(...)

换成：

DVSCameraCfg(...)

原来的：

camera pose
robot mounting
resolution
scene

都不用改。

所以 PiPER 可以直接变成：

PiPER wrist
   │
   ├── RGB
   │
   └── EVIS Event Camera

例如在 Piper task 的 scene config 里加：

from dvs_gen.sensors import DVSCameraCfg

dvs_cam = DVSCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/piper_hand/dvs_cam",
    height=480,
    width=640,
    threshold=0.15,
    event_source="hdr",
    spawn=...,
    offset=...,
)

EVIS 会自动请求：

HdrColor
RGB
motion vectors
depth

再通过 motion-vector interpolation 产生高频 event。

我会怎么改 record_ik_stack.py

其实只需要在原来的 loop 旁边插一条 event recorder。

原来：

env.step(action)
    ↓
Isaac Lab Recorder
    ↓
obs + action

改成：

                     ┌→ Isaac Lab Recorder
                     │   RGB
env.step(action) ─────┤   state
                     │   action
                     │
                     └→ EVIS
                         ↓
                     raw events
                     (x,y,t,p)

EVIS 官方 API 本身就是：

dvs = DVSCamera.from_scene(
    env.scene,
    ["dvs_cam"],
    out_dir="./event_data",
)

env.step(action)
prev = dvs.snapshot()

while running:
    env.step(action)

    cur = dvs.snapshot()

    dvs.warp_and_process(
        prev,
        cur,
        K,
        t_prev,
        DT_FINE,
    )

    prev = cur

episode 完成时：

dvs.flush(env_id, episode_idx)
dvs.reset(reset_ids)

EVIS 本身就会输出：

env0_ep0.h5
env0_ep1.h5
...

每个文件里面：

DVS/cam0/x
DVS/cam0/y
DVS/cam0/t
DVS/cam0/p

也就是我们前面说的原始：

$$ (x,y,t,p) $$

格式。

更重要的是：它已经支持 headless

EVIS 官方自己给的典型命令就是：

python scripts/simulate_warp.py \
    --num_envs 1 \
    --render_hz 125 \
    --warp 8 \
    --max_episodes 3 \
    --enable_cameras \
    --headless

所以：

服务器
+
PiPER
+
自动 cube stacking
+
RGB
+
EVIS

这个组合在技术路径上没有 GUI 依赖。

甚至 voxel 都已经给你做好了

EVIS 现在还提供自己的 PyTorch representation loader：

from dvs_gen.representations import EventReprConfig

cfg = EventReprConfig(
    representation="voxel",
    sensor_size=(640, 480),
    n_time_bins=5,
    slicing={
        "policy": "time",
        "window": 0.05,
    },
)

它目前支持：

raw events
    ↓
├── voxel
├── event_frame
└── time_surface

而且 slicing 同时支持：

fixed time
例如 50 ms

或者

fixed event count
例如 30000 events

官方例子直接就是：

representation="voxel"
n_time_bins=5
window=0.05

这几乎和我们刚才讨论的 EWM 数据设计完全一致。

因此我建议第一版不要把两个 recorder 硬合并

直接：

dataset/
│
├── robot/
│   └── ik_dataset.hdf5
│
└── events/
    ├── env0_ep0.h5
    ├── env0_ep1.h5
    ├── ...
    └── env0_ep49.h5

其中：

ik_dataset.hdf5
    RGB
    state
    action
    timestamp

event h5
    x
    y
    t
    p

然后写一个转换器：

Isaac HDF5
     +
EVIS HDF5
     ↓
timestamp alignment
     ↓
LeRobot v3

最终：

LeRobot timestep t

RGB_t
state_t
action_t
event_start_idx
event_end_idx

+ optional:
event_voxel_t

这样比强行修改 Isaac Lab 的 HDF5 Recorder，让它接受 ragged events 简单得多。

有一个与你前面 action space 问题直接相关的细节

这个现成任务名字是：

Isaac-Stack-Cube-Piper-IK-Rel-v0
                         ↑
                       IK Rel

它自动生成的控制 action 主要是 relative Cartesian / Differential IK action，不是你真机现在说的：

joint delta position

所以如果你的最终目标是对齐 Piper 真机，我建议采集时顺手额外记录：

qpos_t
qpos_t+1
joint_target_t    # 能拿到的话
EEF pose_t
EEF target_t

这样之后可以产生：

EEF relative action

以及

observed joint delta:
qpos_t+1 - qpos_t

如果能拿到 IK controller 产生的 joint target，还可以产生：

command joint delta:
q_target - q_current

这一步最好一开始就保存，数据成本几乎为零。

所以现在你的第一版 EWM 数据生成 pipeline 已经可以非常明确：

AgileX Piper
Isaac-Stack-Cube-Piper-IK-Rel-v0
              ↓
record_ik_stack.py
              ↓
       自动完成任务
              │
       ┌──────┴────────┐
       ↓               ↓
 Isaac Recorder       EVIS
       ↓               ↓
 RGB/state/action    (x,y,t,p)
       └──────┬────────┘
              ↓
       timestamp align
              ↓
        LeRobot v3
              ↓
      fixed 50 ms event
              ↓
        5/10-bin voxel
              ↓
             EWM

如果只是先采 50 条 Piper cube-stacking 数据验证 EWM，我认为这已经是目前工程成本最低的一条路线，甚至不需要先训练一个 policy。


那有没有办法做piper的仿真eval呢

可以，而且 Piper 在 Isaac Lab 里完全可以做闭环仿真 eval。现成的 robot_lab 已经注册了 Piper 的 Reach、Open Drawer，而 AgileX 的 Data Collection 项目还有 Stack Cube，这些都可以直接当 eval 环境。robot_lab 本身就是 Gym/Isaac Lab 环境体系，并且已经给出 Piper 的 train/play 入口。

对你现在这个项目，最容易的是直接把 Isaac-Stack-Cube-Piper-IK-Rel-v0 改成一个 evaluation loop。这个环境本身已经定义了 success termination，具体是 mdp.cubes_stacked；自动采集脚本也是靠这个 success condition 判断轨迹是否成功，而不是靠人工标注。所以你完全可以让 VLA/WAM policy 接管控制，然后统计 success rate。

reset env
   ↓
RGB + Event + state
   ↓
your VLA / EWM
   ↓
action
   ↓
env.step(action)
   ↓
success ?
   ↓
next trial

例如跑：

50 random seeds
↓
success / failure
↓
Success Rate = successes / 50

这已经是一套标准的 simulation policy evaluation。

目前可以直接利用的 Piper 环境大概是：

环境	用途	适合做 VLA eval
Isaac-Reach-Piper-v0	reach target	适合最初 sanity check
Isaac-Open-Drawer-Piper-v0	开抽屉	适合接触操作
Isaac-Stack-Cube-Piper-IK-Rel-v0	方块堆叠	最适合你当前实验

robot_lab 还同时提供了 Reach 的 joint-position、IK-absolute、IK-relative 和 OSC 版本，因此控制空间本身也是可以切换的。

但你的模型 action space 最好先和 eval 环境统一

现在现成 Stack-Cube 自动 controller 的动作是：

[Δx, Δy, Δz,
 Δrx, Δry, Δrz,
 gripper]

也就是 EEF relative Cartesian action。

而你真机 Piper 现在讨论的是：

[Δq1, Δq2, Δq3,
 Δq4, Δq5, Δq6,
 gripper]

也就是 joint delta。

这两个不是一个 action space。

如果你最终论文主要想和 Piper 真机保持一致，我更建议给 Stack-Cube 再加一个：

Isaac-Stack-Cube-Piper-Joint-Rel-v0

把 arm action manager 换成：

RelativeJointPositionAction

于是训练和 eval 都统一成：

$$ a_t=[\Delta q_1,\ldots,\Delta q_6,g] $$

这样最干净。

不建议模型输出 joint delta，然后 eval 时再：

joint delta
→ FK
→ EEF delta

因为多绕了一层，误差和控制语义都会变化。

Event camera 也可以一起进 eval

你训练时如果是：

RGB_t
+
Event[t-50ms,t]
+
state_t
↓
policy
↓
joint delta action

那 eval 完全可以保持一样：

Isaac Lab Piper
      │
      ├── RGB Camera
      │
      ├── EVIS
      │     ↓
      │   raw events
      │     ↓
      │  50 ms voxel
      │
      └── qpos
             ↓
           model
             ↓
        Δjoint action
             ↓
          Piper env

也就是说，EWM 不只是能用 Isaac Sim 采训练数据，也能真正闭环 eval，这是这条路线比较有价值的地方。

第一版 eval 不需要做得很大

你现在主要是验证：

RGB baseline
vs
RGB + Event
vs
RGB + Event Future Prediction

所以完全可以固定一个任务，例如 Stack Cube，然后设计：

50 trials / model

随机：

cube x/y position
cube orientation
initial joint pose

统计：

Success Rate

然后再加一些对 event 特别有意义的扰动：

Normal speed
Fast motion
Low light / brightness perturbation
Motion blur（RGB）
camera exposure variation

这样 Event 的价值会更容易体现。

我会把你的第一版实验设计成这样
                Piper Stack Cube
                       │
              50 eval episodes
                       │
       ┌───────────────┼───────────────┐
       ↓               ↓               ↓
    RGB VLA      RGB + Event      RGB + Event
                                  + Future loss
       │               │               │
       └───────────────┴───────────────┘
                       ↓
                 Success Rate

再额外报告：

Normal
Fast motion
RGB degradation

三种 setting。

这已经可以形成一个很完整的最小实验闭环：

Isaac Piper
→ 自动采数据
→ LeRobot
→ train
→ Piper simulation eval
→ success rate

所以答案不是“能不能”，而是 现成环境已经足够让你搭出 Piper 仿真 benchmark，只是需要把现有 task 的 action space 和你的真机 joint delta 统一一下。 对你现在的规模，这个改动比自己新建一个 benchmark 小很多。