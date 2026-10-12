对于你这个场景，我建议不要在 Isaac Sim → LeRobot 这一步就把 event 不可逆地只保存成 voxel grid。

更合适的是：

Isaac Sim / Event Camera
        ↓
raw events: (x, y, t, p)
        ↓
按 episode 连续保存
        ↓
为每个 LeRobot timestep 保存 event 的 start/end offset
        ↓
训练时 / 预处理时
        ↓
Voxel Grid / Time Surface / Event Tokens

也就是说，原始 event 是 ground truth 数据，voxel grid 是模型输入 representation。这两层最好分开。

我推荐的数据结构

假设你的 policy / dataset 是 20 Hz：

t0          t1          t2          t3
|-----------|-----------|-----------|
   events       events      events
    50 ms        50 ms       50 ms

 action_0     action_1     action_2
 RGB_0        RGB_1        RGB_2

每一个 LeRobot timestep 对应：

{
    "observation.image": RGB_t,
    "observation.state": state_t,
    "action": action_t,

    "event.start_idx": s,
    "event.end_idx": e,
}

而 episode 外单独保存：

events = [
    [x, y, timestamp, polarity],
    [x, y, timestamp, polarity],
    ...
]

例如一个 episode：

episode_00001/
├── events.npy
│      shape = [N_events, 4]
│      columns = x, y, t, polarity
│
├── event_offsets.npy
│      shape = [T, 2]
│      [[start0, end0],
│       [start1, end1],
│       ...]
│
└── normal LeRobot data
       RGB
       state
       action
       task
       timestamp

这样第 i 个 timestep：

s, e = offsets[i]

events_i = events[s:e]

就得到：

[t_i - Δt, t_i)

这段时间内的全部原始 event。ce