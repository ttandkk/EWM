# 桌面背景道具 v1

14 件独立 USD 道具，参照用户提供的外观类别制作；包装与花纹均为项目内程序生成的原创图案。包含魔方、四凸点积木、木块、棋盘块、标记块、骰子、纸箱、纸巾盒、牛奶盒、麦片盒、果汁盒、礼盒、塑料收纳盒和青花陶瓷盒。

每件 `.usdc` 的默认 prim 为 `/Prop`，单位为米，Z 轴向上，原点在底面中心。文件使用相对路径引用 `textures/`，复制整个目录即可搬迁。实际包围盒尺寸记录在 `catalog.json`；默认宽度约 6–11 cm，最高约 15 cm，支持统一缩放。

这些是静态视觉背景资产，不含刚体、质量或碰撞，也不覆盖场景光源、阴影与渲染采样设置。摆放时应避开机器人工作区、移动载体和目标物体，防止遮挡；当前传送带、移动箱子、套碗与试管任务默认使用下述布置，四个球任务使用独立的 [球体资产库](../piper_ball_props/README.md)。`marker_cube` 是装饰图案，不是已标定的 ArUco/AprilTag。塑料盒使用薄壁几何与 USD Preview Surface 透明度，透射外观随渲染器而异。

## 任务默认外观和背景

前两个任务的外观版本为 `piper_first_two_tabletop_assets_v4`；套碗与试管的背景版本为 `piper_bowl_tube_tabletop_assets_v2`。实现位于 `scripts/piper_tasks/tabletop_assets.py`。

| 任务 | 参与操作的物体 | 桌面背景 |
| --- | --- | --- |
| 传送带 | 魔方、红色骰子，均为原来的 4 cm | 牛奶盒、纸巾盒、纸箱，共 3 件 |
| 移动箱子 | 3 cm 红色礼盒在下，3 cm 贴胶带纸箱包裹在上 | 果汁盒、纸巾盒、积木，共 3 件 |
| 圆盘套碗 | 原有绿色、紫色浅碗 | 青花瓷盒、木块、纸巾盒，共 3 件 |
| 旋转试管 | 原有试管及旋转孔架 | 青花瓷盒、木块、纸巾盒，共 3 件 |

任务物体使用原有立方体碰撞形状、质量、摩擦与刚体配置，外层视觉模型与原尺寸和中心对齐。操作礼盒采用 `flat_ribbon` 外观：保留红色包装、金色缎带和盒盖，仅在该实例上去掉凸起的蝴蝶结，使上层包裹可以平放；背景礼盒和原资产仍保留蝴蝶结。盒盖及包裹折缝等小幅尺寸差异逐轴缩放到原来的 3 cm 包围盒。移动箱子的文字指令同步采用红色礼盒／小纸箱包裹的身份；内部历史 `blue` / `orange` 指标字段继续表示第 1 / 2 个物体，以保持已有诊断数据可读。

按用户标注，背景分散在后侧、载体两旁及前方的空闲桌面。传送带的牛奶盒位于前方；移动箱子的果汁盒位于传送带左侧、纸巾盒位于前方。移除传送带右后方的大青花盒、礼盒，以及移动箱子右后方的大塑料盒、青花盒。两场景各保留三件背景，在主视角内完整可见；保留道具的逐 seed 位置、角度与缩放不变，操作物体尺寸保持原值。

背景沿空闲桌面分散摆放，避开底座、传送带、纸箱运动区域和源物块；位置在锚点附近独立扰动 ±1 cm，朝向扰动 ±12°。尺寸固定，使用独立 RNG，仅在 `runtime.reset(seed)` 时更新，同一 seed 可复现，轨迹进行中保持静止。

套碗的纸巾盒位于圆盘左侧，木块位于右前侧；试管的纸巾盒位于孔架左侧。主相机和物体缩放保持原值。锚点如下，位置以米为单位，原资产底面贴桌面：

| 任务 | 道具 | XY 锚点 | 基准偏航 | 缩放 |
| --- | --- | --- | --- | --- |
| 传送带 | 牛奶盒 | `(0.45, 0.012)` | −15° | 0.45 |
| 移动箱子 | 果汁盒 | `(0.24, -0.26)` | −15° | 0.50 |
| 移动箱子 | 纸巾盒 | `(0.445, -0.024)` | 20° | 0.40 |
| 套碗／试管 | 青花瓷盒 | `(0.02, 0.30)` | 15° | 0.65 |
| 套碗 | 木块 | `(0.374, 0.198)` | −12° | 0.60 |
| 套碗 | 纸巾盒 | `(0.19, -0.26)` | 8° | 0.60 |
| 试管 | 木块 | `(0.02, -0.17)` | −12° | 0.60 |
| 试管 | 纸巾盒 | `(0.303, -0.173)` | 8° | 0.60 |

运行时 metadata 和采集／评估的 `initial.appearance` 记录外观版本、seed 及全部背景的精确位置、角度和缩放。前两个任务还记录操作方块身份与视觉变体；套碗和试管的操作物体保持原样。现有录制数据不会被改写；新采数据应按相应外观版本单独保留。

## 在 IsaacLab 任务中摆放

对其他任务或需要额外背景时，在 `make_env_cfg` 之后、创建环境之前调用；传送带、移动箱子、套碗、试管各 3 件，已由任务配置添加：

```python
from piper_tabletop_props import PropPlacement, configure_tabletop_props

cfg = task.make_env_cfg(config, device="cpu", seed=seed)
configure_tabletop_props(
    cfg.scene,
    [
        PropPlacement("wooden_block", xy=(0.82, -0.30), yaw_degrees=15, scale=1.0),
        PropPlacement("milk_box", xy=(0.98, -0.30), yaw_degrees=-10, scale=0.8),
        PropPlacement("ceramic_box", xy=(0.95, 0.30), scale=1.0),
    ],
    prefix="extra_background",  # 与四个任务的默认 background_prop 前缀区分
    table_z=0.0002,  # 当前 Piper CleanTabletop 的上表面；其他桌面请传入实际高度
)
```

`xy` 和 `table_z` 使用环境的局部坐标，yaw 单位为度，scale 为统一比例。返回添加的配置项名称；重复名称会报错，不会覆盖已有资产。上面的坐标仅演示接口，各任务需检查相机可见性和工作区。

若直接编辑 USD stage，可使用：

```python
from piper_tabletop_props import place_tabletop_props
place_tabletop_props(stage, placements, table_z=0.0, parent="/World/BackgroundProps")
```

## 生成与检查

在 EWM 项目根目录执行；构建与静态检查不需要 GPU：

```bash
source migration/deploy_env.sh
"$SIMPY" Event-DataCollection/scripts/build_piper_tabletop_props.py
PYTHONPATH=Event-DataCollection/scripts "$SIMPY" -m unittest discover \
  -s Event-DataCollection/tests -p test_piper_tabletop_props.py -v
```

渲染总览与桌面组合需要已配置的 Isaac Sim GPU 环境；沿用 64 SPP：

```bash
"$SIMPY" Event-DataCollection/scripts/render_piper_tabletop_props.py \
  --output-dir tmp/tabletop_props_preview \
  --headless --enable_cameras --device cpu
```

输出 `catalogue.png`、14 张单件渲染、`tabletop_arrangement.png`、布局 JSON 与验收信息。输出目录必须为新目录。可通过 `--asset-dir tmp/your_props` 预览另一个资产目录，通过 `--layout-json layout.json` 指定组合，格式示例：

```json
[
  {"kind": "wooden_block", "xy": [-0.2, 0.1], "yaw_degrees": 15, "scale": 1.0},
  {"kind": "milk_box", "xy": [0.0, 0.1], "scale": 0.8},
  {"kind": "ceramic_box", "xy": [0.2, 0.1], "yaw_degrees": -10}
]
```

构建器仅写 `--output-dir` 中的同名资产和贴图；要试新材质，请先指定 `tmp/` 下的新目录。运行时不需要 Pillow 或几何构建器，只读取 USD 和 PNG。
