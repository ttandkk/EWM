# 球体皮肤与桌面背景 v1

五件可复用的球体视觉资产：`baseball` 棒球、`globe_ball` 地球球、`mini_basketball` 篮球、`soccer_ball` 足球、`billiard_ball` 黑色 8 号台球。图案由项目内程序生成；地球使用手绘风格大陆轮廓，不作为地理地图。

每件 USD 默认 prim 为 `/Prop`，米制、Z 向上，原点在底面中心，直径 6 cm。球面为平滑三角网格，1024×512 PNG 纹理采用相对路径引用；复制整个目录即可搬迁。不含碰撞、质量或刚体配置，不修改灯光、阴影或渲染采样。

## 四个球任务的默认外观

| 历史任务名 | 操作物体皮肤 | 三个不同的桌面背景球 |
| --- | --- | --- |
| `tennis_rolling` | 原有网球 | 足球、篮球、棒球 |
| `tennis_falling` | 足球 | 棒球、地球球、8 号台球 |
| `tennis_parabolic` | 篮球 | 足球、地球球、8 号台球 |
| `tennis_container` | 棒球 | 篮球、足球、地球球 |

这是外观替换，任务球沿用原球体的半径、质量、摩擦、弹性、轨迹与成功判据，并非真实足球／篮球的尺寸和动力学。新皮肤围绕原球心缩放，原解析球碰撞体只隐藏渲染，不删除。滚动网球仍使用原来的 `tennis_felt_seam_v2`。

背景球均为 6 cm 的静态视觉道具，分别布置在各任务主视角内的空闲桌面；容器任务分布在容器投影两侧及前方，避免预持容器遮挡；每回合按 seed 独立扰动 XY ±1 cm、偏航 ±12°，不消耗任务采样 RNG，飞行过程中保持静止。布局适配原主摄像头，包括容器 spatial 协议的 32 mm 焦距。资产不参与物理碰撞，摆放仍避开底座和接球走廊。

按用户标注，滚动任务的背景足球移至桌面左前侧 `(0.40, -0.305)`，篮球和棒球保留原位。自由落体任务改为前后错开的不对称布局：棒球在左前侧 `(0.32, -0.32)`，地球球在左后侧 `(0.02, -0.185)`，8 号台球在右侧 `(0.16, 0.31)`。两个抛物线任务布局保持原值。

`ball_appearance` 记录 `piper_sports_ball_skins_v1:<kind>`，`ball_skin` 记录皮肤名称；运行时及采集的 `initial.appearance` 记录背景版本和精确布局：滚动与自由落体任务为 `piper_ball_tabletop_assets_v2`，两个抛物线任务为 `piper_ball_tabletop_assets_v1`。任务注册名保持兼容，文字指令同步使用足球／篮球／棒球。既有录制文件不改写，新采数据应保留这些外观元数据。

## 复用与生成

使用现有 `piper_tabletop_props` 接口，显式传入球资产目录：

```python
from piper_ball_props import ASSET_DIR
from piper_tabletop_props import PropPlacement, configure_tabletop_props

configure_tabletop_props(
    scene, [PropPlacement('globe_ball', (0.02, -0.30), yaw_degrees=10)],
    table_z=0.0002, prefix='extra_ball', asset_dir=ASSET_DIR,
)
```

在 EWM 根目录重建和检查：

```bash
source migration/deploy_env.sh
"$SIMPY" Event-DataCollection/scripts/build_piper_ball_props.py
PYTHONPATH=Event-DataCollection/scripts "$SIMPY" -m unittest discover \
  -s Event-DataCollection/tests -p test_piper_ball_props.py -v
PYTHONPATH=Event-DataCollection/scripts "$SIMPY" -m unittest discover \
  -s Event-DataCollection/tests -p test_piper_task_tabletop_assets.py -v
```

构建器支持 `--output-dir`；试验新图案请先写入 `tmp/`。GPU 预览可复用 `render_piper_tabletop_props.py`，传入本目录作为 `--asset-dir`，同时用 `--layout-json` 指定球体布局（原渲染器默认组合使用方块资产）。
