# 绘图与分析脚本说明

本目录保存本次分析脚本副本；`../data/` 保存轻量 JSON，`../figures/` 保存原实验图片。**原始数据、约 401 MB 的 50 轨迹 latent 缓存、PCA 基底、VAE 权重和数据集未打包**。依赖这些文件的脚本需要原工作区中的对应缓存；脚本不训练 EWM。结果和结论的适用范围见上级报告。

## 从打包 JSON 复画

在仓库根目录 `EWM/` 下运行。各命令的 `EWM_PLOT_OUTPUT` **必须指定一个尚不存在的目录**；脚本会创建它，并在目录已存在时拒绝运行，从而保护已归档图片。可自行修改下列输出路径。需要 NumPy、Matplotlib；decoder 图还需要 Pillow。

```bash
OPENBLAS_NUM_THREADS=1 EWM_PLOT_OUTPUT=/tmp/ewm-doc-metrics /usr/bin/python3 -B docs/latent_representation_analysis_20261104/scripts/plot_metrics.py
OPENBLAS_NUM_THREADS=1 EWM_PLOT_OUTPUT=/tmp/ewm-doc-pca /usr/bin/python3 -B docs/latent_representation_analysis_20261104/scripts/plot_falling.py
OPENBLAS_NUM_THREADS=1 EWM_PLOT_OUTPUT=/tmp/ewm-doc-decoder /usr/bin/python3 -B docs/latent_representation_analysis_20261104/scripts/summarize.py
OPENBLAS_NUM_THREADS=1 EWM_PLOT_OUTPUT=/tmp/ewm-doc-pilot-ridge /usr/bin/python3 -B docs/latent_representation_analysis_20261104/scripts/pilot_plot_probe.py
```

`plot_metrics.py` 生成全任务方差和有效秩图、阶段比值图；`plot_falling.py` 生成落球 PCA 训练及留出重建曲线；`summarize.py` 根据逐窗口 JSON 计算并绘制 decoder 图、从归档的 event 示例生成统一亮度调整图；`pilot_plot_probe.py` 生成早期 5 轨迹 ridge 图。汇总是按窗口等权，不应把窗口数当成独立轨迹数。

## 需要外部 latent 缓存的图

以下默认读取 `tmp/latent_expanded_20261104/`；也可用环境变量 `EWM_LATENT_INPUT` 指向同结构的缓存目录。`falling_pca.py` 还需要 SciPy；`visualize.py` 生成所有任务二维 PCA 和探索性 t-SNE，耗时较前述 JSON 绘图更长。

```bash
OPENBLAS_NUM_THREADS=1 EWM_PLOT_OUTPUT=/tmp/ewm-doc-spectra /usr/bin/python3 -B docs/latent_representation_analysis_20261104/scripts/plot_spectra.py
OPENBLAS_NUM_THREADS=1 EWM_PLOT_OUTPUT=/tmp/ewm-doc-falling-pca /usr/bin/python3 -B docs/latent_representation_analysis_20261104/scripts/falling_pca.py
OPENBLAS_NUM_THREADS=1 EWM_PLOT_OUTPUT=/tmp/ewm-doc-2d /usr/bin/python3 -B docs/latent_representation_analysis_20261104/scripts/visualize.py
```

`t-SNE` 是早期自实现探索图，尚未与成熟实现交叉验证或系统检查收敛；不能以散点面积、簇间距离直接比较跨模态紧凑程度。早期 5 轨迹的 `pilot_plot_pca.py` 需要 `/tmp/ewm-event-rgb-study/` 的逐任务 latent；可用 `EWM_PILOT_LATENT_INPUT` 指定其他路径，同样要求设置新建的 `EWM_PLOT_OUTPUT`。

## 研究脚本与原始依赖

- `pca_probe.py` 接受必填的 `--input-dir` 和 `--output-dir`；后者必须不存在。它会重新计算 PCA 并保存落球基底，依赖 50 轨迹 latent 缓存。`phase_probe.py` 同样读取该缓存，使用 `EWM_ANALYSIS_OUTPUT` 指定尚不存在的结果文件。`check_pca.py` 读取原 `tmp/latent_pca_reconstruction_20261104/pca/` 内的基底及 `tmp/latent_expanded_20261104/`，不会写文件。
- `decode_probe.py` 需要原 VAE 权重、原始数据集、冻结基底、Python 深度学习环境等；它的 `--output` 必须是尚不存在的目录。该脚本并非仅凭打包 JSON 可运行，具体参数和运行环境见原实验目录 `tmp/latent_pca_decoder_20261104/README.md`。
- `ewm_flow_conditional_probe.py`、`ewm_phase_residual_probe.py`、`prediction_probe.py` 是早期 5 轨迹代理实验，分别要求通过 `EWM_ANALYSIS_OUTPUT` 指定尚不存在的输出文件；前两者默认读取 `tmp/event_rgb_latent_pilot_20261104/`，可使用 `EWM_PILOT_LATENT_INPUT` 指定缓存。`prediction_probe.py` 还依赖物体状态数据和 PyArrow。它们不是正式 EWM checkpoint 的预测评估。
- `pilot_latent_variance_audit.py` 需要早期 latent，使用 `--input-dir` 指定位置、`--output` 指定尚不存在的文件。各研究脚本保留原实验算法，只调整了缓存定位及防覆盖输出；不要把历史探针作为新数据的独立验证。

## 生成 PDF 报告

`render_report_pdf.py` 从上级 `README.md` 生成包含插图、中文和矢量公式的 PDF。它需要额外的 Python 包 `Markdown`、`Matplotlib`、`Pillow`、`NumPy`、`WeasyPrint`，以及 WeasyPrint 所需的 Pango、HarfBuzz、Fontconfig 等系统库；这些依赖和字体没有打包到本目录，也没有加入项目依赖。请自行准备有简体中文字形的字体，例如 [Noto 官方仓库的 NotoSansCJKsc-Regular.otf](https://github.com/notofonts/noto-cjk/blob/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf)。

从仓库根目录运行（替换为实际字体和 Python 环境路径）：

```bash
python -B docs/latent_representation_analysis_20261104/scripts/render_report_pdf.py \
  --font /path/to/NotoSansCJKsc-Regular.otf \
  --output /tmp/ewm-latent-analysis-report.pdf
```

输出文件必须不存在，脚本采用独占创建，防止覆盖。15 张原始图全部内嵌；较长的八任务 PCA 总图在接近空白行处拆分为四段以改善阅读，原始图片不变。公式生成矢量 SVG 嵌入 PDF，公式字形不是可检索文本。正文和中文字体会嵌入 PDF。生成过程使用系统临时目录存放中间图片。
