## 最新验证：BF16 residual 与 selective checkpoint

本节记录 2026-09-26 对 D768/L12/BS64 dual-event 训练的后续验证。

- **P0：修复 residual dtype promotion。** `camera` 与 `modality` 的 FP32 参数在相加时会把 autocast 下的 residual stream 提升为 FP32；现已在 `EWMContextEncoder.forward` 和 `_encode_events` 中将临时 embedding 转为输入 activation 的 dtype，保留 FP32 master parameters。
  - BS1、GC off：全部 joint block 的 event/action 输入输出现为 BF16，loss 有限。
  - BS64、完整 GC：修复后的源码验证通过，peak 为 34,259 MiB（此 harness 与原 profiler 不同，不能直接与 41.6 GiB 横比）。
  - 同一 BS1 harness 中，GC off peak 从 9,699 MiB 降至 9,636 MiB；dtype 修复正确且无损，但单独不足以消除超大 activation。
- **P1：修复后重测 BS64 selective checkpoint。** 两种互补配置仍在第一步 forward OOM：
  - blocks checkpoint on / codec off：92.46 GiB allocated，仍缺约 334 MiB。
  - blocks checkpoint off / codec on：92.00 GiB allocated，仍缺约 864 MiB。
- 因此，BS64 下 codec 与 joint blocks 都必须 checkpoint；继续细调 checkpoint 粒度的收益预期很低。

**结论：** FP32 promotion 是真实的实现缺陷，已修复；但最主要的显存/计算问题仍是 4,608 event tokens 及其 codec/12-layer joint backbone activation。

下一阶段应做 P2：保持正式配置不变，先实现可回退的 event-token pooling/upsampling ablation（16x16 → 8x8），以测量吞吐、峰值显存以及后续训练质量的变化。

---

从这份 profiler 看，**训练瓶颈几乎完全在 GPU 计算路径**。平均一步 9.78 s，而 H2D 只有 64 ms、context encoder 约 179 ms、diagnostics 约 19 ms，所以这些都不是重点。  

### 1. 第一大瓶颈：Event codec 的卷积

最明显的是：

* `convolution_backward`：**12.78% CUDA time，约 1.21 s/step**
* `cudnn_convolution`：**9.16%，约 0.87 s/step**

也就是说，仅 profiler 里直接看到的卷积前向 + backward 就已经大约：

$$
1.21+0.87 \approx 2.08\text{ s/step}
$$

占整个 9.78 s step 的约 **21%**。

而且还有：

* GroupNorm forward/backward
* SiLU forward/backward
* transpose convolution
* avg_pool3d
* convolution dgrad/wgrad kernels

这些基本也是 event codec 附近产生的计算。  

所以如果你的 event encoder/decoder 是很多层 **3D Conv + GN + SiLU + ConvTranspose**，它现在确实是一个相当重的模块。

---

### 2. 第二大瓶颈：12-layer 主干，不只是 Attention，MLP/Linear 也很重

Transformer 本身也占了非常大的比例。

直接看到：

* `mm`：7.13%
* `addmm`：6.66%
* FlashAttention backward：6.16%
* FlashAttention forward：4.76%
* efficient attention backward：2.24%
* efficient attention forward：1.60%

  

因此你的主干不是单纯“attention 太慢”，而是标准 Transformer 的：

$$
\text{Attention}+\text{QKV/O Linear}+\text{MLP}
$$

整体都比较重。

尤其 D=768、L=12、batch=64，本身已经会产生大量 GEMM。

所以目前可以粗略理解成：

> **Event codec ≈ 一个主要瓶颈，Transformer backbone ≈ 另一个主要瓶颈。**

不是某一个单独 attention kernel 出问题。

---

### 3. 一个很值得注意的问题：`copy / cat / elementwise` 非常多

这个 profile 里比较异常的一点是：

* `copy_`：**12.33%**
* `mul`：**10.48%**
* `cat`：**4.68%**
* `add_`：3.74%
* 大量 vectorized / elementwise kernels
* NCHW↔NHWC conversion 也有明显开销

  

调用次数也很夸张：

* `copy_`: **34,810 次**
* elementwise kernel：很多达到几千到两万次
* `cat`: 1,438 次

这说明你的模型现在还有一个问题：

> **kernel 很碎，tensor manipulation 很多。**

也就是说不只是 FLOPs 大，还存在：

```text
cat
copy
reshape/layout conversion
mul/add
norm
small kernels
```

这种大量小操作。

这通常出现在：

* 每层不断 concat event/action/context token
* 频繁构造 conditioning
* repeatedly expand/repeat tensor
* 3D codec 内 layout 频繁切换
* checkpoint 导致这些操作又执行一遍

因此这里其实还有优化空间。

---

## 4. Gradient checkpointing 很可能是当前最值得先测试的东西

你现在：

> peak allocated = **41.65 GB**

而 GPU 是 96 GB。

也就是说 **还剩超过 50 GB 显存空间**。

此时开启 gradient checkpointing 的性价比就比较值得怀疑了。

Checkpoint 的原理就是：

$$
\text{省 activation memory}
\quad\Longleftrightarrow\quad
\text{backward 时重新 forward}
$$

于是你的：

* Conv
* Attention
* MLP
* elementwise operation

都会产生额外计算。

profile 里面 attention forward 的调用数量明显很多，例如 FlashAttention：

* forward：96 calls
* backward：24 calls



虽然不能简单把这个 4:1 全归因于 checkpointing，但它和存在大量 forward 重算是吻合的。

### 所以我现在最优先建议你做的实验反而非常简单：

```text
D768 / L12 / bin1
batch=64
gradient_checkpointing=False
```

看看：

```text
peak memory
step time
```

不用直接开始正式训练，只 profile 5~10 step。

如果结果比如：

```text
41.6 GB → 65~80 GB
9.8 s → 6~8 s
```

那就明显应该关掉 checkpoint。

即使 batch=64 关 checkpoint OOM，也可以比较：

```text
GC ON,  batch=64
GC OFF, batch=48
GC OFF, batch=32 + grad accumulation
```

重点比较 **samples/sec**，而不是单纯 step/sec。

---

## 5. DINO/T5 完全不用动

这个结果很好：

> `ewm.context_encoder = 178.648 ms / step`

也就是：

$$
178.6/9783.8\approx1.83\%
$$



因此你之前担心：

> T5 + DINO 会不会成为 EWM 的训练负担？

从现在的结果来说，**不会**。

冻结 DINO/T5 只花约 1.8%，完全可以接受。

即使把它优化掉一半：

$$
178\rightarrow 90ms
$$

整步也才改善不到 1%。

没必要在这里花时间。

---

## 6. 数据加载也完全不用优化

你的：

```text
data_wait = 26 ms
H2D      = 64 ms
step     = 9784 ms
```

分别只有：

$$
0.27\%,\quad0.66\%
$$

而实际 profiler 里的 H2D 同样只有约 0.67%。

所以：

* workers 从 4 → 8
* pin memory
* prefetch
* SSD/HDD
* dataloader pipeline

目前都几乎不可能显著提升训练速度。

---

## 7. 我会这样排优化优先级

| 优先级   | 部分                       | 当前判断                    | 是否值得马上优化  |
| ----- | ------------------------ | ----------------------- | --------- |
| **1** | Gradient checkpointing   | 很可能存在大量重算，而且显存只有 41.6G  | **非常值得**  |
| **2** | Event codec Conv3D       | 单前后向直接约 2.1 s/step      | **值得**    |
| **3** | Transformer L12          | Attention + GEMM/MLP 很重 | **值得**    |
| **4** | `copy/cat/mul` 等碎 kernel | 数量异常多                   | **之后值得查** |
| 5     | DINO/T5                  | 179 ms                  | 不值得       |
| 6     | H2D                      | 64 ms                   | 不值得       |
| 7     | dataloader               | 26 ms                   | 不值得       |
| 8     | diagnostics              | 19 ms                   | 不值得       |

### 还有一个值得检查的小细节

你的卡是 **RTX PRO 6000 Blackwell**，但 profile 里出现了不少：

```text
cutlass_80_tensorop
sm80_xmma...
```

例如这些 kernel 占的时间也不低。

这**不一定意味着有问题**，有些库的 kernel 命名不能简单按名字判断实际硬件路径；但考虑到你是 Blackwell，我会顺手确认一下 PyTorch / FlashAttention / xFormers 是否真正走了针对你当前 CUDA/PyTorch 环境的最佳 backend。这个优先级低于“先关 checkpoint 实测”。

**因此当前最关键的结论是：先不要动模型结构。第一件事应该是做 `gradient_checkpointing=False` 的同配置 profile。** 你有 96 GB 显存而目前峰值只有 41.6 GB，这里很可能是最容易拿到训练吞吐提升的地方。


对，**单看 0.7B / 0.3B trainable，9.8 s/step 确实偏慢**。但真正异常的不是“参数少却训练慢”本身，而是你的模型明显处在一个**activation / token / kernel efficiency 主导**的 regime，参数量已经不是主要决定因素。

从这份 profile 看，至少有 4 个信号值得警惕。

第一，**0.3B trainable 并不等于只算 0.3B**。冻结的参数仍然要 forward；更重要的是你的 EWM 里还有大量 3D/event codec 计算。卷积 backward 单项就占 12.8%，卷积 forward 又占 9.2%，再加各种 Conv3D、GroupNorm、SiLU 等。 

但即使考虑这些，**96GB 卡上 BS32 无 checkpoint 直接吃到 93GB，也明显说明 activation 太大了**。这和 0.7B 参数本身不匹配。真正需要查的是每个阶段的 tensor shape，例如：

```text
RGB / event input
↓
event encoder feature
↓
flatten 后 event token 数
↓
与 action / context concat 后 token 数
↓
joint transformer 每层 sequence length
↓
FFN hidden activation
```

很可能某个地方 token 数量非常大。

例如 D=768 的 Transformer，如果最终序列是 1000 token 和 5000 token，虽然参数量完全一样，训练成本差得非常大。尤其 attention：

$$
O(N^2D)
$$

FFN 则：

$$
O(ND^2)
$$

所以 **parameter count 在这里很容易误导**。

第二，我更在意的是你 profiler 里的：

```text
copy_
cat
mul
elementwise
NCHW -> NHWC
NHWC -> NCHW
```

数量非常夸张。

例如：

* `copy_`：34,810 次，CUDA 12.33%
* `mul`：CUDA 10.48%
* `cat`：4.68%
* 还有两万次级别的 elementwise kernel

 

这个比“模型有 0.3B 还是 0.5B trainable”更值得担心。

你的 GPU 很可能不是一直在做高效的大 GEMM，而是在不停：

```text
launch small kernel
copy
cat
layout conversion
norm
mul
add
launch next kernel
```

也就是说 **模型 graph 很碎**。

对于 RTX PRO 6000 Blackwell 这种卡，这是很亏的。

第三个信号尤其可疑：你明明是 **Blackwell RTX PRO 6000**，profile 却大量出现：

```text
sm80_xmma...
cutlass_80_tensorop...
fmha_cutlass...sm80...
```

 

这个我不会直接断定为 bug，因为 profiler kernel name 不一定严格代表“你的 GPU 被当成 SM80 跑”。

但是它**非常值得排查**。

你现在第一件事应该跑：

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())
print(torch.cuda.get_arch_list())
```

然后再查：

```bash
pip show flash-attn
pip show xformers
nvidia-smi
```

因为如果你的 PyTorch / FlashAttention / xFormers 实际没有针对 Blackwell 正确编译，导致 fallback 到兼容 kernel，那么训练速度完全可能大幅低于这张卡应有水平。

第四，你现在最应该做的不是继续试 batch size，而是做一个 **shape + module timing profile**。

我建议直接给每个大模块打 NVTX / CUDA event：

```text
event_encoder
event_decoder
context_encoder
joint_transformer
action_branch
event_branch
loss
backward
optimizer
```

同时打印一次 shape：

```text
event input:
event encoded:
event tokens:
context tokens:
action tokens:
joint tokens:
FFN input:
output:
```

尤其是下面这个数字：

$$
\boxed{\text{joint sequence length}}
$$

我现在最想知道。

因为你这两个事实：

```text
0.7B total
0.3B trainable
```

和：

```text
BS32 no checkpoint > 93 GB
BS64 checkpoint ≈ 9.8 s/step
```

放在一起看，**高度提示问题不是 parameter count，而是中间 representation 太大**。

---

你现在这个 profile，我会把问题优先级重新排成：

| 优先级   | 怀疑点                                               |
| ----- | ------------------------------------------------- |
| **1** | joint/event token 数是不是异常大                         |
| **2** | Blackwell 是否真正使用了合适的 CUDA / FlashAttention kernel |
| **3** | event codec 的 3D Conv 是否设计得过重                     |
| **4** | `copy/cat/layout conversion` 为什么这么多               |
| 5     | gradient checkpointing 重算                         |
| 6     | 参数量                                               |
| 7     | dataloader                                        |

所以你的直觉是对的：**这个速度不能简单用“0.7B 模型所以正常”解释。**

尤其是 **BS32 无 checkpoint 需要 >93GB activation + 大量 sm80/copy/cat kernel**，这两点我认为已经足够值得系统排查。

如果让我选一个下一步，我不会再调 batch，而是让代码输出 **每个模块的 tensor shape、token count、activation size，以及各模块 forward/backward 时间**。基本一轮就能判断到底是“序列太长”，还是“codec 太重”，还是“Blackwell kernel 没吃满”。

建议下一次直接把每个 stage 的 shape、元素数和理论 tensor size 打出来：

event input
↓
codec stage 1
↓
codec stage 2
↓
codec bottleneck
↓
flatten / projection
↓
event tokens

DINO tokens
T5 tokens
action tokens

↓ concat

joint sequence
↓
block attention input
↓
Q/K/V
↓
attention output
↓
FFN expanded activation
↓
event branch
↓
decoder stage 1/2/3

尤其需要记录：

$$ B,\quad T,\quad H,\quad W,\quad C,\quad N_{\text{tokens}} $$

然后计算：

$$ \text{tensor memory} = \prod_i d_i \times \text{bytes per element} $$

比如 BF16 一个 [64, 8, 64, 64, 768] tensor：

$$ 64\times8\times64\times64\times768\times2 \approx 3.2\text{ GB} $$

一个中间 tensor 就 3 GB。

如果 codec 中同时保留几个这样的 activation，再经过：

Conv3D
GroupNorm
SiLU
residual
projection
decoder skip connection

几十 GB 很容易就出来了。

这也是为什么“0.3B trainable”在这里非常有欺骗性。

特别要查 event codec

从你之前 profiler 看，我现在越来越怀疑 codec 的设计才是隐藏的大头之一。

如果你的 event tensor 是类似：

$$ [B,C,T,H,W] $$

并且前几层仍然保持比较高的空间分辨率，那么 Conv3D activation 的大小主要取决于：

$$ B\times C\times T\times H\times W $$

跟模型参数量关系很小。

这也能解释：

codec checkpoint 一关 → 直接从 41 GB → 93 GB。

所以一个很重要的问题是：

你是不是在非常高的空间分辨率上做了太多层 3D convolution？

如果是，我宁愿优化 codec，例如更早：

spatial downsample
temporal downsample / patchify
channel projection

而不是继续折腾 checkpoint。

Joint Transformer 同样需要看 token 数

另一个关键数字就是：

$$ \boxed{N_{\text{joint tokens}}} $$

如果 event codec 出来以后，例如有：

8 future × 16 × 16 spatial
= 2048 event tokens

再乘两个 camera 或多个 branch，很容易变成：

4000~8000 tokens

此时 D=768、L=12 已经完全不是“小 Transformer”了。

虽然只有几亿参数，但每一层都处理几千 token：

$$ \text{FFN cost}\sim O(ND^2) $$

Attention：

$$ O(N^2D) $$

activation memory 同样主要和：

$$ B\times N\times D $$

有关。

所以 0.7B 参数和 9.4 s/step 本身并不矛盾；真正异常的可能是你让这 0.7B 模型处理了一个极大的 spatiotemporal token volume。

有一个实验比“每隔两层 checkpoint”更有价值

直接跑：

batch = 1
gradient checkpointing = OFF

只跑一个 forward/backward，然后记录：

peak memory
codec 每 stage shape
joint token 数
各模块时间

因为 batch=1 可以把“batch 放大效应”去掉。

例如如果 BS1 no-ckpt 都已经：

10~20 GB

那就基本坐实 architecture activation 很重。

反过来如果 BS1 只有：

3~4 GB

而 BS32 直接 93 GB，那么就是基本线性 batch amplification，说明单样本 token volume 本身虽然大，但没有明显 memory leak。

这个实验诊断价值非常高。

当前训练吞吐其实也有一个明确基准

现在：

$$ \frac{64}{9.4}\approx6.81\text{ samples/s} $$

所以后续任何“降低 batch + 少 checkpoint”的方案，都应该超过 6.81 samples/s 才有意义。

例如 BS32：

$$ 32/t > 6.81 $$

要求：

$$ t<4.70\text{ s} $$

也就是说：

BS32 + 部分 checkpoint 必须低于约 4.7 s/step，才真正比现在 BS64 + full checkpoint 快。

这个门槛其实挺高。

所以我现在不太建议为了 checkpoint 做大量实验。

最终我会把判断改成：

你现在的 9.4 s/step 很可能不是 PyTorch 莫名其妙慢，而是 EWM 的计算图确实比“0.7B 参数”看上去重得多。

尤其这次实验已经证明：

codec activation 和 joint-block activation 都异常大。

下一步最值得做的是把 codec 每层 feature shape + joint token 数 + 每阶段 activation MB 全打印出来。只要拿到这张表，我们基本就能立即看出到底是哪一级把 0.7B EWM 变成了一个 96 GB 显存都很吃力的模型。


能，而且我认为 **EventVoxelCodec 是现在最适合轻量化的模块之一**。尤其你已经固定 `bin=1`，当前这套 3D encoder/decoder 明显偏重。

现在每个 `_ResDown3d` 都是：

```text
GroupNorm
3×3×3 Conv
GroupNorm
3×3×3 Conv
stride Conv3D
+
1×1×1 skip Conv
AvgPool3D
```

一共做 4 级；decoder 又是 4 级 `ConvTranspose3D + 两个 3×3×3 Conv`。 整个 codec 的 encoder/decoder 结构也确实如此。

但 `bin=1` 时 temporal stride 全部等于 1：

$$
T=1
$$

所以这些 `3×3×3 Conv` 在 temporal 维实际上没有多少建模意义。

## 第一刀：bin=1 直接改成 2D codec

这是我最推荐先做的，几乎不改变你的模型逻辑。

现在：

$$
[B,2,1,H,W]
$$

直接 squeeze：

$$
[B,2,H,W]
$$

然后全部换成：

```text
Conv3D → Conv2D
ConvTranspose3D → ConvTranspose2D / Upsample+Conv2D
AvgPool3D → AvgPool2D
```

最后 decode 再恢复：

$$
[B,2,H,W]\rightarrow[B,2,1,H,W].
$$

单个主 residual convolution 从：

$$
3\times3\times3=27
$$

kernel elements 变成：

$$
3\times3=9
$$

理论 MAC 直接约降：

$$
3\times.
$$

当然整个 codec 不会快 3 倍，因为 stride conv、norm 等还在，但你 profiler 里卷积相关已经是显著大头，所以整体训练速度有机会得到明显收益。此前 `convolution_backward` 和 `cudnn_convolution` 就已经占了很大比例。

---

## 第二刀：不要每一级都两个 ResNet Conv

当前四层实际上非常“CNN-heavy”。

我更推荐：

```text
Input 2×256×256

Conv 5×5, stride 2
→ C=32, 128×128

Conv 3×3, stride 2
→ C=64, 64×64

ResBlock ×1
→ C=128

Conv 3×3, stride 2
→ C=256, 32×32

Conv 3×3, stride 2
→ D=768, 16×16
```

也就是说从当前：

```text
4 stages
×
(2 residual conv + down conv + skip conv)
```

变成：

```text
4 stride conv
+
1~2 lightweight residual blocks
```

我甚至不认为 event codec 需要很深。

因为真正负责建模：

* temporal relationship；
* future dynamics；
* action interaction；
* semantic conditioning；

的是后面的 12 层 joint Transformer。

codec 的任务更接近：

> 把稀疏 event image 映射成稳定的 spatial tokens。

不需要自己再成为一个大型 vision backbone。

---

## 第三刀：最后不要用昂贵的 3×3 Conv 把通道推到 768

你现在第四级：

```python
_ResDown3d(c3, self.dim, ...)
```

其中 `self.dim=768`。

这意味着高通道阶段会存在：

```text
256 → 768
768 → 768
```

一类昂贵空间卷积。

这是很亏的。

更合理的方式是：

```text
空间 CNN：
2 → 32 → 64 → 128 → 256
              ↓
           16×16
              ↓
1×1 projection:
256 → 768
```

也就是：

$$
\boxed{\text{spatial processing 用低 channel}}
$$

直到最后才：

$$
C=256\rightarrow D=768
$$

用一个 `1×1 Conv` 或 Linear 投影。

这个改动通常会非常明显降低 codec FLOPs。

例如一个 3×3：

$$
768\rightarrow768
$$

的卷积参数/计算核心项是：

$$
9\times768^2
$$

而 1×1：

$$
256\rightarrow768
$$

只有：

$$
256\times768.
$$

量级差距非常大。

---

# 我最推荐的 Light Event Codec

如果暂时不改变 `16×16 token grid`，我会直接做：

```text
Input:
[B, 2, 256, 256]

↓ Conv2d 5×5, s=2
32 × 128 × 128

↓ SiLU
↓ Conv2d 3×3, s=2
64 × 64 × 64

↓ lightweight ResBlock
128 × 64 × 64

↓ Conv2d 3×3, s=2
192 × 32 × 32

↓ SiLU
↓ Conv2d 3×3, s=2
256 × 16 × 16

↓ 1×1 Conv
768 × 16 × 16

↓ flatten

256 × 768 tokens
```

decoder：

```text
768 × 16 ×16
↓ 1×1 Conv
256 × 16 ×16

↓ interpolate 2× + 3×3 Conv
128 × 32 ×32

↓ interpolate 2× + 3×3 Conv
64 × 64 ×64

↓ interpolate 2× + 3×3 Conv
32 ×128 ×128

↓ interpolate 2×
↓ 3×3 Conv
2 ×256 ×256
```

我会优先使用：

```text
bilinear/nearest upsample + Conv2D
```

而不是 `ConvTranspose3D`。

这样通常：

* kernel 更简单；
* activation 更少；
* 更容易被 compiler fuse；
* 避免 transpose convolution 的额外开销。

---

## 甚至还可以更激进：Wan-style patch embedding

如果你的目标主要是**把 event 变成 token**，还可以进一步：

```text
[B,2,256,256]
↓
Conv2D(kernel=4, stride=4)
C=64, 64×64
↓
Conv2D(kernel=4, stride=4)
C=256, 16×16
↓
1×1 projection
768×16×16
```

encoder 只剩 **3 个卷积**。

甚至：

```text
Conv2D(kernel=16, stride=16)
2 → 768
```

就直接得到 16×16 patch tokens。

这就是非常接近 ViT/Wan patchification 的思路。

不过我暂时不会一步走这么激进，因为 event 很稀疏，直接 16×16 patchify 可能损失局部结构。

比较稳的是：

$$
4\times\text{ downsample}
\rightarrow
4\times\text{ downsample}
$$

中间留一点非线性。

---

## Decoder 其实比 Encoder 更值得砍

你现在训练每一步都：

$$
\text{Transformer tokens}
\rightarrow
\text{完整 decoder}
\rightarrow
256\times256\text{ voxel}
\rightarrow
MSE
$$



这非常贵。

所以我会优先让 decoder 极简：

```text
Linear/1×1 projection
+
4次简单 upsample
+
少量 Conv2D
```

不要再搞四个完整 ResUp blocks。

因为最终 event 本身又非常稀疏，未必需要一个很强的生成 decoder。

更进一步，未来如果按照 Wan 的方式改成 latent flow：

$$
\text{loss直接算latent}
$$

那 decoder 训练时都可以完全不跑。

---

## 可以分三个版本做 ablation

| 版本          | Encoder                    | Decoder              | 风险  | 推荐          |
| ----------- | -------------------------- | -------------------- | --- | ----------- |
| Current     | 4× 3D ResDown              | 4× 3D ResUp          | 已知慢 | baseline    |
| **Lite-v1** | **同结构换 2D**                | **同结构换 2D**          | 很低  | **马上做**     |
| **Lite-v2** | 4 stride Conv + 1 ResBlock | Upsample+Conv        | 低～中 | **最推荐最终版本** |
| Lite-v3     | patchify encoder           | very shallow decoder | 中   | 后续          |

我会先做 **Lite-v1**，因为这是几乎纯工程优化：

$$
\text{Conv3D}(T=1)\rightarrow\text{Conv2D}
$$

模型语义基本完全没变。

如果它验证：

```text
loss正常
重建正常
速度明显提升
```

再直接做 Lite-v2。

### 最终我倾向的架构其实很简单

```text
Event
 ↓
轻量 2D encoder
 ↓
低通道 spatial feature
 ↓
1×1 → D=768
 ↓
EWM Transformer
 ↓
1×1 → low channel
 ↓
轻量 2D decoder
 ↓
Event velocity
```

而不是现在这种：

> **一个 12 层 Transformer 前后各挂一个完整的 3D ResNet codec。**

在 `bin=1` 的设定下，我认为当前 codec 明显 over-engineered。最优先值得做的就是 **3D→2D + 高通道 spatial conv→低通道 CNN 后再 1×1 投影**。这两个改动都不需要改变你的核心 EWM 方法。

是，这个提升已经很明显，而且这次和 8×8 token reduction 的性质不同：**8×8 主要是在砍主干 token 成本；Wan-lite 3D 则是在不进一步减少 token 的情况下，直接把 codec 本身做轻了。**

从 4.995 s/step 到 3.865 s/step，相当于单独靠 codec 重构又拿到 **22.6% step-time 降低 / 29.2% 吞吐提升**。这说明之前 profiler 里看到的卷积瓶颈确实不是虚的——原始 4-stage residual 3D codec 明显过重。

而且这个结果有一个很好的信号：显存只从 14.66 GiB 降到 14.38 GiB，变化很小，但速度提升很大。这说明 Wan-lite 的主要收益不是“少存 activation”，而是：

$$
\boxed{\text{实际 FLOPs / kernel 数 / convolution backward 成本下降}}
$$

这正符合你这次结构变化：

* 高分辨率阶段保持低 channel；
* 不再在空间卷积里大量使用 768 channel；
* 只在最后 `1×1×1` 做 `256→768`；
* residual block 数大幅减少；
* `ConvTranspose3D` 换成 nearest upsample + Conv3D。

尤其“**最后才升到 D=768**”应该是非常关键的一刀。因为原先如果在 16×16/8×8 附近做多个 768-channel 3D spatial conv，其 \(C^2\) 成本很高；现在大部分空间处理都发生在 32/64/128/256 channels。

### 现在累计结果已经很有说服力

从最初：

$$
9.368\text{s/step}
$$

到 8×8：

$$
4.995\text{s}
$$

再到 Wan-lite：

$$
3.865\text{s}
$$

总体加速：

$$
9.368/3.865\approx2.42\times
$$

吞吐从：

$$
6.83\rightarrow16.56\ \text{samples/s}.
$$

也就是说，目前已经把“一个看起来异常慢的 0.7B EWM”拉回了一个明显合理得多的区间。

我会把目前两个优化的贡献明确拆开：

| 改动                | step time |     相对收益 | 说明                       |
| ----------------- | --------: | -------: | ------------------------ |
| 原始 16×16 residual |   9.368 s | baseline | token + codec 都重         |
| 8×8 residual      |   4.995 s |    1.88× | **token geometry 是最大瓶颈** |
| 8×8 Wan-lite 3D   |   3.865 s |  再 1.29× | **codec 本身也明显过重**        |

这两个结果一起说明，你之前的问题不是某一个奇怪的 PyTorch bug，而是两个结构问题叠加：

$$
\boxed{\text{过多 event tokens}+\text{过重 codec}}
$$

现在都击中了。

### 下一步我同意先做短训质量，而不是继续疯狂优化

我会把 `wan_lite_3d + 8×8` 作为目前的 candidate architecture，先和原 baseline 做质量对照。

最少比较：

* event loss
* active/inactive MSE
* x0 active/inactive MSE
* precision / recall / F1
* action loss
* 几段 decoded future event 可视化

建议至少比较三个版本，而不是只比最老和最新：

```text
A. 16×16 residual codec
B. 8×8 residual codec
C. 8×8 wan_lite_3d
```

这样能区分：

* A→B：空间压缩造成的质量变化；
* B→C：codec 轻量化造成的质量变化。

如果 B 和 C 基本重合，那就说明 Wan-lite 几乎是“免费”优化。

### 现在还可以重新考虑 D=512

因为现在 token 和 codec 都压下来了，3.865 s/step 中 Transformer 的 QKV/FFN 占比会相对重新上升。

所以接下来 D768→512 的收益可能比之前更值得测试。

如果 D512 再能做到例如：

$$
3.865 \rightarrow 2.5\sim3.0\text{s/step}
$$

同时短训质量保持，那么最终：

$$
\boxed{8\times8,\ Wan\text{-lite 3D},\ D512,\ L12}
$$

会是一个非常合理的主模型候选。

不过顺序上还是：

$$
\boxed{\text{先确认 8×8 + Wan-lite 不伤质量，再降 D}}
$$

不要一次同时改 token、codec 和 embedding dim，否则最后 performance 掉了很难知道是哪一项导致的。

现在这个 **3.865 s/step** 的结果已经足够支持把 Wan-lite 3D 当作正式候选，而不是单纯的 profiling hack。
