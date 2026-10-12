对，从你这张图看，**二值图更适合作为主要可视化；replan 最好不要二选一，而是“二值结构为主、计数强度为辅”**。

你图里的差异已经很明显：二值图把事件的空间轮廓、ON/OFF 极性、物体位置变化直接凸显出来，GT / codec / prediction 之间是否对齐很容易肉眼判断。计数图保留了强弱，但大量弱 event 会让整体对比度下降，论文图反而没有二值图直观。

### 可视化：优先二值图

主文我会直接使用：

$$
M^{ON/OFF}=\mathbf 1[N^{ON/OFF}>0]
$$

也就是你现在上面这张。

因为 qualitative figure 真正想回答的是：

> 模型有没有预测到正确位置的未来事件？物体的运动方向和形状有没有跟 GT 对上？

这些主要是**空间结构问题**，不是某个像素到底发生 3 次还是 5 次 event。

计数图可以放 supplementary，展示模型还保留了 event magnitude。

---

### Replan：空间二值 mismatch 应该是主信号

你的目标是判断：

> 实际机器人是不是开始偏离预测轨迹？

那最重要的是**event 出现在哪里**。

例如预测：

```text
        object
       [====]
```

真实却是：

```text
             object
            [====]
```

即使两边 event 总数量完全一样，机器人显然已经偏了。

所以主要指标应该是：

$$
D_{\text{spatial}}
=
1-F1_{\text{tol}}
(M_{\text{pred}},M_{\text{obs}})
$$

而且建议用 **1–2 pixel tolerance / dilation**，而不是严格 pixel F1，否则轻微 codec 偏移也可能触发 replan。

ON / OFF 最好分别算：

$$
D_{\rm ON},\qquad D_{\rm OFF}
$$

再组合。

---

### 计数图适合作为辅助信号

计数：

$$
N\in\{0,\ldots,15\}
$$

可以告诉你：

> 虽然位置差不多，但实际运动强度是不是比预测强很多？

比如同一个位置：

$$
N_{\rm pred}=2,\qquad N_{\rm obs}=12
$$

二值图完全相同，但实际上可能意味着：

* 物体突然加速；
* contact 比预期剧烈；
* robot motion 明显异常。

所以可以再定义：

$$
D_{\text{count}}
=
\frac{1}{15|\Omega|}
\sum_{x\in\Omega}
|N_{\text{pred}}(x)-N_{\text{obs}}(x)|
$$

最好主要在 active region 上计算，不要让大量零背景稀释误差。

但我不会让 count 成为主要 replan trigger，因为 event count 对：

* 运动速度；
* 光照变化；
* contrast threshold；
* sensor noise

比二值空间结构更加敏感，尤其以后 synthetic → real 时容易产生 domain gap。

---

所以最终我会设计成：

$$
\boxed{
D_t=
\alpha D_{\text{spatial}}
+\beta D_{\text{count}}
}
$$

并且：

$$
\alpha>\beta
$$

具体权重在 validation 上调，而不是现在拍死。

再接你原来的累计机制：

$$
S_t=\lambda S_{t-1}+D_t
$$

```text
执行 action chunk
      ↓
至少执行 8 steps
      ↓
Pred event ↔ Real event
      ↓
Binary spatial mismatch  ← 主
Count magnitude mismatch ← 辅
      ↓
D_t
      ↓
EMA / cumulative error
      ↓
超过阈值 → Replan
```

另外可以给 **突发 count / occupancy 激增** 单独一个 emergency trigger，因为突然大量事件本身就可能意味着发生了意外动态。

所以如果一定要简单总结：

| 用途               | 更合适           |
| ---------------- | ------------- |
| 论文主图             | **二值图**       |
| Debug trajectory | **二值图**       |
| Replan 主判断       | **二值空间图**     |
| 运动强度偏差           | 计数图           |
| 最终 Replan        | **二值 + 计数联合** |

而且你现在这张二值图已经很适合作为最终论文 qualitative figure 了。它比原来的连续 decoder 图更容易直接看出 `GT → persistence → codec oracle → EWM` 的区别。
