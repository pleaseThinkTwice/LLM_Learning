# 第 8 章｜PEFT 与 LoRA 系列

> **本章定位**：LoRA 是最常用的参数高效微调方法。本章除了推导和工程细节（rank、α、初始化、QLoRA 的 NF4 与双重量化、合并与多 adapter 服务），重点回答一个研究问题：**LoRA 和全参微调真的等价吗？** 近两年的研究给出了相当一致的答案：在 SFT 的小数据场景和 RL 场景下，正确配置的 LoRA 可以接近全参微调；但在大量新知识的学习（如代码、数学的继续预训练）中，LoRA 明显"学得更少、也忘得更少"。

> **配套章节**：SFT → 第 7 章 / 继续预训练 → 第 6 章 / 量化 → 第 13 章 / 多 LoRA 推理服务 → 第 15 章

> **标记约定**：📄 论文结论 / 🧮 可推导 / 💭 直觉或有争议 / ❓ 开放问题

---

## 📌 本章导航

| 序号 | 题目 | 难度 | 类型 |
|---|---|---|---|
| Q1 | 为什么需要 PEFT？全参微调的代价 | ⭐⭐⭐ | 概念 |
| Q2 | PEFT 方法谱：Adapter / Prefix / Prompt / LoRA | ⭐⭐⭐⭐ | 对比 |
| Q3 | **LoRA 原理、初始化与推导** | ⭐⭐⭐⭐⭐ | 必问 |
| Q4 | **rank、α、学习率怎么选** | ⭐⭐⭐⭐⭐ | 工程 |
| Q5 | **QLoRA：NF4、双重量化、分页优化器** | ⭐⭐⭐⭐⭐ | 深度 |
| Q6 | LoRA 变体：AdaLoRA、DoRA、VeRA、LoRA+、PiSSA、GaLore | ⭐⭐⭐⭐ | 前沿 |
| Q7 | 合并、多 adapter 服务与模型融合 | ⭐⭐⭐ | 工程 |
| Q8 | MoE 模型的参数高效微调 | ⭐⭐⭐⭐ | 高级 |
| Q9 | **研究问题：LoRA 与全参微调等价吗？** | ⭐⭐⭐⭐⭐ | 研究视角 |

---

## Q1：为什么需要 PEFT？全参微调的代价 ⭐⭐⭐

### 🎯 一句话标答

> 全参微调需要为所有参数保存梯度和优化器状态（混合精度 Adam 下每参数约 16 字节，第 5 章），并且每个任务都要保存一份完整模型。PEFT 冻结原模型，只训练很少的参数，**显存和存储成本大幅下降**，同一个 base 模型可以搭配多个 adapter。代价是在需要学习大量新信息的任务上，**能力可能不如全参微调**（Q9）。

### 🧮 资源对比（7B 模型）

| | 全参微调（bf16 + Adam） | LoRA（r=16，所有线性层） |
|---|---|---|
| 可训练参数 | 约 7B | 约 4000 万（约 0.6%） |
| 模型状态显存 | 约 112 GB（16 字节/参数） | 约 14 GB（冻结的 bf16 权重）+ 很小的 LoRA 状态 |
| 每个任务的存储 | 约 14 GB | 约 80 MB |

🧮 LoRA 参数量估算：LLaMA-2 7B 每层有 4 个 $4096\times4096$ 的注意力投影和 3 个 $4096\times11008$ 的 MLP 矩阵。 $r=16$ 时，每个矩阵增加 $r(d_{in}+d_{out})$ 个参数：注意力部分 $4\times16\times8192$，MLP 部分 $3\times16\times15104$，每层约 117 万，32 层共约 3700 万。

💭 激活显存不会因为 LoRA 而减少（前向反向仍然要经过整个模型），所以长序列训练仍需要 gradient checkpointing 等手段。

> ❌ 旧版写"LoRA 训练速度 1.5–2 倍""性能是全参的 95–99%""差距通常小于 1%"，都没有出处。📄 LoRA 原论文报告的是在 GPT-3 175B 上训练吞吐提升约 25%；性能差距取决于任务（Q9）。

---

## Q2：PEFT 方法谱：Adapter / Prefix / Prompt / LoRA ⭐⭐⭐⭐

### 🎯 一句话标答

> **Adapter** 在层内插入小的瓶颈网络；**Prefix tuning** 在每层注意力的 K、V 前拼接可学习向量；**Prompt tuning** 只在输入 embedding 前加可学习的软提示；**LoRA** 给权重矩阵加一个低秩增量 $BA$。LoRA 训练后可以**合并回原权重**，推理零额外开销，因此最常用。

### 🔍 对比

| 方法 | 改动位置 | 推理开销 | 出处 |
|---|---|---|---|
| Adapter | 子层后插入降维–非线性–升维模块 | 增加串行计算 | 📄 Houlsby et al. 2019 |
| Prefix tuning | 每层的 K、V 前拼接可学习的前缀 | 占用上下文长度 | 📄 Li & Liang 2021 |
| Prompt tuning | 输入层的软提示 | 占用少量上下文 | 📄 Lester et al. 2021 |
| P-Tuning v2 | 每层都加可学习前缀（深层提示） | 占用上下文长度 | 📄 Liu et al. 2022 |
| LoRA | 权重矩阵加低秩增量 | 合并后为零 | 📄 Hu et al. 2021 |
| (IA)³ | 用可学习向量逐元素缩放 K、V 和 FFN 激活 | 可合并 | 📄 Liu et al. 2022（T-Few） |

### 🔥 高频追问

**Q：LoRA 比 Adapter 好在哪？**

A：📄 LoRA 论文测量了 Adapter 在**小 batch、短序列**推理时会带来明显的延迟增加（因为 Adapter 是额外的串行层，GPU 无法通过并行来摊销）；而 LoRA 合并后与原模型结构完全相同。此外，LoRA 不改变网络结构，实现和部署都更简单。

**Q：Prompt tuning 还有用吗？**

A：📄 Lester et al. 2021 发现，随着模型规模增长（在 T5 上到约 10B 参数），prompt tuning 与全参微调的差距逐渐消失；小模型上差距明显。📄 P-Tuning v2 通过在每一层都加入可学习前缀，在不同规模上都能接近全参微调。实践中 LoRA 更通用，prompt 类方法较少使用。

### 📚 出处
- Houlsby et al. *Parameter-Efficient Transfer Learning for NLP*. ICML 2019. arXiv:1902.00751
- Li & Liang. *Prefix-Tuning*. ACL 2021. arXiv:2101.00190
- Lester et al. *The Power of Scale for Parameter-Efficient Prompt Tuning*. EMNLP 2021. arXiv:2104.08691
- Liu et al. *P-Tuning v2*. ACL 2022. arXiv:2110.07602
- Liu et al. *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning*（(IA)³）. NeurIPS 2022. arXiv:2205.05638

---

## Q3：LoRA 原理、初始化与推导 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 📄 LoRA 冻结预训练权重 $W_0\in\mathbb{R}^{d_{out}\times d_{in}}$，把更新参数化为 $\Delta W = BA$， $B\in\mathbb{R}^{d_{out}\times r}$， $A\in\mathbb{R}^{r\times d_{in}}$， $r\ll\min(d_{in},d_{out})$。前向为 $h = W_0x + \frac{\alpha}{r}BAx$。**$A$ 随机初始化、 $B$ 初始化为 0**，因此训练开始时 $\Delta W=0$，模型与原模型完全相同。

### 🧮 推导与细节

**前向**：

$$h = W_0x + \frac{\alpha}{r}BAx$$

**参数量**：每个矩阵从 $d_{out}d_{in}$ 降到 $r(d_{in}+d_{out})$。 $d_{in}=d_{out}=4096$、 $r=8$ 时，从约 1678 万降到 65536，约为 0.39%。

**梯度**：令 $g = \partial\mathcal{L}/\partial h$（对单个样本），则

$$\frac{\partial\mathcal{L}}{\partial B} = \frac{\alpha}{r}\thinspace g\thinspace (Ax)^T,\qquad \frac{\partial\mathcal{L}}{\partial A} = \frac{\alpha}{r}\thinspace B^Tg\thinspace x^T$$

🧮 由于 $B=0$，第一步时 $\partial\mathcal{L}/\partial A = 0$，只有 $B$ 会被更新；之后两者才一起变化。

**合并**：训练后 $W' = W_0 + \frac{\alpha}{r}BA$，推理与原模型结构完全一致。

### 📄 原论文的关键实验与分析

- **加在哪些矩阵上**：📄 在 GPT-3 175B、固定可训练参数预算的实验中，同时调整 $W_q$ 和 $W_v$（每个 $r=4$）或同时调整 $W_q, W_k, W_v, W_o$（每个 $r=2$），比只调整单个矩阵（ $r=8$）效果更好。结论是"用较小的 rank 覆盖更多矩阵"比"在少数矩阵上用大 rank"更好。
- **秩真的很低吗**：📄 他们比较了 $r=8$ 和 $r=64$ 学到的子空间，发现最重要的几个方向高度重合，说明 $\Delta W$ 的"有效秩"很低。
- **$\Delta W$ 学到了什么**：📄 分析发现 $\Delta W$ 与 $W_0$ 有一定相关性，但它**放大的是 $W_0$ 中没有被强调的方向**，放大倍数很大（ $r=4$ 时约为 20 倍左右）。作者据此推测，LoRA 放大了预训练中已学到、但不突出的任务相关特征。
- 📄 理论背景：Aghajanyan et al. 2021（*Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*）发现，预训练模型微调时只需要在一个很低维的随机子空间中优化就能达到接近全参微调的效果，并且模型越大、预训练越充分，所需的"内在维度"越低。

> ❌ 旧版写"对 ΔW 做 SVD，前 r 个奇异值解释 >80% 的方差""LLaMA 系实践通常加 QKVO、不加 FFN"，没有出处。📄 QLoRA 论文的结论恰恰是：**要达到全参微调的效果，需要把 LoRA 加到所有线性层上**（包括 MLP）。

### 💻 实现

```python
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # 与 HF PEFT 默认一致
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * (self.drop(x) @ self.A.T @ self.B.T)

    @torch.no_grad()
    def merge(self):
        self.base.weight += (self.scale * self.B @ self.A).to(self.base.weight.dtype)
```

### 🔥 高频追问

**Q：为什么是 $B=0$、 $A$ 随机，而不是反过来？**

A：两种方式都能保证初始 $\Delta W=0$。📄 Hayou et al. 2024（*The Impact of Initialization on LoRA Finetuning Dynamics*）从理论和实验上比较了两者：**$B=0$、 $A$ 随机**的初始化允许使用更大的学习率而不发生不稳定，从而学习得更高效； $A=0$ 的初始化在相同学习率下更新幅度不足。这为默认做法提供了理论依据。

**Q：两个都随机初始化不行吗？**

A：🧮 那样初始时 $\Delta W = BA \neq 0$，模型一开始就偏离了预训练模型，相当于注入了随机噪声。

### 📚 出处
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. arXiv:2106.09685
- Aghajanyan et al. ACL 2021. arXiv:2012.13255
- Hayou et al. *The Impact of Initialization on LoRA Finetuning Dynamics*. NeurIPS 2024. arXiv:2406.08447

---

## Q4：rank、α、学习率怎么选 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> $r$ 决定增量的容量， $\alpha/r$ 决定增量的缩放。常见起点： $r=16$、 $\alpha=16$ 或 $32$，**LoRA 加到所有线性层**。📄 LoRA 的最优学习率通常比全参微调**高约一个数量级**。 $\alpha/r$ 的缩放使最优学习率对 $r$ 不太敏感；📄 rsLoRA 指出，在 rank 很大时应改用 $\alpha/\sqrt r$ 的缩放。

### 🧮 为什么要除以 $r$？

📄 LoRA 论文的说法是：用 $\alpha/r$ 缩放后，改变 $r$ 时不需要重新调学习率（在 Adam 下， $\alpha$ 的作用近似于调整学习率）。

📄 **rsLoRA**（Kalajdzievski 2023, *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*）：从初始化时前向和反向信号的尺度出发，证明当 $r$ 增大时， $\alpha/r$ 会使 LoRA 的更新随 $r$ 增大而**过小**，导致大 rank 时学习变慢、提升有限；应使用 $\alpha/\sqrt r$，此时增大 rank 能带来持续的性能提升。HF PEFT 中可通过 `use_rslora=True` 开启。

### 📄 学习率

- 📄 Biderman et al. 2024 的扫描显示，LoRA 的最优学习率比全参微调高约一个数量级。
- 📄 Thinking Machines（Schulman 等, 2025 博客 *LoRA Without Regret*）在 SFT 和 RL 上做了大规模扫描，得出的经验结论包括：LoRA 的最优学习率约为全参微调的 10 倍；由于 $1/r$ 缩放，最优学习率**几乎不依赖于 rank**；LoRA 对大 batch 的容忍度不如全参微调。
- 📄 **LoRA+**（Hayou et al., ICML 2024）：从宽度无穷大极限的分析出发，指出 $A$ 和 $B$ 应使用不同的学习率，建议 $\eta_B = \lambda\eta_A$， $\lambda$ 取 16 左右，能提升效果和收敛速度。

### 🔍 经验起点

| 场景 | 建议 |
|---|---|
| 指令遵循、风格调整、小数据 SFT | $r=8$– $32$，所有线性层，学习率约为全参的 10 倍 |
| 大量新知识（代码、数学、新语言） | 考虑全参微调或继续预训练；如果用 LoRA，需要更大的 rank，并接受能力上限（Q9） |
| RL（GRPO 等） | 📄 很小的 rank 就可能足够（Q9） |

> ❌ 旧版的"r=16, α=32, lr=3e-4"等表格、"α/r ≤ 2–4 是合理范围"等经验数字没有出处。上面给出的是有出处的结论；具体数值仍应在自己的任务上做扫描。

### 📚 出处
- Kalajdzievski. *rsLoRA*. 2023. arXiv:2312.03732
- Hayou et al. *LoRA+*. ICML 2024. arXiv:2402.12354
- Biderman et al. *LoRA Learns Less and Forgets Less*. TMLR 2024. arXiv:2405.09673
- Schulman et al. *LoRA Without Regret*. Thinking Machines Lab: Connectionism, 2025（博客）

---

## Q5：QLoRA：NF4、双重量化、分页优化器 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 📄 QLoRA（Dettmers et al., NeurIPS 2023）把冻结的 base 模型量化为 **4-bit NormalFloat（NF4）**，在其上训练 bf16 的 LoRA，反向传播穿过量化权重（计算时反量化到 bf16）。三个关键技术：**NF4**、**双重量化**（对量化常数再量化）、**分页优化器**（利用统一内存应对显存峰值）。论文表明可以在单张 48GB GPU 上微调 65B 模型，并且 **NF4 + 双重量化能完全恢复 16-bit LoRA 的效果**。

### 🧮 NF4

**思路**：如果权重近似服从零均值正态分布，那么把正态分布的**分位点**作为量化点，可以让每个量化区间分到大致相同数量的权重。

**做法**：
1. 按 block（默认 64 个元素）做 absmax 缩放，把权重归一化到 $[-1, 1]$。
2. 用标准正态分布的分位点构造 16 个量化值，并保证**0 被精确表示**（正半轴 8 个、负半轴 7 个、加上 0）。
3. 📄 得到的 16 个值（bitsandbytes 实现）：
```
[-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
  0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
```

📄 论文称 NF4 对正态分布的数据是"信息论上最优"的数据类型。💭 这一说法有争议：后续有分析指出，在 block absmax 归一化之后，块内数据的分布已不再是标准正态，严格意义上的最优性不成立。

### 🧮 双重量化

- 一级量化：每 64 个权重一个 fp32 缩放常数，额外开销 $32/64 = 0.5$ bit/参数。
- 二级量化：把这些常数用 FP8 量化，每 256 个常数一个 fp32 的二级缩放常数，额外开销 $8/64 + 32/(64\times256) \approx 0.127$ bit/参数。
- 🧮 每参数节省约 0.373 bit。📄 论文报告对 65B 模型约节省 3 GB。

### 📄 分页优化器

使用 NVIDIA 统一内存，在 GPU 显存不足时，**自动把优化器状态分页到 CPU 内存**，需要时再换回来，以处理长序列 mini-batch 等造成的显存峰值，避免 OOM。

### 📄 实验结论

- NF4 + 双重量化的 QLoRA 能完全恢复 16-bit LoRA 在 MMLU 上的表现；用 FP4 的 QLoRA 比 16-bit LoRA 低约 1 个百分点。
- 在评测设置规范的学术基准上，4-bit QLoRA 与 16-bit 全参微调、16-bit LoRA 的表现相当。
- 要达到全参微调的效果，需要把 LoRA 加到所有线性层上。
- 他们用 QLoRA 训练的 Guanaco 系列模型在 Vicuna 基准上表现很强。

> ❌ 旧版的 MMLU 对比表（全参 63.9 / LoRA 63.4 / QLoRA 63.5 / INT4 61.8）和"4×3090 能微调 70B""性能损失约 1–2%"没有出处。论文中对比的是 **FP4** 而不是 INT4。

### 🔥 高频追问

**Q：QLoRA 训完怎么部署？**

A：常见做法是把 LoRA 合并到 bf16 权重中再部署。⚠️ 有一个细节：LoRA 是在**量化后的权重**上训练的，它学到的增量补偿了量化误差的一部分。如果合并到**原始的 bf16 权重**上，函数会和训练时略有不同；更严格的做法是合并到**反量化后的权重**上。合并后如果再重新量化（例如为了 4-bit 推理），还会引入新的量化误差。

**Q：QLoRA 能用于继续预训练吗？**

A：可以运行，但受 LoRA 本身"学得少"的限制（Q9）。需要注入大量新知识时，全参训练通常更好。

### 📚 出处
- Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023. arXiv:2305.14314
- Dettmers et al. *8-bit Optimizers via Block-wise Quantization*. ICLR 2022. arXiv:2110.02861

---

## Q6：LoRA 变体：AdaLoRA、DoRA、VeRA、LoRA+、PiSSA、GaLore ⭐⭐⭐⭐

### 🔍 主要变体

| 方法 | 核心思想 | 出处 |
|---|---|---|
| **AdaLoRA** | 用 $P\Lambda Q$ 的 SVD 形式参数化增量，根据重要性得分**剪掉不重要的奇异值**，在不同矩阵间自适应分配 rank 预算 | 📄 Zhang et al., ICLR 2023 |
| **DoRA** | 把权重分解为**幅度**和**方向**： $W' = m\cdot\frac{W_0+BA}{\lVert W_0+BA\rVert_c}$（按列归一化），幅度向量 $m$ 直接训练，方向用 LoRA 更新 | 📄 Liu et al., ICML 2024（NVIDIA） |
| **VeRA** | 所有层**共享一对冻结的随机矩阵** $A, B$，每层只训练两个缩放向量 $d, b$： $\Delta W = \Lambda_b B\Lambda_d A$；参数量比 LoRA 少一个数量级以上 | 📄 Kopiczko et al., ICLR 2024 |
| **LoRA+** | $B$ 的学习率设为 $A$ 的 $\lambda$ 倍（约 16） | 📄 Hayou et al., ICML 2024 |
| **rsLoRA** | 缩放因子改为 $\alpha/\sqrt r$ | 📄 Kalajdzievski 2023 |
| **PiSSA** | 用 $W_0$ 的**主奇异向量**初始化 $A, B$，冻结剩余部分，收敛更快 | 📄 Meng et al., NeurIPS 2024 |
| **LoRA-GA** | 用梯度的 SVD 初始化 $A, B$，使第一步更新逼近全参微调的梯度方向 | 📄 Wang et al., NeurIPS 2024 |
| **GaLore** | 不是加低秩增量，而是把**梯度投影到低秩子空间**来压缩优化器状态，仍然是全参训练 | 📄 Zhao et al., ICML 2024 |
| **LongLoRA** | 训练时使用分组移位的稀疏注意力（S²-Attn），并让 embedding 和 norm 层可训练，高效扩展上下文 | 📄 Chen et al., ICLR 2024 |

### 🔥 高频追问

**Q：DoRA 为什么有效？**

A：📄 DoRA 论文分析了全参微调和 LoRA 的权重更新：在全参微调中，幅度的变化和方向的变化呈**负相关**（方向变化大时幅度变化小），而 LoRA 的两者呈正相关，说明 LoRA 难以独立地调整两者。DoRA 通过解耦幅度和方向，让学习模式更接近全参微调，在多个任务上一致地优于 LoRA。

> ❌ 旧版写"DoRA 在 r=4 时约等于 LoRA r=16"，没有出处。

**Q：VeRA 只训练向量，为什么还能工作？**

A：💭 它依赖"随机投影"在高维空间中的表达能力：冻结的随机矩阵提供了一组固定的方向，每层只学习在这些方向上的缩放。这与 Aghajanyan et al. 2021 的"内在维度"结论相呼应：在随机子空间中优化就足够完成很多微调任务。

**Q：GaLore 和 LoRA 的本质区别？**

A：LoRA 限制的是**权重更新的形式**（低秩增量）；GaLore 限制的是**优化器看到的梯度**（每隔一段时间更新投影子空间），权重本身可以发生满秩的变化。因此 GaLore 可以用于预训练，并且对显存的节省主要来自优化器状态。

### 📚 出处
- Zhang et al. *AdaLoRA*. ICLR 2023. arXiv:2303.10512
- Liu et al. *DoRA*. ICML 2024. arXiv:2402.09353
- Kopiczko et al. *VeRA*. ICLR 2024. arXiv:2310.11454
- Meng et al. *PiSSA*. NeurIPS 2024. arXiv:2404.02948
- Wang et al. *LoRA-GA*. NeurIPS 2024. arXiv:2407.05000
- Zhao et al. *GaLore*. ICML 2024. arXiv:2403.03507
- Chen et al. *LongLoRA*. ICLR 2024. arXiv:2309.12307

---

## Q7：合并、多 adapter 服务与模型融合 ⭐⭐⭐

### 🎯 一句话标答

> 单任务部署时把 LoRA 合并进权重（ $W' = W_0 + \frac{\alpha}{r}BA$），推理零开销；多任务/多租户时保持 adapter 独立，由服务系统在同一 batch 中为不同请求应用不同的 adapter（📄 Punica、S-LoRA，vLLM 等推理引擎也已支持）。多个微调模型还可以通过**模型融合**合成一个模型。

### 📄 多 LoRA 服务

- 📄 **Punica**（Chen et al., MLSys 2024）：提出 SGMV（Segmented Gather Matrix-Vector multiplication）kernel，让同一个 batch 中使用不同 LoRA 的请求可以一起计算，base 模型的计算只做一次。
- 📄 **S-LoRA**（Sheng et al., MLSys 2024）：把所有 adapter 存在主存中，按需加载到 GPU；用**统一分页**同时管理 KV Cache 和 adapter 权重以减少碎片；配合定制的 kernel 和张量并行策略，可以同时服务**数千个** adapter。

> ⚠️ 旧版写"Base 权重 KV Cache 共享"，概念混淆：不同请求共享的是 **base 模型的权重和计算**，KV Cache 是每个请求自己的。

### 📄 模型融合

| 方法 | 做法 | 出处 |
|---|---|---|
| Task Arithmetic | 任务向量 $\tau = W_{ft} - W_0$，可以相加（多任务）或相减（遗忘某种行为） | 📄 Ilharco et al., ICLR 2023 |
| TIES-Merging | 修剪小幅度的更新、解决符号冲突、只合并符号一致的部分 | 📄 Yadav et al., NeurIPS 2023 |
| DARE | 随机丢弃大部分更新并对剩余部分重新缩放，再合并 | 📄 Yu et al., ICML 2024 |
| Model Soups | 对同一预训练模型、不同超参数微调出的权重取平均 | 📄 Wortsman et al., ICML 2022 |

### 🔥 高频追问

**Q：合并后还能继续微调吗？**

A：可以，把合并后的模型当作新的 base 即可（例如 SFT LoRA → 合并 → DPO LoRA）。💭 每次合并都会在 bf16 中引入舍入误差，链路过长时要注意。

---

## Q8：MoE 模型的参数高效微调 ⭐⭐⭐⭐

### 🎯 一句话标答

> MoE 的 FFN 由很多专家组成，PEFT 需要决定"调哪些专家、怎么调"。📄 DeepSeek 的 **ESFT**（Expert-Specialized Fine-Tuning）发现：同一任务的路由分布高度集中，不同任务激活的专家差异很大；因此**只微调与任务最相关的少数专家**、冻结其余专家和其他模块，就能匹配甚至超过全参微调，同时更高效、遗忘更少。

### 🔍 设计选择

| 方案 | 说明 |
|---|---|
| 只在注意力层加 LoRA | 最简单，但没有覆盖占大部分参数的专家 |
| 每个专家各自加 LoRA | 参数量随专家数线性增长；冷门专家见到的 token 少，训练不充分 |
| 📄 ESFT：只调任务相关的专家 | 按路由统计（平均门控分数或被选中的 token 比例）挑选专家，全参训练这些专家，其余冻结 |
| 微调路由器 | 改变专家选择；可能破坏负载均衡 |

📄 ESFT 论文还发现，**专家粒度越细**的 MoE 模型，越容易挑出与下游任务最相关的专家组合。

### 📄 反过来：用"LoRA 专家"缓解遗忘

- 📄 **LoRAMoE**（Dou et al., ACL 2024）：在 dense 模型中引入多个 LoRA 作为"专家"并用路由器组合，一部分专家负责保留世界知识，另一部分学习下游任务，以缓解大规模 SFT 中的知识遗忘。
- 📄 Zadouri et al. 2023（*Pushing Mixture of Experts to the Limit*）：用多个轻量的 PEFT 模块组成 MoE，以极少的参数接近全参微调。

> ❌ 旧版写"DeepSeek-V3 报告里详细讨论过 PEFT：Attention 加 LoRA、FFN 选择性加"，DeepSeek-V3 报告中没有这部分内容，已替换为 DeepSeek 的 ESFT 论文。

### 📚 出处
- Wang et al.（DeepSeek）. *Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models*. EMNLP 2024. arXiv:2407.01906
- Dou et al. *LoRAMoE*. ACL 2024. arXiv:2312.09979
- Zadouri et al. 2023. arXiv:2309.05444

---

## Q9：研究问题：LoRA 与全参微调等价吗？ ⭐⭐⭐⭐⭐

### 📄 证据一：LoRA 学得少，忘得少

📄 **Biderman et al., TMLR 2024**（*LoRA Learns Less and Forgets Less*）在代码和数学两个领域、继续预训练（约 200 亿 token）和指令微调（约 10 万条）两种设置下比较 LoRA 与全参微调：
- 在大多数设置中，**LoRA 的目标领域性能明显低于全参微调**。
- 但 LoRA 更好地**保留了 base 模型在目标领域之外的能力**，遗忘更少；它的正则化效果强于 weight decay 和 dropout 等常见方法。
- 全参微调学到的权重扰动的秩比常见 LoRA 配置的 rank **高 10–100 倍**，这可以部分解释两者的差距。
- 建议：LoRA 的学习率要比全参微调高，并且应加到所有层上。

### 📄 证据二：LoRA 的解在结构上不同

📄 **Shuttleworth et al., NeurIPS 2025**（*LoRA vs Full Fine-tuning: An Illusion of Equivalence*）：即使下游表现相同，LoRA 训练出的权重矩阵中会出现**"入侵维度"**：与预训练权重的所有奇异向量都几乎正交、却拥有很大奇异值的新方向；全参微调则没有。通过事后缩小这些维度的奇异值，可以显著改善对预训练分布的建模，而下游任务性能只轻微下降，说明入侵维度与遗忘有因果关系。在顺序微调多个任务时，积累了入侵维度的 LoRA 模型表现更差。更高 rank 并使用 rank 稳定缩放的 LoRA，与全参微调更接近。

### 📄 证据三：正确配置时，LoRA 可以匹配全参微调

📄 **Thinking Machines, *LoRA Without Regret*（2025）**：在 SFT（Tülu 3、OpenThoughts3 数据）和 RL（MATH、GSM8K 等）上用 Llama 3 和 Qwen3 做了大规模扫描，结论包括：
- 在中小规模的 SFT 数据上，把 LoRA 加到**所有层（尤其是 MLP 和 MoE 层）**，LoRA 能匹配全参微调；只加在注意力层上效果明显更差，即使参数量相同。
- 当数据量超过 LoRA 的容量时，LoRA 开始落后于全参微调。
- 在 **RL** 中，即使 **rank 为 1** 的 LoRA 也能匹配全参微调。作者的解释是：基于结果的 RL 每个 episode 只提供 $O(1)$ 比特的信息，需要的容量很小。
- 最优学习率约为全参微调的 10 倍；对大 batch 的容忍度不如全参微调。

### 💡 综合起来

| 场景 | LoRA 能否替代全参 |
|---|---|
| 风格、格式、指令遵循（小数据 SFT） | 能，前提是所有层都加、学习率调好 |
| RL（结果奖励） | 能，很小的 rank 就够 |
| 大量新知识（继续预训练、大规模领域 SFT） | 通常不能完全替代，但遗忘更少 |
| 顺序多任务 | 要注意入侵维度带来的遗忘积累 |

❓ **开放问题**：
- 能否在训练过程中检测并抑制入侵维度？
- "LoRA 容量"与数据量之间是否存在可预测的 scaling 关系？
- 为什么 RL 需要的容量如此之小？这对理解 RL 学到了什么（第 11 章）有什么启示？

### 🧪 动手验证思路

1. 在同一 SFT 数据上分别训练 LoRA（只加注意力 / 加所有线性层）和全参微调，在目标任务和通用基准上比较，验证"学得少、忘得少"。
2. 对训练好的 LoRA，计算 $W_0$ 和 $W_0+\Delta W$ 的 SVD，按 Shuttleworth et al. 的方法统计入侵维度。
3. 比较 $\alpha/r$ 与 $\alpha/\sqrt r$ 在 $r = 8, 64, 256$ 时的学习曲线。
4. 实现 NF4 量化，对一个真实模型的权重矩阵比较 NF4 与均匀 INT4、FP4 的量化误差。

### 📚 出处
- Biderman et al. TMLR 2024. arXiv:2405.09673
- Shuttleworth et al. *LoRA vs Full Fine-tuning: An Illusion of Equivalence*. NeurIPS 2025. arXiv:2410.21228
- Schulman et al. *LoRA Without Regret*. Thinking Machines Lab, 2025

---

## 📝 本章小结

| 关键点 | 你必须能脱口而出 | 性质 |
|---|---|---|
| LoRA 公式 | $h = W_0x + \frac{\alpha}{r}BAx$； $B=0$、 $A$ 随机 | 🧮 |
| 参数量 | $r(d_{in}+d_{out})$ | 🧮 |
| 初始化 | $B=0$ 允许更大学习率（Hayou 2024） | 📄 |
| 加在哪里 | 所有线性层（含 MLP）；小 rank 覆盖更多矩阵更好 | 📄 |
| 缩放 | $\alpha/r$；大 rank 用 $\alpha/\sqrt r$（rsLoRA） | 📄 |
| 学习率 | 约为全参的 10 倍；LoRA+： $\eta_B \approx 16\eta_A$ | 📄 |
| QLoRA | NF4（正态分位点 + 精确的 0）、双重量化省约 0.37 bit/参数、分页优化器；NF4 恢复 16-bit LoRA 效果 | 🧮 + 📄 |
| DoRA / VeRA | 幅度–方向解耦 / 共享冻结随机矩阵 + 缩放向量 | 📄 |
| 多 LoRA 服务 | Punica 的 SGMV、S-LoRA 的统一分页 | 📄 |
| MoE PEFT | ESFT：只调任务相关专家 | 📄 |
| 等价性 | 学得少忘得少；入侵维度；RL 中 rank 1 即可 | 📄 |

## ✅ 自测题

1. 推导 LoRA 中 $A$、 $B$ 的梯度，并说明为什么第一步只有 $B$ 会被更新。
2. 计算 LLaMA-2 7B 在 $r=16$、所有线性层加 LoRA 时的可训练参数量。
3. 写出 NF4 的构造方法，并计算双重量化每参数节省多少 bit。
4. 为什么 QLoRA 训练出的 adapter 最好合并到反量化后的权重上？
5. 用三篇研究回答"LoRA 能否替代全参微调"，并说明在什么场景下答案不同。
6. 你要为 100 个企业客户分别定制模型，给出训练和服务的完整方案。

---

> **下一章预告**：第 9 章｜RLHF 与 PPO——奖励模型与 Bradley-Terry、PPO 的推导（重要性采样、裁剪、GAE）、KL 约束、reward hacking 与过优化的 scaling law。
