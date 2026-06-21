# 第 8 章｜PEFT 与 LoRA 系列

> **本章定位**：LoRA 是 LLM 微调的"事实标准"，**完整推导是必考题**。QLoRA 的 NF4 + Double Quantization + Paged Optimizer 是字节、DeepSeek 等岗位的细节题。本章重点：LoRA 数学推导 + 超参选择 + QLoRA 工程。

> **配套章节**：SFT → 第 7 章 / 量化 → 第 5 部分推理优化

---

## 📌 本章导航

| 序号 | 题目 | 难度 | 频率（字节/阿里/Meta/DeepSeek） | 类型 |
|---|---|---|---|---|
| Q1 | 为什么需要 PEFT？全参微调的代价 | ⭐⭐⭐ | 4 / 3 / 2 / 2 | 概念 |
| Q2 | PEFT 方法谱：Adapter/Prefix/Prompt/LoRA | ⭐⭐⭐⭐ | 5 / 3 / 3 / 3 | 对比 |
| Q3 | **LoRA 完整原理与推导** | ⭐⭐⭐⭐⭐ | **10 / 6 / 5 / 6** | 必问 |
| Q4 | **LoRA 的 rank r 和 alpha 怎么选** | ⭐⭐⭐⭐⭐ | **7 / 4 / 4 / 5** | 工程 |
| Q5 | **QLoRA 完整解析** | ⭐⭐⭐⭐⭐ | **8 / 5 / 4 / 6** | 深度 |
| Q6 | DoRA / AdaLoRA / VeRA 等改进 | ⭐⭐⭐ | 3 / 2 / 2 / 2 | 前沿 |
| Q7 | LoRA 的合并部署与多 adapter | ⭐⭐⭐ | 4 / 2 / 2 / 3 | 工程 |
| Q8 | LoRA 在 MoE 上的特殊问题 | ⭐⭐⭐ | 3 / 1 / 2 / 4 | 高级 |

---

## Q1：为什么需要 PEFT？全参微调的代价 ⭐⭐⭐

### 🎯 一句话标答

> 全参微调要更新所有参数，**显存爆炸 + 存储多份模型 + 训练慢**——对 70B 模型代价巨大。PEFT 只调少量参数（通常 < 1%），**显存省、训得快、能跨任务共享 base model**。

### 🗣️ 30 秒口语版

"全参微调（Full Fine-Tuning）的代价主要在三个维度——

**1. 显存爆炸**：训 7B 模型需要至少 80GB（Adam 优化器），70B 需要约 800GB——单卡放不下，需要复杂的并行。

**2. 存储成本**：每个微调任务都要存一份**完整模型副本**——70B 模型 140GB（BF16），10 个任务 = 1.4TB。

**3. 训练成本**：全参更新慢，每步反向都要计算所有参数梯度。

**PEFT（Parameter-Efficient Fine-Tuning）**的核心思想——**冻结大部分参数，只训练少量新增参数**。优势——

- **显存省**：只需要训练那部分参数的梯度和优化器状态。LoRA 训 7B 模型可能只要 16GB
- **存储省**：每个任务只存几 MB 的 LoRA 权重，base model 共享
- **训得快**：可学习参数少，反向传播快
- **多任务复用**：同一个 base + 多个 LoRA = 任务切换只需要换 adapter

**代价**：表达力上限略低于全参微调。但在大多数场景下差距很小（< 1%），所以工业上 PEFT 是主流。"

### 🔍 全参 vs PEFT 资源对比

以训 7B 模型为例：

| 维度 | 全参微调 | LoRA |
|---|---|---|
| 可训练参数 | 7B (100%) | ~10M (0.1%) |
| 显存（BF16+Adam） | ~112 GB | ~16 GB |
| 单卡可训（80GB H100） | 不够 | 富余 |
| 训练速度 | 1.0× | 1.5-2× |
| 存储 per task | 14 GB | 30 MB |
| 性能（vs 全参） | 100% | 95-99% |

### 🔥 高频追问 Top 3

**Q：什么场景下应该用全参，什么场景用 PEFT？**

A：**经验法则**——
- **新领域 / 新语言**（base model 完全没见过）：全参微调，需要大幅调整表示
- **新任务 / 新格式**（base model 见过相关数据）：PEFT 即可
- **资源充足，追求极致**：全参微调
- **资源有限 / 快速迭代 / 多任务**：PEFT

实际工业：开源开发者几乎都用 PEFT（LoRA），大公司预算大才做全参。

**Q：PEFT 的性能差距能补回来吗？**

A：**大部分场景差距 < 1%**，可以忽略。如果差距明显——
- 增加 LoRA rank
- 在更多层上加 LoRA（不只 attention，也 FFN）
- 试 DoRA / AdaLoRA 等改进
- 数据量充足时，差距会进一步缩小

对极致质量场景（如 SOTA benchmark），全参 + 高质量数据仍是首选。

**Q：PEFT 之外还有什么省显存的方法？**

A：
- **量化训练**：QLoRA 把 base model 量化到 4-bit
- **Gradient Checkpointing**：用计算换显存
- **ZeRO / FSDP**：分布式显存优化
- **CPU Offload**：把 optimizer states 卸到 CPU

这些方法可以**组合使用**——QLoRA + Gradient Checkpointing 能让单张 24GB 消费级 GPU 训 7B 模型。

### ⚠️ 常见陷阱

- 不要说"PEFT 一定比全参好"——本质是 trade-off
- 要懂存储成本——多任务场景下 PEFT 优势巨大

---

## Q2：PEFT 方法谱：Adapter / Prefix / Prompt / LoRA ⭐⭐⭐⭐

### 🎯 一句话标答

> 四类 PEFT：**Adapter** 在层间插入小网络、**Prefix Tuning** 在每层 attention 前加可学习的"前缀向量"、**Prompt Tuning** 只在输入端加可学习 prompt、**LoRA** 在权重矩阵旁加低秩分解——**LoRA 综合最优**，是事实标准。

### 🗣️ 30 秒口语版

"PEFT 方法可以按**插入位置**分四类——

**1. Adapter（2019, Houlsby）**：在 Transformer 每层后插入一个小 bottleneck 网络（降维-激活-升维）。**优点**：模块化清晰。**缺点**：**推理时增加了串行步骤**——延迟变高。

**2. Prefix Tuning（2021）**：在每层的 attention 前加一组**可学习的 KV 向量**（prefix），让 attention 同时关注这些 prefix。**优点**：不动原权重。**缺点**：占用序列长度，可学习参数少时效果差。

**3. Prompt Tuning（2021）**：极端简化版——只在**输入 embedding 端**加可学习的 prompt token。**优点**：实现极简。**缺点**：效果显著差于其他方法，特别是小模型。

**4. LoRA（2021）**：在每个**权重矩阵** W 旁加 **W + BA** 的低秩分解。**优点**：表达力强、训练完可合并、零推理开销。**缺点**：相对其他方法可学习参数稍多（但仍然 < 1%）。

**当前格局**：**LoRA 一统天下**，其他几种主要在学术研究中提及。"

### 🔍 四种 PEFT 对比

| 方法 | 插入位置 | 推理延迟 | 表达力 | 可学习参数 | 当前地位 |
|---|---|---|---|---|---|
| Adapter | 层间 bottleneck | 增加 5-10% | 中 | 0.5-2% | 历史方案 |
| Prefix Tuning | Attention prefix | 增加（序列变长） | 弱 | 0.1-0.5% | 已淘汰 |
| Prompt Tuning | Input embedding | 几乎无 | 弱（小模型差） | 0.01-0.1% | 已淘汰 |
| **LoRA** | 权重矩阵旁 | **零（可合并）** | **强** | **0.1-1%** | **主流** |

### 🔥 高频追问 Top 3

**Q：为什么 LoRA 胜过 Adapter？**

A：三点——
- **零推理延迟**：LoRA 训练完可以把 BA 加到 W 上，推理时不增加任何计算。Adapter 必须保留 bottleneck 结构，每层多一次小矩阵乘
- **表达力更强**：LoRA 的低秩分解可以更灵活，rank 调整空间大
- **实现简洁**：LoRA 加在权重旁，不破坏原结构；Adapter 改变了网络拓扑

**Q：Prompt Tuning 完全没用了吗？**

A：在**大模型**（>10B）上，Prompt Tuning 可以接近 full fine-tuning——但**小模型**上效果显著差。所以现在用得少。

一个例外是**多任务场景**——为每个任务训一个 soft prompt，base model 完全共享，理论上比 LoRA 还省空间（每任务只几个 KB）。但实际效果不如 LoRA，所以也未推广。

**Q：能不能把多种 PEFT 组合？**

A：**可以但收益小**。比如 LoRA + Prefix Tuning，理论上能 cover 更广，但实践中 LoRA 单独就够，组合反而增加超参复杂度。

唯一常见组合是 **LoRA + 不同层不同 rank**（如 AdaLoRA，下面会讲）。

### ⚠️ 常见陷阱

- 不要把所有 PEFT 混为一谈——四类机制不同
- LoRA 是事实标准，但要懂为什么胜出（零延迟 + 表达力）

---

## Q3：LoRA 完整原理与推导 ⭐⭐⭐⭐⭐（必问）

### 🎯 一句话标答

> LoRA 假设**权重更新 ΔW 是低秩的**——把原本要训的 ΔW（d×d）分解成两个小矩阵 B（d×r）和 A（r×d）的乘积。**只训 BA、冻结 W**，参数量从 d² 降到 2dr，r << d。

### 🗣️ 30 秒口语版

"LoRA 的完整故事可以分三段——

**1. 关键 insight**：微调时 ΔW 的本征秩很低

LoRA 的作者 Hu et al. 观察到——**预训练已经塞进了大部分知识，微调时权重变化 ΔW 是低秩的**。这个假设来自对真实微调任务的实证分析——把 ΔW 做 SVD，发现前几个奇异值占了大部分能量。

**2. 数学形式**

原始权重矩阵 W ∈ R^(d×d)（attention 的 Q/K/V/O 投影、FFN 的矩阵都符合）。

LoRA 不直接训 ΔW，而是分解成两个小矩阵：

`ΔW = B · A`，其中 B ∈ R^(d×r)，A ∈ R^(r×d)，r << d

前向变成：

`y = W·x + ΔW·x = W·x + B·A·x`

可学习参数从 d² 降到 2dr。如果 d=4096, r=8，参数量从 16.8M 降到 65K——**减少 250×**。

**3. 关键细节**：

- **初始化**：A 用高斯初始化，B 用零初始化——保证训练开始时 BA = 0，等于原模型
- **缩放系数 α**：实际形式是 `y = Wx + (α/r)·BAx`，α 控制 LoRA 的影响强度
- **应用层**：通常加在 Attention 的 Q、V 投影（实测效果最好），有时也加 K、O 和 FFN

训练完后，可以**合并**：W' = W + (α/r)BA，得到一个**和原模型完全等价的新模型**，推理时无额外开销。"

### 📐 完整数学推导

**原始 Transformer 层中的某个线性变换**：
$$h = Wx, \quad W \in \mathbb{R}^{d \times d}$$

**LoRA 假设**：微调时的 ΔW 可以用低秩矩阵近似：
$$\Delta W \approx BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}, r \ll d$$

**前向公式**：
$$h = Wx + \Delta Wx = Wx + BAx$$

**带缩放系数 α 的标准形式**：
$$h = Wx + \frac{\alpha}{r} BAx$$

α 是个超参（通常 16 或 32），r 是 rank（通常 8 或 16）。**α/r 用来调节 LoRA 模块的强度，让训练对 r 的选择不那么敏感**。

**参数量对比**（设 d=4096）：

| 配置 | 参数量 | 与原 W 比例 |
|---|---|---|
| 完整 W（d×d） | 16,777,216 | 100% |
| LoRA r=8 (2dr) | 65,536 | 0.39% |
| LoRA r=16 | 131,072 | 0.78% |
| LoRA r=64 | 524,288 | 3.13% |

**关键初始化**：
- $A \sim \mathcal{N}(0, \sigma^2)$ 高斯初始化
- $B = 0$ 全零初始化

→ 初始时 BA = 0，模型行为和原 base model 完全一致。这样**训练初期不会破坏 base model**，慢慢学习 ΔW。

### 💻 简洁实现

```python
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base_layer = base_layer  # 冻结的原层
        self.r = r
        self.scale = alpha / r
        
        in_dim = base_layer.in_features
        out_dim = base_layer.out_features
        
        # 低秩矩阵
        self.A = nn.Parameter(torch.randn(r, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, r))
        
        # 冻结原参数
        for p in self.base_layer.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + self.scale * lora_out
    
    def merge(self):
        """训练完后合并到原权重"""
        with torch.no_grad():
            self.base_layer.weight += self.scale * (self.B @ self.A)
```

### 🔍 LoRA 应用层选择（论文实验结果）

| 应用层 | r=4 性能 | 参数量 |
|---|---|---|
| 只加 Q | 中 | 最少 |
| 只加 V | 中（略优于 Q） | 最少 |
| **Q + V** | **最佳** | 2× |
| Q + K + V + O | 略优于 QV | 4× |
| 包含 FFN | 微弱提升 | 大幅增加 |

**经验法则**：先加 Q+V，效果不够再扩展到 K+O 和 FFN。LLaMA 系的实践通常加 Q+K+V+O（不加 FFN）。

### 🔥 高频追问 Top 3

**Q：为什么 ΔW 是低秩的？有理论解释吗？**

A：这是个**经验性假设**，但有几个支持论据——

- **预训练理论**：预训练让权重位于"语义流形"上，微调只是在这个流形上小幅移动，自由度低
- **实证 SVD 分析**：对真实任务的 ΔW 做 SVD，前 r 个奇异值通常解释 > 80% 的方差
- **任务特异性**：单个任务需要的能力子空间是有限的，不需要全维度调整

但**有反例**——多任务、新领域的 ΔW 可能不低秩，这时 LoRA 效果会下降，需要全参或者更大 rank。

**Q：为什么 B 要全零初始化，A 不行吗？**

A：核心要求是 **BA = 0 在训练开始时**。两种方式都行——
- A=0, B=随机
- A=随机, B=0

但**B=0 更常用**，因为：
- B 是"上行投影"，决定输出方向。零初始化让模型从中性状态开始学
- A=0 也能 work，但有些场景下 A 的方向需要继承，零初始化打破这种继承
- 大部分实现都用 B=0，是约定俗成

不能两者都随机——那样 BA ≠ 0，训练初期就破坏了 base model。

**Q：LoRA 的 r 怎么选？8 / 16 / 64 各代表什么？**

A：详见 Q4。简短版——
- **r=4-8**：极轻量，适合简单任务
- **r=16-32**：常见，适合 SFT
- **r=64-128**：接近全参微调的能力，适合复杂任务
- **r >= 128**：边际收益递减，不如直接全参

### ⚠️ 常见陷阱

1. **要现场推导 LoRA 公式**：高频白板题
2. **初始化方向不能搞反**：B=0, A=随机
3. **要懂 α 的作用**：α/r 是缩放，不是简单系数

### 🏢 大厂偏好

- **字节 / DeepSeek**：必问，要求白板推导
- **应用算法岗**：会问"你的项目里用了多大 r？为什么？"

### 📚 延伸阅读
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - 原论文

---

## Q4：LoRA 的 rank r 和 alpha 怎么选 ⭐⭐⭐⭐⭐（工程）

### 🎯 一句话标答

> r 控制**表达力上限**，α 控制 **LoRA 影响强度**。经验法则：**r=16, α=32**（α=2r）是好默认；任务简单可降到 r=8，任务复杂或全参替代可升到 r=64+。

### 🗣️ 30 秒口语版

"这是个工程经验题，能答好体现你真的调过 LoRA。

**r（rank）的选择**：

直觉——r 越大，LoRA 的表达力越强，但参数和显存也越大。

- **r=4-8**：极轻量，适合**风格调整、简单任务**（如让模型说话风格变）
- **r=16-32**：**事实标准**，覆盖大多数 SFT 场景
- **r=64-128**：接近全参微调能力，适合**复杂领域适配**
- **r=256+**：边际收益递减，不如直接全参微调

**α（alpha）的选择**：

α 在公式 `α/r · BAx` 中作为缩放系数。经验——

- **α = 2r**（如 r=16 时 α=32）：**最常用**，让有效缩放系数 = 2
- **α = r**：缩放系数 = 1，LoRA 影响适中
- **α << r**：LoRA 影响弱，几乎不起作用
- **α >> r**：LoRA 影响强，可能过度

**为什么要 α/r 缩放**？这样**改变 r 时不用重新调 lr**——比如 r 从 8 变到 16，α 保持 32，则缩放从 4 变到 2，模型行为变化可控。

**实战配方**：
- 默认起点：**r=16, α=32**
- 效果不够：升 r 到 32 或 64
- 显存紧张：降 r 到 8
- 学习率：1e-4 到 5e-4（比全参微调大！）"

### 🔍 r 和 α 的实战参考

| 场景 | r | α | lr |
|---|---|---|---|
| 风格调整 / 简单 SFT | 4-8 | 16 | 5e-4 |
| **标准 SFT** | **16** | **32** | **3e-4** |
| 复杂领域适配 | 32-64 | 64 | 2e-4 |
| 接近全参替代 | 128+ | 256 | 1e-4 |

### 💡 为什么 LoRA 的 lr 比全参微调大？

这是个常被忽略的细节——

全参微调 lr 一般 1e-5 到 5e-5；LoRA 推荐 1e-4 到 5e-4，**大一个数量级**。

**原因**：LoRA 训练的是 BA，是从零开始的低秩矩阵——没有"已经训好的权重需要保护"的问题。可以用大 lr 加速收敛。全参微调要保护已有权重，必须小 lr。

**实际经验**：
- LoRA: 3e-4（≈ 全参 3e-5 × 10）
- 全参: 3e-5

### 🔥 高频追问 Top 3

**Q：α 是不是越大越好？**

A：**不是**。α 太大有两个问题——
- LoRA 输出占主导，base model 信号被压制
- 训练不稳定，loss 可能 spike

经验上 **α/r ≤ 2-4** 是合理范围。

**Q：能不能不同层用不同 r？**

A：可以——这就是 **AdaLoRA** 的思路（下面会讲）。AdaLoRA 让模型自适应决定每层的 r：重要层 r 大，不重要层 r 小。理论上更优，但工程上复杂。

实践中**所有层用同一 r 已经够好**，不一定要 AdaLoRA。

**Q：rank 是不是越大越好？训出来再剪枝呢？**

A：**rank 大有边际收益**，但训练成本也大。LoRA 之后剪枝的工作有，但收益小——LoRA 本来就是低秩，进一步剪反而损失能力。

**经验**：直接选合适的 r，不要"训大再剪"。

### ⚠️ 常见陷阱

1. **α 和 r 的关系要记住**：α = 2r 是常用配置
2. **LoRA 用大 lr**：比全参微调大 10×
3. **不要无脑加大 r**：边际收益递减

### 🏢 大厂偏好

- **应用算法岗**：必问，要求基于具体场景给配置
- **字节 / 阿里**：可能问"r=128 和全参微调还有差距吗"

---

## Q5：QLoRA 完整解析 ⭐⭐⭐⭐⭐（深度）

### 🎯 一句话标答

> QLoRA = **4-bit 量化 base model + LoRA**，三个关键创新：**NF4 数据类型**（4-bit Normal Float）、**Double Quantization**（量化常数也量化）、**Paged Optimizer**（CPU offload 防 OOM）——单张 24GB GPU 微调 65B 模型。

### 🗣️ 30 秒口语版

"QLoRA 是 2023 年 Dettmers 等人的工作，把 LoRA 推到了消费级 GPU 能用的程度。

**核心思路**：base model 不需要训，那为什么用 BF16 加载？**用 4-bit 量化加载即可**——只在 LoRA 部分用高精度。

但简单 4-bit 量化（如 INT4）有问题——精度太低，模型质量崩坏。QLoRA 做了三个关键改进——

**1. NF4（4-bit Normal Float）**：
- 普通 INT4：把 [-1, 1] 均匀分成 16 个区间
- NF4：根据**标准正态分布**设计 16 个量化点——更密集地覆盖高频区域，**精度比 INT4 高一截**
- 神经网络权重近似服从正态分布，所以 NF4 适配

**2. Double Quantization**：
- 量化时需要存 scale factor（每 64 个权重一个）——这些 scale 本身是 FP32
- 对 scale **再做一次量化** → 8-bit
- 减少 0.5 bit/参数 的额外开销

**3. Paged Optimizer**：
- 用 NVIDIA Unified Memory，**让 optimizer states 在 GPU 和 CPU 之间自动 page**
- 显存峰值降低，OOM 时自动 swap 而不是 crash
- 关键在于 memory spikes（如梯度累积的某个时刻）能 graceful 处理

**效果**：单张 RTX 3090 (24GB) 能微调 33B 模型；4×3090 能微调 70B。**性能损失约 1-2%**，对大多数场景可接受。

QLoRA 让"个人 / 小团队 fine-tune 大模型"成为可能——是 2023 年最重要的工程突破之一。"

### 📐 三大创新详解

#### 创新 1：NF4 量化

**普通 4-bit 量化**（INT4）：
- 把数值范围 [-1, 1] 均匀切分成 16 段
- 量化点：-1.0, -0.875, -0.75, ..., 0.75, 0.875, 1.0
- 适合**均匀分布**的数据

**NF4 量化**：
- 基于标准正态分布 N(0,1) 的分位点
- 量化点在中心密集，边缘稀疏
- 16 个量化点的实际值（论文 Table 1）：
```
[-1.0, -0.696, -0.525, -0.395, -0.284, -0.184, -0.091, 0.0,
 0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723, 1.0]
```

**为什么 NF4 更好**？神经网络权重经过初始化和训练，**接近正态分布**——大部分权重在中心附近，少数在边缘。NF4 在中心给更多量化点，**总体量化误差更小**。

**实测**：NF4 精度损失比 INT4 小约 50%。

#### 创新 2：Double Quantization

**问题**：4-bit 量化时，每个 block（默认 64 个权重）有一个 FP32 的 scale。这些 scales 占用：

- 每参数 = 4-bit (权重) + 32-bit / 64 (scale) = 4.5 bit/参数

**Double Quantization** 把 scales 再量化到 8-bit：

- 每参数 = 4-bit + 8-bit/64 + 32-bit/256 (二次 scale) ≈ 4.13 bit/参数

**节省 ~0.4 bit/参数**——对 70B 模型，节省约 3GB 显存。看起来小，但累积起来是真金白银。

#### 创新 3：Paged Optimizer

**问题场景**：训练过程中存在 **memory spikes**——
- 梯度计算的某个瞬间
- Optimizer step 期间
- 这些 spikes 可能超过 GPU 显存，导致 OOM

**Paged Optimizer**：
- 用 NVIDIA Unified Memory 机制
- Optimizer states 分配在 CPU 和 GPU 共享的内存池
- spike 时自动 page 出 GPU 到 CPU；需要时再 page 回来
- 相比硬性 OOM，**graceful 慢一点但能跑完**

类似 OS 的 swap，但是给 GPU 用。

### 🔍 QLoRA 训练流程

```
1. 加载 base model：
   - 用 NF4 量化，每参数 ~4 bit
   - 70B 模型从 140GB (BF16) 降到约 35GB
   - 加 Double Quantization 进一步降到 ~28GB

2. 反量化进行前向：
   - Forward 时，把 4-bit 权重反量化到 BF16
   - 算完即弃，不持久占显存
   - 这是 QLoRA 的"compute dequant"

3. LoRA 部分用 BF16：
   - LoRA 的 B、A 矩阵都是 BF16
   - 梯度只对 LoRA 部分算

4. 反向传播：
   - 梯度只更新 LoRA
   - Base model 量化权重完全不动

5. Optimizer step：
   - 只更新 LoRA 的优化器状态
   - 如果显存紧，用 Paged Optimizer
```

### 💡 QLoRA 的性能影响

DEttmers 论文的实验：

| 配置 | MMLU |
|---|---|
| LLaMA-65B 全参微调 | 63.9 |
| LLaMA-65B LoRA (BF16) | 63.4 |
| **LLaMA-65B QLoRA (NF4)** | **63.5** |
| LLaMA-65B INT4 + LoRA | 61.8 |

**关键观察**：
- QLoRA (NF4) 几乎不损失性能
- NF4 比 INT4 显著好（差 1.7 个点）
- LoRA vs 全参的差距本来就小

### 🔥 高频追问 Top 3

**Q：NF4 为什么比 INT4 好？**

A：核心是**适配权重分布**。神经网络权重接近正态分布——大部分在中心，少数在边缘。

- **INT4**（均匀）：每个量化点覆盖等宽区间。中心权重密集但只有 ~3 个量化点服务，边缘权重稀疏却也有 ~3 个量化点，**浪费**
- **NF4**（正态）：中心密集（~8 个量化点）、边缘稀疏（~4 个），**资源分配匹配数据分布**

类比 Huffman 编码——高频用短码、低频用长码。NF4 是同样思想的量化版。

**Q：QLoRA 推理时怎么处理？**

A：两种选择——
- **保持量化**：推理时仍用 NF4，节省显存。但需要框架支持 NF4 推理（bitsandbytes 等）
- **合并 + 反量化**：训练完后，把 LoRA 合并到 base model 中并反量化回 BF16，得到一个标准的 BF16 模型。**这是更常见的部署方式**，能用所有标准推理框架（vLLM、TGI 等）

实践中第二种更主流——训练用 QLoRA 省钱，部署用 BF16 标准化。

**Q：QLoRA 能做继续预训练吗？**

A：**理论上可以，但不推荐**。继续预训练需要更新所有参数学习新知识，LoRA 的低秩约束太强。

经验做法：
- 继续预训练用全参（或 ZeRO）
- SFT / 对齐用 QLoRA

### ⚠️ 常见陷阱

1. 不要把 NF4 和 INT4 搞混
2. 要懂 Double Quantization 的作用——量化常数也量化
3. Paged Optimizer 是 NVIDIA Unified Memory，不是简单的 CPU offload

### 🏢 大厂偏好

- **应用算法岗 / 小团队**：必问，QLoRA 是降本利器
- **字节 / DeepSeek**：会让你详述三大创新

### 📚 延伸阅读
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

---

## Q6：DoRA / AdaLoRA / VeRA 等改进 ⭐⭐⭐

### 🎯 一句话标答

> LoRA 的改进主要在三个方向：**AdaLoRA** 自适应 rank 分配、**DoRA** 把权重分解成方向+幅度、**VeRA** 用极少参数（共享 BA 矩阵）——**DoRA 是 2024 最受关注的改进**。

### 🗣️ 30 秒口语版

"LoRA 之后有一堆改进，挑几个重要的——

**AdaLoRA（2023）**：让每层自动决定 r 大小——重要层 r 大，不重要层 r 小。通过迭代剪枝实现。**理论上更优**，但**工程复杂**，未广泛采用。

**DoRA（2024, NVIDIA）**：核心 insight——把权重分解成**方向（direction）+ 幅度（magnitude）**两部分，只对方向用 LoRA、幅度直接学。`W = m * (V + LoRA) / ||V + LoRA||`。

DoRA 的论文显示**在小 rank 下显著胜过 LoRA**（比如 r=4 时 DoRA ≈ LoRA r=16）。**2024 起越来越多采用**，是 LoRA 的下一代。

**VeRA（2024）**：极端节俭版——所有层的 LoRA 共享同一对 BA 矩阵，每层只学一对 scalar (b, d) 来缩放。**参数量比 LoRA 少 10×**，但效果略弱。

**LoRA+ (2024)**：观察到 A 和 B 的最优 lr 不同——B 的 lr 应该大于 A（比如 16×）。简单改进就能提升。

**其他**：QA-LoRA（量化感知）、LoRA-FA（冻结 A）、LongLoRA（长上下文专用）等。"

### 🔍 主要改进对比

| 方法 | 核心思想 | 相比 LoRA 优势 | 复杂度 |
|---|---|---|---|
| AdaLoRA | 自适应 rank | 每层 r 优化 | 高 |
| **DoRA** | **方向+幅度分解** | **小 r 显著更好** | **中** |
| VeRA | 共享 BA + scalar | 参数少 10× | 低 |
| LoRA+ | A、B 不同 lr | 简单提升 | 极低 |
| LongLoRA | 长上下文专用 | 支持 32K+ | 中 |

### 🔥 高频追问 Top 3

**Q：DoRA 的方向+幅度分解具体怎么做？**

A：DoRA 把权重 W 分解成：
$$W = m \cdot \frac{V}{\|V\|}$$
- $m \in \mathbb{R}^d$：幅度向量
- $V \in \mathbb{R}^{d \times d}$：方向矩阵
- $V/\|V\|$ 是按列归一化

微调时——
- **m**：直接全量训练（向量小，不贵）
- **V**：用 LoRA 训练（V + BA）
- 最后归一化

**直觉**：神经网络微调中，**方向和幅度的变化是相对独立的**——LoRA 把它们混在一起学，效率低；DoRA 解耦后更高效。

**Q：什么时候应该用 DoRA 替代 LoRA？**

A：**小 rank** 场景（r ≤ 8）。这时 LoRA 表达力不够，DoRA 能挤出更多性能。**大 rank** 场景两者接近，用哪个都行。

**Q：未来 PEFT 还会有什么大改进？**

A：可能的方向——
- **任务感知 PEFT**：根据任务类型自动决定 PEFT 配置
- **Cross-task transfer**：多任务 LoRA 之间互相迁移
- **Memory-efficient PEFT**：进一步降低显存
- **理论分析**：理解 ΔW 低秩性的本质

但**LoRA 本身已经接近成熟**，未来大改进可能来自**和量化、并行的结合**。

### ⚠️ 常见陷阱

- 不要只知道 LoRA——DoRA 是 2024 的趋势
- AdaLoRA 理论好但工程复杂，没成为主流

---

## Q7：LoRA 的合并部署与多 adapter ⭐⭐⭐

### 🎯 一句话标答

> 训练完后 LoRA 可以**直接合并**到原权重——`W' = W + αBA/r`，得到普通模型。也可以**保持分离**支持**多 adapter 切换**（同一 base 服务多任务）。

### 🗣️ 30 秒口语版

"LoRA 部署有两种选择——

**方式 1：合并部署**

把 LoRA 合并到 base model：
`W' = W + (α/r) · BA`

合并后是个普通的 BF16 模型，可以用 vLLM、TGI 等任何标准推理框架。**适合单一任务场景**。

**方式 2：保持分离，运行时加载**

Base model 加载一次，多个 LoRA adapter 按需切换。比如——
- 中文对话场景：加载 chinese-lora
- 代码生成场景：切换到 code-lora

这种方式**节省显存**（多个任务共享 base）+ **快速切换**。HuggingFace PEFT 库直接支持。

**Multi-LoRA 部署**：S-LoRA、PunicaServe 等工作支持**单 GPU 上同时服务多个 LoRA**——一次性 batch 里有不同任务的请求，路由到不同 adapter。**适合 SaaS 多租户场景**。"

### 📐 合并实现

```python
# LoRA 合并代码
def merge_lora_to_base(base_layer, lora_A, lora_B, alpha, r):
    """把 LoRA 合并到 base 权重，等价于一次完整的 BF16 模型"""
    delta_W = (alpha / r) * (lora_B @ lora_A)  # (out_dim, in_dim)
    base_layer.weight.data += delta_W
    return base_layer
```

合并后的模型可以**保存为标准 HuggingFace 格式**：

```python
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
peft_model = PeftModel.from_pretrained(base, "your-lora-adapter")
merged = peft_model.merge_and_unload()  # 合并并返回普通模型
merged.save_pretrained("merged-llama-3-8b")
```

### 🔥 高频追问 Top 3

**Q：合并后能再次微调吗？**

A：可以——把合并后的模型当成新 base，再加新 LoRA 训练。这是**逐步微调**的常见做法：
- Base → SFT-LoRA → 合并 → DPO-LoRA → 合并

但要小心——多次合并会**累积训练误差**，建议链路尽量短。

**Q：多 LoRA 同时服务怎么实现高效？**

A：S-LoRA 这类系统的核心思路——
- **Base 权重 KV Cache 共享**：所有请求共用 base 前向
- **LoRA 部分动态加载**：每个请求带 adapter ID
- **Unified Paging**：LoRA 权重在 GPU 上 paging，按需加载

实测 S-LoRA 单卡能服务**几百个 LoRA**，吞吐和单一模型接近。

**Q：QLoRA 训完怎么部署？**

A：见 Q5 的最后一个追问——通常**合并并反量化回 BF16**：
1. 用 BF16 加载 base model
2. 加载 QLoRA adapter
3. 合并 LoRA 到 BF16 权重
4. 保存为标准 BF16 模型
5. 用 vLLM 等推理

这样部署时享受 BF16 标准框架的所有优化。

### ⚠️ 常见陷阱

- LoRA 合并是无损的——`W' = W + αBA/r` 数学上等价
- Multi-LoRA 服务需要专门系统（S-LoRA），不是简单 add

### 📚 延伸阅读
- [S-LoRA](https://arxiv.org/abs/2311.03285) - 多 LoRA 服务系统

---

## Q8：LoRA 在 MoE 上的特殊问题 ⭐⭐⭐

### 🎯 一句话标答

> MoE 模型的 LoRA 有特殊问题——**每个 expert 是独立的 FFN**，怎么加 LoRA 是个设计选择：**全 expert 共享一个 LoRA**（参数少但能力弱）vs **每 expert 独立 LoRA**（参数多但能力强）。

### 🗣️ 30 秒口语版

"MoE + LoRA 是个 2024 才被广泛讨论的问题，是 DeepSeek 系面试的高级题。

**MoE 模型结构**：FFN 不是一个，而是 E 个 expert（DeepSeek-V3 有 256 个）。每个 token 经过 router 选 top-k 个 expert 处理。

**LoRA 加在哪里的选择**——

**方案 1：只加在 Attention 层**——最简单，避开了 expert 问题。但 FFN 是表达力的大头，效果可能受限。

**方案 2：每个 expert 加独立 LoRA**——每个 expert 有自己的 BA 矩阵。参数量大（× E），但每个 expert 学到的能力被独立微调。

**方案 3：全 expert 共享一个 LoRA**——所有 expert 用同一对 BA。参数极少，但所有 expert "同步"被影响，破坏了 expert 专业化。

**方案 4：Router 加 LoRA**——只调 expert 选择策略，不动 expert 本身。新颖但效果未充分验证。

**当前最佳实践**：Attention 全加 + FFN 选择性加（top-k 高频 expert），或者只调 router。DeepSeek 论文里 PEFT 部分讨论过这个问题。"

### 🔍 不同方案对比

| 方案 | 参数量 | 表达力 | 训练稳定性 | 适用场景 |
|---|---|---|---|---|
| 仅 Attention | 小 | 中 | 高 | 简单 SFT |
| 全 expert 独立 LoRA | 大（×E） | 强 | 中 | 全面微调 |
| 共享 LoRA | 极小 | 弱 | 高 | 风格调整 |
| Router LoRA | 极小 | 中 | 中 | 任务分配优化 |

### 🔥 高频追问 Top 3

**Q：MoE LoRA 的训练有什么坑？**

A：
- **Load balance**：某些 expert 见到的 token 少，LoRA 训不充分
- **冷启动**：训练初期 router 可能不稳定，影响 LoRA 学习
- **显存放大**：每 expert 独立 LoRA 会让参数量乘 E 倍

**Q：DeepSeek-V3 的 PEFT 怎么做？**

A：DeepSeek 报告里详细讨论过——
- Attention 部分加 LoRA
- FFN 部分（包括 MoE expert）选择性加
- 用更小的 lr 防遗忘

具体配置因任务而异，但**MoE 模型 PEFT 不像 Dense 那样标准化**。

**Q：MoE LoRA 部署有什么挑战？**

A：合并时——
- 每个 expert 独立合并自己的 LoRA
- 共享 LoRA 需要乘到所有 expert
- 部署时和 dense LoRA 一样可以合并

挑战在于**多 LoRA 服务时的 expert 路由**——不同 adapter 可能改变路由策略，复杂度高。

### ⚠️ 常见陷阱

- MoE LoRA 没有标准方案——需要任务特定调整
- Load balance 问题在 MoE LoRA 上更突出

### 🏢 大厂偏好

- **DeepSeek 系**：可能问 MoE PEFT 设计
- **应用算法岗**：MoE 模型还不普及，问的少

---

## 📝 本章小结

| 关键点 | 你必须能脱口而出 |
|---|---|
| 全参代价 | 显存爆炸 + 存储多份 + 训得慢 |
| 四类 PEFT | Adapter / Prefix / Prompt / LoRA |
| LoRA 公式 | y = Wx + (α/r)·BAx，B=0、A 高斯初始化 |
| LoRA 参数量 | 2dr（vs 全参 d²），节省 d/(2r) 倍 |
| α 和 r 关系 | 通常 α = 2r |
| LoRA lr | 比全参大 10× |
| QLoRA 三大创新 | NF4 量化 + Double Quantization + Paged Optimizer |
| NF4 vs INT4 | NF4 适配正态分布，精度高 |
| DoRA | 方向+幅度分解，小 r 显著优于 LoRA |
| LoRA 合并 | W' = W + αBA/r，数学等价 |

## ✅ 自测题

1. 用 30 行 PyTorch 写一个 LoRALinear 类，包含初始化、forward、merge 三个方法
2. 解释为什么 LoRA 假设 ΔW 是低秩的？有哪些证据支撑？
3. QLoRA 的 NF4 量化为什么比 INT4 好？给出量化点的设计原理
4. r=16, α=32 改成 r=32 后，α 应该设多少？为什么这个关系很重要？
5. 你要训一个支持 10 种语言的对话 LoRA，怎么设计部署架构？

---

> **下一章预告**：第 9 章｜RLHF 与 PPO——完整的奖励模型训练、PPO 算法推导、Reward Hacking 问题。RLHF 是 ChatGPT 出圈的关键，但实现复杂，是字节 / DeepMind 等顶尖团队的高频考点。
