# 第 21 章｜MoE 架构与训练

> **本章定位**：MoE（Mixture-of-Experts）已经是前沿大模型的主流架构：DeepSeek-V3、Qwen3-235B、Kimi K2、Mixtral 都是 MoE。前面的章节在不同地方碰到过它：第 6 章的专家并行（EP）、第 8 章的 MoE 微调、第 15 章的 MoE 推理部署。本章把它当成**一个完整的研究对象**，讲清四件事：路由怎么做、负载怎么平衡、专家怎么切分、为什么能 scale。重点是**证据链**：哪些设计有消融支持，哪些仍有争议（例如共享专家），哪些是开放问题。

> **配套章节**：FFN 与 SwiGLU → 第 3 章 / 训练稳定性、z-loss → 第 5 章 / Scaling Law 与专家并行 → 第 6 章 / MoE 的 PEFT → 第 8 章 Q8 / MoE 推理与超大 EP → 第 15 章 Q6

---

## 📏 标注约定

| 标记 | 含义 |
|---|---|
| 📄 **[论文]** | 原论文/技术报告中明确写出的结论（附出处） |
| 🧮 **[推导]** | 可以严格推出来的数学事实 |
| 💭 **[直觉]** | 常见解释，没有严格证明，或有争议 |
| ❓ **[开放]** | 尚无共识 |

---

## 📌 本章导航

| 序号 | 题目 | 难度 | 类型 |
|---|---|---|---|
| Q1 | MoE 是什么？为什么"参数多但计算少" | ⭐⭐⭐⭐ | 基础 |
| Q2 | **路由：Top-k gating、容量与 token dropping** | ⭐⭐⭐⭐⭐ | 必问 + 推导 |
| Q3 | **负载均衡：辅助损失、z-loss 与 aux-loss-free** | ⭐⭐⭐⭐⭐ | 必问 + 前沿 |
| Q4 | **细粒度专家与共享专家（DeepSeekMoE）** | ⭐⭐⭐⭐⭐ | 必问 |
| Q5 | Token choice vs Expert choice | ⭐⭐⭐⭐ | 进阶 |
| Q6 | MoE 的 Scaling Law：稀疏度、粒度与"有效参数" | ⭐⭐⭐⭐⭐ | 研究视角 |
| Q7 | 训练稳定性与基础设施：精度、初始化、EP、dropless | ⭐⭐⭐⭐ | 工程 |
| Q8 | 专家真的"专"吗？Upcycling 划算吗？ | ⭐⭐⭐⭐ | 证据辨析 |
| Q9 | 研究前沿：百万专家、记忆层、动态计算 | ⭐⭐⭐⭐⭐ | 研究视角 |

---

## Q1：MoE 是什么？为什么"参数多但计算少" ⭐⭐⭐⭐

### 🎯 一句话标答

> MoE 把 Transformer 的 FFN 换成 $N$ 个并列的"专家"FFN 加一个路由器，每个 token 只送进其中 $k$ 个专家（$k \ll N$）。于是**总参数随 $N$ 增长，而每个 token 的计算量只随 $k$ 增长**。这是"条件计算"（conditional computation）思想的落地：📄 Shazeer et al. 2017 首次在 LSTM 语言模型上做到了 1000 倍以上的容量提升且计算效率损失很小；📄 GShard（2020）和 Switch Transformer（2021）把它搬进 Transformer。

### 🗣️ 30 秒口语版

"Dense 模型里每个 token 都要过全部参数。MoE 的想法是：参数是用来'存知识'的，但不是每个 token 都需要全部知识。所以把 FFN 拆成很多个专家，用一个很小的线性路由器给每个 token 打分，只选分数最高的几个专家去算，再按分数加权求和。

结果是两个轴解耦了：**总参数**决定模型能装多少东西，**激活参数**决定每个 token 花多少算力。比如 DeepSeek-V3 总参数 671B，但每个 token 只激活 37B。

代价也很明确：显存要装下全部参数，推理是 memory-bound 的问题会更严重（第 15 章 Q6）；训练时要解决负载不均和通信问题。"

### 📐 结构

一层 MoE（以 top-$k$ 为例）：

$$y = \sum_{i \in \mathcal{T}_k(x)} g_i(x)\, E_i(x), \qquad \mathcal{T}_k(x) = \text{TopK}\big(\{s_i(x)\}_{i=1}^N, k\big)$$

- $E_i$：第 $i$ 个专家，通常就是一个 SwiGLU FFN（第 3 章）。
- $s_i(x)$：路由打分，最常见是 $s = \text{softmax}(W_r x)$（$W_r \in \mathbb{R}^{N \times d}$）。📄 DeepSeek-V3 改用 $s_i = \sigma(u^\top e_i)$（sigmoid），再在选中的专家上归一化。
- $g_i$：门控权重，一般是选中专家的分数在 top-$k$ 内重新归一化。

🧮 **参数与计算**：设专家中间维度为 $h$，SwiGLU 专家参数约 $3dh$。$N$ 个专家的总参数约 $3Ndh$，每 token 计算约 $2 \times 3kdh$ FLOPs。总参数 / 激活参数 ≈ $N/k$（只看 FFN）。

### 📄 代表模型的配置

| 模型 | 总参数 / 激活参数 | 路由专家数 / 每 token 激活 | 共享专家 | 出处 |
|---|---|---|---|---|
| Mixtral 8x7B | 47B / 13B | 8 / 2 | 无 | Jiang et al. 2024 |
| OLMoE-1B-7B | 7B / 1B | 64 / 8 | 无 | Muennighoff et al. 2024 |
| DeepSeek-V3 | 671B / 37B | 256 / 8 | 1 | DeepSeek-AI 2024 |
| Qwen3-235B-A22B | 235B / 22B | 128 / 8 | 无 | Qwen Team 2025 |
| Kimi K2 | 1.04T / 32.6B | 384 / 8 | 1 | Kimi Team 2025 |

💭 趋势很清楚：**专家越来越多、越来越小，激活数固定在 8 左右**。为什么这样做，见 Q4 和 Q6。

### ⚠️ 常见陷阱

- 说"Mixtral 8x7B 有 56B 参数"：attention 等非专家参数是共享的，总参数是 📄 47B，激活 13B。
- 把"激活参数"当成推理成本的全部：decode 时显存带宽要读的是**被选中的专家**，但显存必须**装下全部**专家。batch 大了以后，几乎所有专家都会被某个 token 选中（第 15 章 Q6）。
- 说"MoE 每个专家学一个领域"：这是常见误解，证据见 Q8。

### 📚 出处
- Shazeer et al. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR 2017. arXiv:1701.06538
- Lepikhin et al. *GShard*. ICLR 2021. arXiv:2006.16668
- Fedus, Zoph, Shazeer. *Switch Transformers*. JMLR 2022. arXiv:2101.03961
- Jiang et al. *Mixtral of Experts*. 2024. arXiv:2401.04088
- DeepSeek-AI. *DeepSeek-V3 Technical Report*. 2024. arXiv:2412.19437
- Qwen Team. *Qwen3 Technical Report*. 2025. arXiv:2505.09388
- Kimi Team. *Kimi K2: Open Agentic Intelligence*. 2025. arXiv:2507.20534

---

## Q2：路由：Top-k gating、容量与 token dropping ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 路由器是一个 $d \to N$ 的线性层，给每个 token 选 top-$k$ 个专家。离散的 top-$k$ 不可导，梯度只能通过**被选中专家的门控权重 $g_i$** 回传给路由器。为了让每个专家的计算量可预测，早期系统给每个专家设**容量** $C = \frac{T}{N}\times \text{cf}$（cf 为 capacity factor），超出容量的 token 被**丢弃**（直接走残差）。现代训练系统（MegaBlocks、DeepSeek-V3）用 dropless 实现或足够好的负载均衡来避免丢 token。

### 📐 路由的梯度从哪里来

🧮 输出 $y = \sum_{i \in \mathcal{T}_k} g_i E_i(x)$ 对路由参数的梯度只经过 $g_i$：

$$\frac{\partial \mathcal{L}}{\partial W_r} = \sum_{i \in \mathcal{T}_k} \frac{\partial \mathcal{L}}{\partial y}^{\!\top} E_i(x)\; \frac{\partial g_i}{\partial W_r}$$

几个推论：
1. **没被选中的专家拿不到梯度**，路由器也不知道"选它会怎样"。这就是 MoE 容易**路由坍缩**（少数专家越训越强、越来越常被选）的根源，需要 Q3 的负载均衡来对抗。
2. 📄 Switch Transformer 用 $k=1$，并论证 top-1 也能工作；而 Shazeer 2017 曾认为 $k \ge 2$ 才能让路由器学到"比较"。💭 当前主流用细粒度专家 + $k=8$，这个争论已不那么重要。
3. 如果在 top-$k$ 内重新做 softmax 归一化（Mixtral 的做法），只选一个专家时 $g=1$ 恒定，**路由器得不到梯度**；所以 top-1 的 Switch 不做重新归一化，直接用原始 softmax 概率作 $g$。

### 📐 容量与 token dropping

📄 Switch Transformer 的定义：

$$\text{expert capacity} = \frac{\text{tokens per batch}}{\text{number of experts}} \times \text{capacity factor}$$

- 超出容量的 token 不经过该层专家，表示直接通过残差连接传到下一层。
- 📄 Switch 报告较小的 capacity factor（1.0、1.25）效果更好，并指出**保持较低的丢 token 率对 scale 很重要**（加上负载均衡损失后通常 < 1%）。
- 📄 MegaBlocks（Gale et al. 2022）把 MoE 计算改写成块稀疏矩阵乘，**从不丢 token**，也不需要 padding，报告相对 Tutel 的 MoE 端到端训练提速最高 40%。
- 📄 DeepSeek-V3 明确写道：由于负载均衡做得好，**训练全程和推理都不丢 token**。

### 🔥 高频追问

**Q：推理时为什么不能丢 token？**

A：🧮 训练时丢 token 是一种正则化，代价是那层少算一次；推理时容量取决于同一个 batch 里的其他请求，**你的输出会因为别人的请求而改变**，既不确定也不公平。所以推理一般用不丢 token 的实现，或设足够大的容量。💭 这也是 MoE 推理非确定性的来源之一。

**Q：路由用 softmax 还是 sigmoid？**

A：📄 DeepSeek-V3 从 V2 的 softmax 改为 sigmoid，并在选中专家上归一化。💭 一种解释是：softmax 让专家之间互相竞争（概率和为 1），sigmoid 让每个专家的亲和度独立，更适合配合 Q3 的偏置调节。目前没有看到系统的对照实验来定论，属于 ❓。

### 📚 出处
- Fedus et al. *Switch Transformers*. arXiv:2101.03961（§2.2 容量与负载均衡损失）
- Gale et al. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts*. MLSys 2023. arXiv:2211.15841
- DeepSeek-AI. *DeepSeek-V3*. arXiv:2412.19437（§2.1.2）

---

## Q3：负载均衡：辅助损失、z-loss 与 aux-loss-free ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 负载不均有两个后果：**路由坍缩**（少数专家垄断，其余专家训不到）和**算力浪费**（EP 下最忙的卡决定整步时间）。经典解法是 📄 Switch 的辅助损失 $\alpha N \sum_i f_i P_i$（$\alpha=10^{-2}$）；📄 ST-MoE 加了 **router z-loss** 稳定训练；但辅助损失本身会产生"干扰梯度"、损害主任务，于是 📄 DeepSeek 提出 **aux-loss-free**：给每个专家加一个只用于选择、不参与加权的偏置，按负载在线调节。另一条线是 📄 Qwen 的发现：**平衡的统计范围**（micro-batch vs global-batch）比损失形式更关键。

### 📐 Switch 辅助损失

📄 Fedus et al. 2021：对 $N$ 个专家、含 $T$ 个 token 的 batch $\mathcal{B}$，

$$\mathcal{L}_{\text{bal}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i, \qquad f_i = \frac{1}{T}\sum_{x\in\mathcal{B}} \mathbb{1}\{\arg\max p(x) = i\}, \qquad P_i = \frac{1}{T}\sum_{x\in\mathcal{B}} p_i(x)$$

- $f_i$：实际分到专家 $i$ 的 token 比例（**不可导**）；$P_i$：路由器给专家 $i$ 的平均概率（**可导**）。梯度只通过 $P_i$。
- 乘以 $N$ 是为了让损失在均匀分布时恒为 $\alpha$（$\sum_i \frac1N\cdot\frac1N = \frac1N$），与专家数无关。
- 📄 论文全程用 $\alpha = 10^{-2}$，并称它"足够大以保证均衡，又足够小不至于压过主损失"。

🧮 **为什么它鼓励均匀**：若把 $f$ 近似为 $P$，则 $\sum_i P_i^2 \ge \frac{1}{N}$（Cauchy–Schwarz），等号当且仅当 $P_i = 1/N$。实际中 $f$ 被当作常数，梯度 $\partial \mathcal{L}/\partial P_i \propto f_i$：**越忙的专家，路由概率被压得越狠**。

### 📐 Router z-loss

📄 ST-MoE（Zoph et al. 2022）：

$$\mathcal{L}_z = \frac{1}{B}\sum_{j=1}^{B}\Big(\log \sum_{i=1}^{N} e^{x_i^{(j)}}\Big)^2$$

其中 $x^{(j)}$ 是第 $j$ 个 token 的路由 logits。它惩罚路由 logit 的整体尺度，避免 softmax 进入数值不稳定的区间（与第 5 章输出层的 z-loss 同源）。📄 论文选 $c_z = 0.001$，结论是 **router z-loss 能稳定训练且不损失质量**（他们测试的其他稳定化手段大多以质量为代价）。

### 📐 Aux-loss-free 负载均衡

📄 Wang et al. 2024（DeepSeek），并在 DeepSeek-V3 中采用：

$$g'_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} + b_i \in \text{TopK}\big(\{s_{j,t}+b_j\}, K_r\big) \\ 0, & \text{otherwise}\end{cases}$$

- 偏置 $b_i$ **只影响"选谁"，不影响加权**（加权仍用原始 $s_{i,t}$），所以不产生任何梯度干扰。
- 每步结束后，**过载的专家 $b_i \mathrel{-}= \gamma$，欠载的 $b_i \mathrel{+}= \gamma$**。📄 DeepSeek-V3 在前 14.3T token 用 $\gamma = 0.001$，最后 500B token 设为 0。
- 📄 V3 仍保留一个极小的**序列级**辅助损失（$\alpha = 0.0001$），"只是为了避免单条序列内的极端不均衡"。

### 📐 关键洞察：平衡的"范围"

📄 **DeepSeek-V3 §4.5.3 的对照**：在 1B MoE 上，验证 loss 为序列级辅助损失 2.258、aux-loss-free 2.253、**batch 级辅助损失 2.253**；3B MoE 上是 2.085 vs 2.080 vs 2.080。也就是说，**aux-loss-free 的优势主要来自"在 batch 范围而不是序列范围求平衡"**，换成 batch 级辅助损失也能拿到同样的效果。

📄 **Qiu et al. 2025（Qwen，*Demons in the Detail*）**：主流框架在 micro-batch 内计算 $f_i$，而大模型训练的 micro-batch 只有很少几条序列，等于**强迫每条序列内部均匀**，一段代码的 token 也被平均分给所有专家，抑制了专家专精。改成 **global-batch** 统计 $f_i$（多一步通信）后，在最大 42.8B 参数、400B token 的实验中，困惑度和下游任务都有明显提升，专家的领域专精也明显增强。📄 Qwen3 MoE 采用了 global-batch 负载均衡损失。

> 💭 把两条证据放在一起：**"平衡"应该在足够大、足够多样的范围上统计**。约束越局部，越接近"每个 token 随机分配"，专家越难专精。

### 🔥 高频追问

**Q：aux-loss-free 有没有代价？**

A：📄 DeepSeek-V3 自己指出 batch 级平衡的两个效率隐患：单条序列或小 batch 内可能不均衡；推理时的领域偏移会导致负载不均。前者用序列级小损失兜底，后者靠推理侧的冗余专家部署（第 15 章 Q6）。

**Q：能不能完全不要负载均衡？**

A：📄 OLMoE（§4.1.6）的消融中，不用负载均衡损失时，第一层的 token 在训练初期**全部**被分给同一个专家，之后也只有极少数专家被使用，其余专家成了"占显存但不被使用的死重"；训练和验证 loss 都更差。💭 但 OLMoE 也提到，负载均衡损失可能正是阻止专家"按需"被更频繁使用的原因；如何允许"合理的不均衡"是 ❓ 开放问题（见 Q9 动态计算）。

### 📚 出处
- Fedus et al. *Switch Transformers*. arXiv:2101.03961（式 4–6）
- Zoph et al. *ST-MoE: Designing Stable and Transferable Sparse Expert Models*. 2022. arXiv:2202.08906（§3.3）
- Wang et al. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*. 2024. arXiv:2408.15664
- Qiu et al. *Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models*. 2025. arXiv:2501.11873
- DeepSeek-AI. *DeepSeek-V3*. arXiv:2412.19437（§2.1.2、§4.5.3）

---

## Q4：细粒度专家与共享专家（DeepSeekMoE） ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 📄 DeepSeekMoE 提出两条改动：**细粒度切分**（把 $N$ 个专家各切成 $m$ 份，变成 $mN$ 个小专家、激活 $mK$ 个，计算量不变但组合方式指数级增多）和**共享专家**（$K_s$ 个专家对所有 token 恒定激活，承担通用知识，减少路由专家之间的冗余）。细粒度已成为共识（Qwen3、OLMoE、Kimi K2 都用）；共享专家则有争议：DeepSeek、Kimi K2 保留，📄 OLMoE 的消融认为无益，📄 Qwen3 去掉了。

### 📐 细粒度为什么有用：组合数

🧮 在激活计算量相同的前提下：
- 16 个专家选 2：$\binom{16}{2} = 120$ 种组合；
- 把每个专家切成 4 份，64 个选 8：$\binom{64}{8} \approx 4.4\times 10^9$ 种组合。

💭 组合越多，每个 token 越能"拼出"恰好需要的知识，专家也越容易专精。这是直觉，正式的证据来自 Q6 的粒度 scaling law。

📄 **DeepSeekMoE 的实验**：2B 规模下与 GShard 2.9B 相当（后者专家参数和计算是 1.5 倍）；16B 模型以约 40% 的计算达到 LLaMA2 7B 的水平；145B 的初步实验以 28.5%（甚至 18.2%）的计算达到 DeepSeek 67B 的水平。

### 📐 共享专家：证据不一致

| 立场 | 证据 |
|---|---|
| 支持 | 📄 DeepSeekMoE：在**训练好的**模型上关掉共享专家、改为多激活一个路由专家（计算量不变），Pile loss 从 1.808 升到 2.414；作者据此认为共享专家学到了路由专家没有的通用知识。DeepSeek-V2/V3、Kimi K2 均保留 1 个共享专家 |
| 反对 | 📄 OLMoE（§4.1.3）：在激活参数和总参数相同的条件下（32 选 4 vs 1 共享 + 31 选 3），共享专家**没有带来提升**；作者推测原因是固定一个专家减少了可用的组合数 |
| 实践 | 📄 Qwen2.5-MoE 有共享专家，**Qwen3-MoE 去掉了**（技术报告原文："Unlike Qwen2.5-MoE, the Qwen3-MoE design excludes shared experts"） |

> 🧮 **注意两类证据回答的不是同一个问题**：DeepSeekMoE 是"训练后摘除"实验，证明的是**已训练模型依赖共享专家**（模型围绕它分工，摘掉当然掉点）；OLMoE 是"从头训练有/无"的对照，回答的才是**要不要在设计里放共享专家**。前者不能推出后者。
>
> ❓ **开放问题**：共享专家的收益可能取决于负载均衡的范围（Q3）。如果平衡在序列级强制执行，"通用知识"无处安放，共享专家就有价值；如果平衡在 global-batch 上统计，路由专家本身就能学出"常用专家"。这个假设目前没有直接验证。

### 🔥 高频追问

**Q：细粒度能无限细下去吗？**

A：不能。🧮 专家越小，单个专家的矩阵乘越"瘦"，GPU 利用率越低；路由器的计算和 all-to-all 的通信次数随激活专家数增加。📄 DeepSeek-V3 用节点限制路由（每个 token 最多发往 4 个节点）控制通信。📄 Kimi K2 在 V3 的基础上把专家数从 256 加到 384（激活仍是 8），依据是他们的稀疏度 scaling law（Q6）；同时把 attention head 从 128 减到 64 以降低推理开销。

### 📚 出处
- Dai et al. *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*. ACL 2024. arXiv:2401.06066
- Muennighoff et al. *OLMoE: Open Mixture-of-Experts Language Models*. ICLR 2025. arXiv:2409.02060
- Qwen Team. *Qwen3 Technical Report*. arXiv:2505.09388（§2）
- Kimi Team. *Kimi K2*. arXiv:2507.20534（Table 2）

---

## Q5：Token choice vs Expert choice ⭐⭐⭐⭐

### 🎯 一句话标答

> **Token choice**（每个 token 选 $k$ 个专家）是主流；**Expert choice**（📄 Zhou et al. 2022：每个专家从 batch 里选容量 $C$ 个 token）天然完全均衡、不需要辅助损失，且每个 token 能用到可变数量的专家。原论文报告训练收敛快 2 倍以上；但 📄 OLMoE 在 dropless + 负载均衡损失的设置下发现 token choice 效果更好，expert choice 只是吞吐快约 20%。Expert choice 的根本问题是**违反因果性**：一个 token 被不被选，取决于同 batch 内的其他（包括未来的）token，自回归解码时无法直接使用。

### 📐 为什么 expert choice 与自回归冲突

🧮 Expert choice 对每个专家在整个 batch 的 token 上做 top-$C$。训练时序列的所有 token 都在场，第 $t$ 个 token 是否被专家 $i$ 选中，取决于第 $t+1, t+2, \dots$ 个 token 的分数。推理时未来 token 还不存在，于是**训练和推理的路由规则不一致**。💭 这也是它更常用于 encoder 或多模态（📄 OLMoE 提到丢弃一些图像 token 危害较小）的原因。

### 📚 出处
- Zhou et al. *Mixture-of-Experts with Expert Choice Routing*. NeurIPS 2022. arXiv:2202.09368
- Muennighoff et al. *OLMoE*. arXiv:2409.02060（§4.1.4）

---

## Q6：MoE 的 Scaling Law：稀疏度、粒度与"有效参数" ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> MoE 有两个独立的 scaling 轴：**总参数**和**每 token 计算量**。📄 Clark et al. 2022 给出了路由模型的统一 scaling law 与"有效参数量"；📄 Krajewski et al. 2024 引入**粒度**超参数，发现专家大小等于 FFN 的常见做法"几乎在任何计算预算下都不是最优"，且 MoE 相对 dense 的效率优势随规模扩大而**增大**；📄 Abnar et al. 2025 发现在给定约束下存在**最优稀疏度**；📄 Kimi K2 报告在固定激活参数下，持续增加专家数（稀疏度）一直在降低 loss。

### 📐 三条证据链

1. **统一 scaling law**（📄 Clark et al. 2022, DeepMind）：在五个数量级、上百个专家的范围内拟合，参数量和计算量是两个独立的轴；可以把 MoE 换算成一个"有效参数量"，使所有模型在同一条曲线上 scale。
2. **粒度**（📄 Krajewski et al. 2024）：定义粒度 $G$ = FFN 中间维度 / 专家中间维度。把训练 token 数、模型大小和 $G$ 一起纳入 scaling law 后得出：
   - 最优粒度通常大于 1（即专家应比 dense FFN 小）；
   - MoE 与 dense 的效率差距**随规模和训练预算扩大**。
3. **稀疏度**（📄 Abnar et al. 2025, Apple；📄 Kimi K2）：
   - Abnar et al.：在固定参数量或固定总训练算力等不同约束下，都存在一个最优稀疏度（不活跃参数的比例），它同时改善训练效率和模型效果。
   - Kimi K2 的"稀疏度 scaling law"（稀疏度 = 总专家数 / 激活专家数）：在固定激活参数下，达到相同验证 loss 1.5 时，稀疏度 48 相比稀疏度 8、16、32 分别减少 **1.69×、1.39×、1.15×** 的 FLOPs。他们也指出稀疏度越高，基础设施越复杂，最终选择了 48（384 / 8）。

### 🔥 高频追问

**Q：同样的总参数，MoE 和 dense 谁更好？**

A：💭 在**相同总参数**下，dense 模型每 token 用的计算多得多，通常 loss 更低；📄 DeepSeekMoE 就把"同总参数的 dense 模型"当作 MoE 的**上界**。MoE 的价值在**相同计算**下比较。所以说"MoE 比 dense 强"时要说清楚是 iso-FLOPs 还是 iso-params。

**Q：MoE 的 scaling law 对推理意味着什么？**

A：🧮 训练最优 ≠ 部署最优。MoE 显存占用由总参数决定，带宽需求由激活参数和 batch 决定。第 6 章 Q3 说过，考虑推理成本时会选择"小模型多训"；MoE 把这个权衡变成了三维的（总参数、激活参数、训练 token 数）。❓ 包含推理成本的 MoE scaling law 还没有公认的形式。

### 📚 出处
- Clark et al. *Unified Scaling Laws for Routed Language Models*. ICML 2022. arXiv:2202.01169
- Krajewski et al. *Scaling Laws for Fine-Grained Mixture of Experts*. ICML 2024. arXiv:2402.07871
- Abnar et al. *Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models*. 2025. arXiv:2501.12370
- Kimi Team. *Kimi K2*. arXiv:2507.20534（§2.3 Sparsity Scaling Law）

---

## Q7：训练稳定性与基础设施：精度、初始化、EP、dropless ⭐⭐⭐⭐

### 🎯 一句话标答

> MoE 比 dense 更容易不稳定。📄 Switch 的经验：路由器局部用 **fp32**（selective precision），其余保持 bf16；把初始化尺度缩小到默认的 1/10。📄 ST-MoE 系统比较了多种稳定化手段，结论是 **router z-loss** 稳定且不伤质量，而 GEGLU、RMSNorm 这类乘性结构提升质量但**降低**稳定性。基础设施上，核心是**专家并行（EP）+ all-to-all 通信**，以及 dropless 的块稀疏计算（MegaBlocks）和通信与计算的重叠（DeepSeek-V3 的 DualPipe、节点限制路由）。

### 📐 要点

- **路由器精度**：📄 Switch 把路由器的输入转为 fp32，在局部做 softmax 等计算后再转回，避免在整个模型上用 fp32 带来的通信开销。
- **初始化**：📄 Switch 把权重初始化的尺度系数从 1.0 降到 0.1，报告这显著改善稳定性。
- **乘性结构与稳定性**：📄 ST-MoE："Transformer modifications with more multiplicative components (GEGLU, RMS normalization) worsen stability, but boost quality."
- **专家并行**：每张卡放一部分专家，每层前后各做一次 all-to-all（dispatch 和 combine）。📄 DeepSeek-V3 训练时每层的路由专家均匀部署在 8 个节点的 64 张卡上，每个 token 最多发往 4 个节点，以减少跨节点 IB 流量（→ 第 6 章 Q4、第 15 章 Q6）。
- **MoE 的 PEFT 与推理**：见第 8 章 Q8、第 15 章 Q6。

### ⚠️ 常见陷阱

- 只在单卡上做 MoE 小实验，然后把吞吐结论外推到多机：MoE 的瓶颈在 all-to-all 和负载不均，单卡观察不到。
- 微调时继续用很大的负载均衡系数：💭 微调数据分布很窄，强制均衡会和任务目标冲突。📄 ST-MoE 还报告 sparse 模型在小数据集上微调更容易过拟合，并给出了不同的超参数建议（如更小的 batch、更大的学习率）。

### 📚 出处
- Fedus et al. *Switch Transformers*. arXiv:2101.03961（§2.4）
- Zoph et al. *ST-MoE*. arXiv:2202.08906（§3）
- Gale et al. *MegaBlocks*. arXiv:2211.15841
- DeepSeek-AI. *DeepSeek-V3*. arXiv:2412.19437（§3.2）

---

## Q8：专家真的"专"吗？Upcycling 划算吗？ ⭐⭐⭐⭐

### 🎯 一句话标答

> "每个专家负责一个领域"是流行的误解。早期证据偏向**词法/句法层面**的专精：📄 ST-MoE 发现 encoder 专家有明显专精（标点、专有名词等），decoder 专家几乎没有；📄 Mixtral 在 The Pile 的不同主题上看不到明显的按主题分配，路由更像与句法相关，且相邻 token 常被分到同一专家。但**专精程度取决于负载均衡的范围**：📄 OLMoE、DeepSeek-V3（aux-loss-free）和 Qwen（global-batch）都观察到了更强的领域专精。至于从 dense 模型 **upcycling** 成 MoE：📄 Komatsuzaki et al. 2022 认为划算，📄 OLMoE 发现在长训练下从头训练的 MoE 很快追上并反超。

### 📐 专精的证据

| 研究 | 观察 |
|---|---|
| 📄 ST-MoE | encoder 专家专精明显；decoder 专家缺乏有意义的专精；多语言专家专精，但不按语言划分 |
| 📄 Mixtral | 不同主题（ArXiv、PubMed、PhilPapers）的专家分配分布很相似，只有 DM Mathematics 略有不同；连续 token 被分到同一专家的比例显著高于随机 |
| 📄 OLMoE | 观察到较强的领域专精和词表专精，专家很少共同激活 |
| 📄 DeepSeek-V3 | 在 Pile 测试集上，aux-loss-free 模型比辅助损失基线表现出更强的专家专精模式 |
| 📄 Qiu et al. 2025 | global-batch 负载均衡明显增强了领域专精 |

💭 一种统一的读法：**专精不是 MoE 的天然性质，而是训练约束允许它出现时才会出现**。序列级强制均衡会把专精"平均掉"。

### 📐 Upcycling

- 📄 **Sparse Upcycling**（Komatsuzaki et al. 2022）：把 dense 的 FFN 复制成 $N$ 份作为初始专家，路由器从头学。报告在 T5 和 ViT 上，只用约 50% 的原 dense 预训练成本，就显著超过 dense 版本，也超过用 100% 预算从头训练的 sparse 模型。
- 📄 **OLMoE 的反例**：对已训练 2T token 的 OLMo-1B 做 upcycling，从头训练的 MoE 在 500B token 后就追上，约 600B token 后反超。作者的解释：被大量过训的 dense 权重已处于很"优"的区域，复制后缺少探索；而且 upcycled 模型受制于 dense 模型的超参数（例如没有 QK-Norm）。
- 📄 实践中 Qwen2、MiniCPM 等采用过 upcycling（据 OLMoE 的整理）。

> 💭 结论：**算力紧张、只训少量 token 时 upcycling 划算；长训练时从头训练可能更好**。两篇论文的设置不同（encoder-decoder + expert choice vs decoder-only + token choice），不能简单比较。

### 📚 出处
- Zoph et al. *ST-MoE*. arXiv:2202.08906（§7）
- Jiang et al. *Mixtral*. arXiv:2401.04088（§5 Routing analysis）
- Muennighoff et al. *OLMoE*. arXiv:2409.02060（§4.1.5、§5）
- Komatsuzaki et al. *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints*. ICLR 2023. arXiv:2212.05055

---

## Q9：研究前沿：百万专家、记忆层、动态计算 ⭐⭐⭐⭐⭐

### 方向 1：把粒度推到极限

- 📄 **PEER**（He 2024, *Mixture of A Million Experts*）：用 product key 检索，从超过一百万个极小的专家里选择，报告在性能-计算权衡上优于 dense FFN 和粗粒度 MoE。这是 Q6 粒度 scaling law 的极端版本。
- 📄 **Memory Layers at Scale**（Berges et al. 2024, Meta）：用可训练的 key-value 查找层增加参数而不增加 FLOPs，在 1T token、最多 128B 记忆参数的规模上，报告优于两倍计算量的 dense 模型，在计算和参数都匹配时也优于 MoE，**事实类任务上提升尤其明显**。
- ❓ "专家"和"记忆"的边界在哪里？当专家小到一个向量时，MoE 就变成了一个可学习的检索系统。

### 方向 2：每个 token 用不同的计算量

- 📄 **MoE++**（Jin et al. 2024）：加入三类"零计算专家"（zero、copy、constant），让简单 token 少用甚至不用 FFN 专家，报告专家前向吞吐提升 1.1–2.1×。
- 📄 **LongCat-Flash**（美团, 2025）：560B 总参数，借助零计算专家，每 token 激活 18.6B–31.3B（平均约 27B），并用 shortcut-connected MoE 扩大计算与通信的重叠窗口。
- ❓ 动态计算与负载均衡天然冲突（Q3）：如何既让"难 token 多算"，又不让 EP 下的卡忙闲不均？

### 方向 3：路由本身

- ❓ 路由器只是一个线性层，只看当前层的隐藏状态。📄 MoE++ 用 gating residual 让路由参考上一层的选择；路由能否利用更多上下文、能否做层间规划，仍有空间。
- ❓ **路由可解释性**：专家激活是否可以作为稀疏、可解释的特征（与第 23 章的稀疏自编码器类比）？

### 🧪 可以动手的实验思路（文字）

1. **负载均衡范围的消融**：用 OLMoE 开源代码在小规模（几亿参数、几十 B token）上对比序列级、micro-batch、global-batch 三种统计范围，记录 loss 和每个专家在不同领域数据上的激活分布，复现 Qiu et al. 2025 和 DeepSeek-V3 §4.5.3 的结论。
2. **共享专家之争**：在相同激活和总参数下，对比"有共享专家 + 序列级平衡"和"无共享专家 + global-batch 平衡"四种组合，检验 Q4 提出的假设。
3. **路由坍缩的动力学**：关掉负载均衡损失，记录每个专家被选中的比例随训练步数的变化，观察坍缩何时发生、是否可逆。
4. **专精度量**：用 OLMoE 公开的 checkpoint，统计每层专家在不同 Pile 子集上的激活分布的 KL 散度，看专精是否集中在某些层（Mixtral 报告首尾层差异最大）。

### 📚 出处
- He. *Mixture of A Million Experts*. 2024. arXiv:2407.04153
- Berges et al. *Memory Layers at Scale*. 2024. arXiv:2412.09764
- Jin et al. *MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts*. ICLR 2025. arXiv:2410.07348
- Meituan LongCat Team. *LongCat-Flash Technical Report*. 2025. arXiv:2509.01322

---

## 📝 本章小结

| 关键点 | 你必须能说清楚 | 性质 |
|---|---|---|
| MoE 的本质 | 总参数随 $N$ 增长，计算随 $k$ 增长；条件计算 | 📄 + 🧮 |
| 路由梯度 | 只经过被选中专家的 $g_i$；未选中的专家无梯度 → 坍缩风险 | 🧮 |
| 容量 | $C = \frac{T}{N}\times\text{cf}$；超出的 token 走残差；dropless（MegaBlocks） | 📄 |
| Switch 平衡损失 | $\alpha N\sum f_iP_i$，$\alpha=10^{-2}$；梯度只经 $P_i$ | 📄 + 🧮 |
| Router z-loss | $(\log\sum e^{x})^2$，$c_z=10^{-3}$，稳定且不伤质量 | 📄 |
| Aux-loss-free | 偏置只管选择不管加权；按负载 $\pm\gamma$；V3 用 $\gamma=0.001$ | 📄 |
| 平衡的范围 | batch/global-batch 优于序列级；它是 aux-loss-free 优势的主要来源 | 📄 |
| 细粒度 | 组合数指数增长；粒度 scaling law 支持小专家 | 📄 + 🧮 |
| 共享专家 | DeepSeek/Kimi 用，OLMoE 消融无益，Qwen3 去掉 | 📄 + ❓ |
| Expert choice | 天然均衡但违反因果性 | 📄 + 🧮 |
| 专精 | 取决于平衡约束；不要说"一个专家一个领域" | 📄 + 💭 |
| Upcycling | 短训练划算，长训练可能被从头训练反超 | 📄 |

## ✅ 自测题

1. 写出 top-$k$ MoE 层的公式，并说明路由器的梯度从哪里来。为什么 top-1 时不能在选中的专家内重新归一化？
2. 写出 Switch 负载均衡损失，解释为什么要乘 $N$，以及为什么梯度只通过 $P_i$。
3. 解释 aux-loss-free 中偏置"只影响选择、不影响加权"的意义。DeepSeek-V3 §4.5.3 的对照实验说明了什么？
4. 用组合数说明细粒度专家的动机。专家能无限切细吗？
5. 共享专家有哪些支持和反对的证据？提出一个能区分两种解释的实验。
6. 为什么 expert choice 路由很难用于自回归解码？
7. "MoE 比 dense 好"这句话缺了什么前提？
8. **场景题**：你在 64 卡上训练一个 MoE，发现每步时间的方差很大、loss 正常但吞吐很低。请给出排查思路。

<details>
<summary>第 8 题参考思路</summary>

1. **看每层每个专家的 token 数分布**：若少数专家过载 → 负载不均。检查负载均衡损失系数、统计范围（micro-batch 还是 global-batch）、aux-loss-free 的 $\gamma$。
2. **看 all-to-all 的耗时占比**：若通信占大头 → 检查 EP 的放置（专家是否跨节点过多）、是否启用节点限制路由、通信与计算是否重叠。
3. **看是否在丢 token / padding**：容量因子太大会浪费计算，太小会丢 token；考虑 dropless 实现。
4. **看专家的矩阵乘形状**：专家太小、每个专家分到的 token 太少，GEMM 利用率低 → 考虑 grouped GEMM 或调整粒度。
5. **看数据**：某些领域的数据集中出现时，负载会随 batch 组成波动（这正是 batch 级平衡的代价之一）。

</details>

---

> **下一章预告**：第 22 章｜多模态——视觉编码器（ViT、CLIP、SigLIP）、连接器（线性投影、Q-Former、交叉注意力）、原生多模态与图像 token 化。
