# 第 1 章｜Transformer 深度解析

> **本章定位**：Transformer 是所有后续章节的地基。本章的目标不是背公式，而是对每个设计都能回答三个问题：**它解决了什么问题？证据是什么？哪些地方至今仍有争议？** 能做到这一点，面试中的追问和将来读论文都会轻松很多。

> **配套章节**：位置编码 → 第 2 章 / 归一化与激活 → 第 3 章 / KV Cache 与推理 → 第 12 章 / FlashAttention 与推理引擎 → 第 15 章

> **标记约定**（见 README）：📄 论文结论 / 🧮 可推导 / 💭 直觉或有争议的解释 / ❓ 开放问题

---

## 📌 本章导航

| 序号 | 题目 | 难度 | 类型 |
|---|---|---|---|
| Q1 | 描述 Transformer 整体架构 | ⭐⭐⭐ | 开场 |
| Q2 | 为什么现在的 LLM 大多是 Decoder-Only？ | ⭐⭐⭐⭐⭐ | 概念深度 |
| Q3 | 手推 Self-Attention 计算过程 | ⭐⭐⭐⭐⭐ | 必问 + 手撕 |
| Q4 | 为什么除以 √d_k？ | ⭐⭐⭐⭐ | 数学 |
| Q5 | Self-Attention 的复杂度 | ⭐⭐⭐⭐ | 系统 |
| Q6 | Multi-Head 的作用，为什么不用单头？ | ⭐⭐⭐⭐⭐ | 易答错 |
| Q7 | MHA → MQA → GQA → MLA 演化 | ⭐⭐⭐⭐⭐ | 加分 |
| Q8 | Causal Mask 原理与实现 | ⭐⭐⭐ | 实现细节 |
| Q9 | Encoder / Decoder / Enc-Dec 选型 | ⭐⭐⭐ | 应用 |
| Q10 | 研究前沿：Attention 之后是什么？ | ⭐⭐⭐⭐⭐ | 研究视角 |

---

## Q1：请描述 Transformer 的整体架构 ⭐⭐⭐

### 🎯 一句话标答

> 📄 Transformer（Vaswani et al. 2017）是一个以**注意力**为核心、去掉了循环和卷积的序列模型。原始版本是 Encoder-Decoder 结构，每层由**多头注意力 + 逐位置前馈网络（FFN）**组成，每个子层都包上**残差连接和 LayerNorm**。现代 LLM 大多只保留 Decoder，并做了 Pre-Norm、RMSNorm、SwiGLU、RoPE、GQA 等改动。

### 🗣️ 30 秒口语版

"原始 Transformer 是为机器翻译设计的 Encoder-Decoder 结构。Encoder 双向读入源句，Decoder 带因果 mask 自回归地生成目标句，并通过 cross-attention 读取 Encoder 的输出。原论文两边各 6 层。

每层有两个核心子层：多头自注意力负责 token 之间的信息交换，FFN 对每个位置单独做非线性变换。每个子层外面都有残差连接和 LayerNorm。

现在的主流 LLM，比如 GPT、LLaMA、Qwen、DeepSeek，都是 Decoder-Only，只保留带因果 mask 的自注意力和 FFN。和原论文相比，主要改动有：LN 从子层之后挪到之前（Pre-Norm），换成 RMSNorm；激活从 ReLU 换成 SwiGLU；位置编码从正弦换成 RoPE；注意力从 MHA 换成 GQA 或 MLA，以减小 KV Cache。"

### 📐 结构细节

**原始 Transformer（Post-Norm）**：

```
Encoder 层:  x = LN(x + SelfAttn(x))
             x = LN(x + FFN(x))

Decoder 层:  x = LN(x + MaskedSelfAttn(x))
             x = LN(x + CrossAttn(x, enc_out))
             x = LN(x + FFN(x))
```

**现代 Decoder-Only LLM（Pre-Norm，以 LLaMA 为例）**：

```
h = x + Attn(RMSNorm(x))          # 因果自注意力，GQA + RoPE
x = h + FFN(RMSNorm(h))           # SwiGLU
... 重复 L 层 ...
logits = LMHead(RMSNorm(x))       # Pre-Norm 需要最后再做一次归一化
```

> ⚠️ 不要把两种写法混在一张图里。原论文是 Post-Norm；Pre-Norm 是后来的改动（第 3 章 Q4）。

**原论文 base 模型的超参**（📄 Vaswani et al. 2017, Table 3）：$d_{model}=512$，$h=8$，$d_k=d_v=64$，$d_{ff}=2048$，$N=6$。

**参数量的粗算**（忽略 norm 和 bias，标准 MHA + $d_{ff}=4d$）：

| 子层 | 参数量 |
|---|---|
| Attention（$W_Q, W_K, W_V, W_O$） | $4d^2$ |
| FFN（两层，$d_{ff}=4d$） | $8d^2$ |
| 每层合计 | $12d^2$ |

🧮 所以 FFN 约占每层参数的 2/3。总参数量约为 $12Ld^2$ 加上词表嵌入。这个公式能用来快速估算模型规模，例如 GPT-3 175B：$L=96, d=12288$，$12 \times 96 \times 12288^2 \approx 174\text{B}$。用了 GQA 后 attention 参数会更少，SwiGLU 取 $d_{ff}=\frac{8}{3}d$ 时 FFN 仍约为 $8d^2$（第 3 章 Q6）。

### 🔥 高频追问

**Q：Transformer 为什么能取代 RNN？**

A：📄 原论文从三个角度对比了 self-attention、RNN 和 CNN（Vaswani et al. 2017, Table 1）：
- **每层计算量**：self-attention 为 $O(n^2 d)$，RNN 为 $O(n d^2)$。
- **串行操作数**：self-attention 为 $O(1)$，RNN 为 $O(n)$。这决定了训练能否并行。
- **任意两位置间的最长路径**：self-attention 为 $O(1)$，RNN 为 $O(n)$。路径越短，长程依赖越容易学。

代价是对序列长度的二次复杂度（Q5）。

**Q：FFN 是干什么的？只有 Attention 不行吗？**

A：两个角度：
- 📄 **理论**：Dong, Cordonnier, Loukas 2021（*Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth*）证明，去掉残差连接和 MLP 的纯注意力网络，输出会随深度**双指数地**收敛到秩为 1 的矩阵，也就是所有 token 的表示趋于相同。残差连接和 MLP 能抑制这种退化。
- 📄 **可解释性**：Geva et al. 2021（*Transformer Feed-Forward Layers Are Key-Value Memories*）把 FFN 的第一层看成 key、第二层看成 value，发现很多 key 对应可解释的文本模式。后续的知识编辑工作（如 ROME, Meng et al. 2022）也把事实性知识定位在中间层的 FFN。

💭 一个常用的比喻是"Attention 负责 token 之间的通信，FFN 负责每个 token 内部的计算"。

**Q：层数和宽度怎么权衡？"深而窄"一定更好吗？**

A：📄 不一定。Kaplan et al. 2020 发现，在固定（非嵌入）参数量下，**只要形状不太极端，loss 对深宽比的依赖很弱**：深宽比在 40 倍的范围内变化，loss 只变化几个百分点。所以"深而窄一定更好"不是一个可靠的说法。实践中层数还受流水线并行切分、推理延迟（层数越多，串行步骤越多）和训练稳定性的影响。

💭 关于"浅层学语法、深层学语义"：📄 Tenney et al. 2019（*BERT Rediscovers the Classical NLP Pipeline*）用探针发现，BERT 中词性、句法、语义角色、指代等信息的峰值大致按经典 NLP 流水线的顺序分布在不同层。但这是针对 BERT 的探针结果，不要把它说成所有 LLM 的普遍规律。

### ⚠️ 常见陷阱

1. 区分"原始 Transformer"（Enc-Dec、Post-Norm、ReLU、正弦位置编码）和"现代 LLM"（Decoder-Only、Pre-Norm、SwiGLU、RoPE、GQA）。
2. 能说出 $12Ld^2$ 的参数量估算。
3. FFN 是每层都有的中间模块，不是输出层。

### 📚 出处
- Vaswani et al. *Attention Is All You Need*. NeurIPS 2017. arXiv:1706.03762
- Dong et al. *Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth*. ICML 2021. arXiv:2103.03404
- Geva et al. *Transformer Feed-Forward Layers Are Key-Value Memories*. EMNLP 2021. arXiv:2012.14913
- Meng et al. *Locating and Editing Factual Associations in GPT*（ROME）. NeurIPS 2022. arXiv:2202.05262
- Kaplan et al. *Scaling Laws for Neural Language Models*. 2020. arXiv:2001.08361
- Tenney et al. *BERT Rediscovers the Classical NLP Pipeline*. ACL 2019. arXiv:1905.05950
- Jay Alammar. *The Illustrated Transformer*（图解入门）

---

## Q2：为什么现在的 LLM 大多是 Decoder-Only？ ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 没有一个单一的"证明"。比较有说服力的理由是：**目标简单统一**（只做下一个 token 预测，任何文本都是训练数据）；📄 **在纯无监督预训练后零样本泛化最好**（Wang et al. 2022 的对照实验）；**推理工程简单**（一套 KV Cache，便于复用前缀）；以及 GPT 系列跑通之后形成的**生态惯性**。但这个结论有条件，Encoder-Decoder 在某些设置下更好，2025 年也有 T5Gemma 这样的回归尝试。

### 🗣️ 30 秒口语版

"我会分成证据和解释两部分说。

证据方面，最直接的是 BigScience 2022 年的对照实验：同样约 5B 参数、约 170B token，比较因果 decoder、非因果 decoder 和 encoder-decoder。结论是，**只做无监督预训练时，因果 decoder + 自回归目标的零样本泛化最好**；但如果再加多任务指令微调，非因果结构加 MLM 目标反而最好。所以 Decoder-Only 的优势是有条件的。

解释方面：第一，目标统一，任何文本都能直接训练，所有任务都能写成续写；第二，推理简单，只有一套 KV Cache，prefix caching、投机解码都容易做；第三，工程和生态惯性，GPT-3 之后 scaling law、训练框架、推理引擎都围绕 decoder-only 优化。

还有一个理论层面的说法：苏剑林认为双向注意力矩阵容易出现低秩问题，而因果 mask 下的注意力矩阵是下三角的，对角线为正，所以一定满秩，表达能力可能更强。不过这只是一种解释，没有被广泛验证。"

### 📄 对照实验：Wang et al. 2022（BigScience）

| 设置 | 结论 |
|---|---|
| 只做无监督预训练，测零样本 | **因果 decoder + 自回归语言建模**最好 |
| 预训练 + 多任务提示微调，测零样本 | **非因果 decoder + MLM** 最好 |
| 结构转换 | 预训练好的模型可以在两种结构之间高效地互相适配 |

这篇论文告诉你：**"Decoder-Only 最好"依赖于训练流程**。这是一个比"因为 GPT 成功了"更有力的回答。

### 💡 常见理由逐条审视

**1. 目标统一、数据无限** —— 成立。任何文本都可以直接用于下一个 token 预测；分类、翻译、问答都能写成续写（GPT-2/GPT-3 的核心论点）。

**2. 监督信号密度** —— 部分成立，要说清楚。
- 因果 LM 中每个位置都计算 loss；BERT 的 MLM 只对约 15% 被 mask 的位置计算 loss。
- 📄 ELECTRA（Clark et al. 2020）正是针对 MLM 只从少量 token 学习这一低效问题，改为对所有 token 判断是否被替换。
- 💭 但"6 倍信号"只是数 token 的粗算：MLM 每个预测位置能看到双向上下文，信息量不同，不能直接按位置数比较。

**3. In-Context Learning** —— 要准确表述。
- 📄 GPT-3（Brown et al. 2020）展示了少样本 in-context learning，并且整体上随模型规模增大而增强。
- ❌ 旧版写的"GPT-3 发现超过 60B 后 zero/few-shot 突然涌现"没有出处。"涌现能力"的说法来自 📄 Wei et al. 2022（*Emergent Abilities of Large Language Models*），而 📄 Schaeffer et al. 2023（*Are Emergent Abilities of Large Language Models a Mirage?*，NeurIPS 2023）认为很多"突然涌现"是不连续评测指标造成的假象。面试时把两边都讲出来，体现你对证据的判断力。
- Encoder-Decoder 同样能做 in-context learning，只是格式上不如把示例直接拼进一个序列自然。

**4. 推理工程** —— 成立。
- Decoder-Only 只有一套自注意力 KV Cache；Enc-Dec 需要先跑 encoder，decoder 还要维护 self-attention 和 cross-attention 两套 cache。
- 多轮对话中，新输入直接接在已有序列后面，前缀的 KV 可以复用（prefix caching，→ 第 12、15 章）。

**5. Scaling Law** —— 📄 Kaplan 2020 和 Chinchilla（Hoffmann et al. 2022）的实验都基于 decoder-only Transformer。但这说明的是"研究资源集中在这条路线上"，不能推出 Enc-Dec 不能 scale（T5、UL2 都有大规模版本）。

**6. 低秩论证** —— 💭 苏剑林《为什么现在的 LLM 都是 Decoder-only 的架构？》（科学空间，2023）：
- 🧮 因果注意力矩阵 $A$ 是下三角矩阵，对角元 $A_{ii} = \text{softmax}(\cdot)_{ii} > 0$，所以 $\det A = \prod_i A_{ii} > 0$，**一定满秩**。
- 双向注意力矩阵 $\text{softmax}(QK^T/\sqrt{d})$ 来自秩不超过 $d_k$ 的 $QK^T$，在 $n \gg d_k$ 时可能出现低秩瓶颈（另见 Q6 的 Bhojanapalli et al. 2020）。
- ❓ 这是一个有启发性的解释，但"满秩是否等于更好"没有被严格验证。

### 🔥 高频追问

**Q：BERT 是不是过时了？**

A：在生成任务上基本退出了，但在"只需要理解、不需要生成"的场景还在用，比如分类、序列标注、轻量级重排序。📄 2024 年的 ModernBERT（Warner et al.）对 encoder 做了现代化改造（RoPE、更长上下文、更大数据），说明这条线仍在更新。

**Q：Embedding 模型是不是都用 Encoder？**

A：❌ 旧版说"Embedding 模型大多是 Encoder（BGE、E5）"已经过时。早期主流确实是 BERT 类 encoder，但从 2024 年起，基于 decoder LLM 的 embedding 模型在 MTEB 榜单上表现突出，例如 E5-Mistral（Wang et al. 2024）、NV-Embed、gte-Qwen、Qwen3-Embedding。常见做法是取最后一个 token 的表示，或者把因果 mask 改成双向后再做对比学习（📄 LLM2Vec, BehnamGhader et al. 2024）。

**Q：Encoder-Decoder 还有前途吗？**

A：📄 2025 年 Google 发布了 **T5Gemma**：用预训练好的 decoder-only Gemma 2 初始化 encoder 和 decoder，再用 UL2 或 PrefixLM 目标继续训练，并支持不同大小的 encoder 和 decoder 组合（如 9B encoder + 2B decoder）。Google 称其在质量与推理开销之间取得了更好的平衡；随后又推出了基于 Gemma 3、支持多模态的 T5Gemma 2。这说明 Enc-Dec 在"输入长、输出短"的场景（摘要、理解后生成）仍有竞争力。Whisper 等语音模型也一直是 Enc-Dec。

### ⚠️ 常见陷阱

- 不要只答"因为 GPT 跑通了"：这是结果，不是原因。
- 不要说"GPT-3 在 60B 处突然涌现"。
- 不要把 Decoder-Only 的优势说成无条件的：Wang et al. 2022 证明了它依赖训练流程。

### 📚 出处
- Wang et al. *What Language Model Architecture and Pretraining Objective Works Best for Zero-Shot Generalization?* ICML 2022. arXiv:2204.05832
- Brown et al. *Language Models are Few-Shot Learners*（GPT-3）. NeurIPS 2020. arXiv:2005.14165
- Wei et al. *Emergent Abilities of Large Language Models*. TMLR 2022. arXiv:2206.07682
- Schaeffer et al. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS 2023. arXiv:2304.15004
- Clark et al. *ELECTRA*. ICLR 2020. arXiv:2003.10555
- Hoffmann et al. *Training Compute-Optimal Large Language Models*（Chinchilla）. 2022. arXiv:2203.15556
- 苏剑林.《为什么现在的 LLM 都是 Decoder-only 的架构？》科学空间, 2023. https://kexue.fm/archives/9529
- Warner et al. *ModernBERT*. 2024. arXiv:2412.13663
- BehnamGhader et al. *LLM2Vec*. COLM 2024. arXiv:2404.05961
- Google. *T5Gemma* 博客（2025-07）：https://developers.googleblog.com/en/t5gemma/

---

## Q3：请详细推导 Self-Attention 的计算过程 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 每个 token 投影出 Query、Key、Value；用 Q 和所有 K 的点积（缩放后）经过按行 softmax 得到注意力权重，再对 V 加权求和：$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$。

### 🗣️ 30 秒口语版

"直觉上，self-attention 让每个 token 回头看序列中的所有 token，按相关性加权汇总信息。

每个 token 的表示会被投影成三个向量：Query 表示'我要找什么'，Key 表示'我能被怎样匹配'，Value 表示'我要传递的内容'。用 Q 和所有 K 做点积得到相关性分数，除以 √d_k 缩放，按行做 softmax 变成权重，再用权重对 V 加权求和，就得到这个 token 的新表示。"

### 📐 推导

**Step 0**：输入 $X \in \mathbb{R}^{n \times d_{model}}$。

**Step 1**：线性投影，$W^Q, W^K \in \mathbb{R}^{d_{model} \times d_k}$，$W^V \in \mathbb{R}^{d_{model} \times d_v}$：
$$Q = XW^Q,\quad K = XW^K,\quad V = XW^V$$

**Step 2**：分数矩阵
$$S = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times n}$$
$S_{ij}$ 表示位置 $i$ 对位置 $j$ 的关注程度（缩放原因见 Q4）。

**Step 3**：按行 softmax
$$A_{ij} = \frac{\exp(S_{ij})}{\sum_{k} \exp(S_{ik})}$$
每一行之和为 1。

**Step 4**：加权聚合
$$\text{Output}_i = \sum_j A_{ij} V_j, \qquad \text{Output} = AV \in \mathbb{R}^{n \times d_v}$$

🧮 **一个有用的改写**：$S = X W^Q (W^K)^T X^T / \sqrt{d_k}$。中间的 $W^Q (W^K)^T \in \mathbb{R}^{d_{model} \times d_{model}}$ 是一个**秩不超过 $d_k$ 的双线性形式**。这说明：
- Q、K 分开参数化，等价于学习一个低秩的、**非对称**的相似度矩阵；
- 这个低秩结构也是 MLA 中"权重吸收"技巧的基础（Q7）。

### 💻 手撕代码

```python
import math
import torch
import torch.nn.functional as F

def self_attention(X, W_Q, W_K, W_V, mask=None):
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V                     # (..., n, d_k / d_v)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))  # (..., n, n)
    if mask is not None:                                    # mask: True = 允许关注
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)                        # 按行归一化
    return attn @ V
```

实际工程中直接用 `F.scaled_dot_product_attention(q, k, v, is_causal=True)`，它会自动选择 FlashAttention 等高效 kernel。

**实现要点**：
1. `transpose(-2, -1)` 只转置最后两维，支持任意 batch / head 维度。
2. mask 必须加在 softmax **之前**的 logit 上。
3. softmax 在最后一维（key 维）上做。
4. 数值稳定：softmax 的标准实现会先减去每行最大值，PyTorch 已内置。

### 🔥 高频追问

**Q：为什么要 Q、K、V 三个矩阵？直接用 $XX^T$ 不行吗？**

A：
- 🧮 $XX^T$ 是**对称**的，$i$ 对 $j$ 的分数等于 $j$ 对 $i$ 的分数，无法表示有方向的关系（例如"代词找它指代的名词"）。$XW^Q(W^K)^TX^T$ 一般不对称。
- 🧮 $XX^T$ 的对角线 $\|x_i\|^2$ 通常最大，每个 token 会倾向于关注自己。
- V 单独投影，让"用什么来匹配"（Q、K 子空间）和"传递什么内容"（V 子空间）解耦。

**Q：Softmax 能换成别的吗？**

A：可以，这是一个活跃的研究方向：
- **线性注意力**：📄 Katharopoulos et al. 2020（*Transformers are RNNs*）用核函数特征映射 $\phi(q)^T\phi(k)$ 代替 $\exp(q^Tk)$，利用矩阵乘法结合律把复杂度降到 $O(n)$，并且可以写成 RNN 形式。📄 Performer（Choromanski et al. 2020）用随机特征近似 softmax 核。
- 📄 **Linformer**（Wang et al. 2020）不是核方法，而是把 K、V 沿序列维投影到固定长度，属于**低秩近似**。
- 📄 **Sigmoid attention**（Ramapuram et al. 2024, Apple）用逐元素 sigmoid 代替 softmax，并分析了需要的偏置项和稳定性条件。
- 近两年更主流的是**线性注意力 / 状态空间模型与 softmax 注意力的混合架构**（见 Q10）。

**Q：怎么从理论上理解 attention？**

A：📄 Tsai et al. 2019（*Transformer Dissection*）把 attention 看成**核平滑器**（kernel smoother）：输出是 value 的加权平均，权重由 query 和 key 之间的核函数决定。这个视角把 attention 和经典的 Nadaraya-Watson 核回归联系了起来，也自然引出了各种核近似方法。

### ⚠️ 常见陷阱

1. 先缩放再 softmax，顺序不能反。
2. mask 加在 softmax 之前。
3. softmax 按行（每个 query）做。
4. 在 MHA 中 $d_k = d_{model}/h$，不要和 $d_{model}$ 混淆。

### 📚 出处
- Vaswani et al. 2017, §3.2
- Katharopoulos et al. *Transformers are RNNs*. ICML 2020. arXiv:2006.16236
- Choromanski et al. *Rethinking Attention with Performers*. ICLR 2021. arXiv:2009.14794
- Wang et al. *Linformer*. 2020. arXiv:2006.04768
- Ramapuram et al. *Theory, Analysis, and Best Practices for Sigmoid Self-Attention*. 2024. arXiv:2409.04431
- Tsai et al. *Transformer Dissection*. EMNLP 2019. arXiv:1908.11775

---

## Q4：为什么 Self-Attention 要除以 √d_k？ ⭐⭐⭐⭐

### 🎯 一句话标答

> 🧮 若 $q, k$ 的各分量独立、均值 0、方差 1，则 $q \cdot k$ 的方差为 $d_k$。不缩放时 logit 的尺度随 $\sqrt{d_k}$ 增长，softmax 会趋向 one-hot，梯度变得很小。除以 $\sqrt{d_k}$ 把方差拉回 1。📄 原论文的措辞是"我们**怀疑**大 $d_k$ 会让点积变大，把 softmax 推入梯度极小的区域"，这是一个合理的启发式，不是严格定理。

### 🗣️ 30 秒口语版

"假设 q 和 k 的每一维独立、均值 0、方差 1，点积是 d_k 个乘积之和，方差就是 d_k。d_k = 128 时，标准差约为 11，logit 之间的差距很容易达到几十，softmax 几乎变成 one-hot。

softmax 饱和时，雅可比矩阵 diag(p) − ppᵀ 接近零矩阵，梯度很难传回去。除以 √d_k 后方差变回 1，softmax 保持在比较平滑的区域。

需要补充的是，原论文自己用的是'我们怀疑'这个说法，而且训练过程中 q 和 k 不再独立，所以 μP 理论会建议用 1/d_k 而不是 1/√d_k。"

### 🧮 推导

设 $q_i, k_i$ 相互独立，$\mathbb{E}[q_i]=\mathbb{E}[k_i]=0$，$\text{Var}(q_i)=\text{Var}(k_i)=1$：
$$\mathbb{E}[q_ik_i] = 0, \qquad \text{Var}(q_ik_i) = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1$$
各分量独立，所以
$$\text{Var}(q\cdot k) = \sum_{i=1}^{d_k}\text{Var}(q_ik_i) = d_k \quad\Rightarrow\quad \text{Var}\left(\frac{q\cdot k}{\sqrt{d_k}}\right) = 1$$

🧮 **softmax 饱和为什么会导致梯度消失**：令 $p = \text{softmax}(s)$，则
$$\frac{\partial p}{\partial s} = \text{diag}(p) - pp^T$$
当 $p$ 接近 one-hot（$p_j \approx 1$，其余 $\approx 0$）时，这个矩阵的每一项都接近 0。

📄 原论文第 3.2.1 节的脚注给出了上面的方差计算，正文用的是"We suspect that..."。

### 🔥 高频追问

**Q：为什么是 $\sqrt{d_k}$，不是 $d_k$？**

A：🧮 在"各分量独立"的假设下，目标是让 logit 的**标准差**保持为常数，所以除以标准差 $\sqrt{d_k}$。如果除以 $d_k$，logit 的标准差变成 $1/\sqrt{d_k}$，分布会过于平坦。

📄 **但这个假设在训练中不一定成立。** μP（Yang et al. 2022, *Tensor Programs V*）的论点是：训练后 $q$ 和 $k$ 会变得相关（对齐），此时点积按 $d_k$ 而不是 $\sqrt{d_k}$ 增长，所以应使用 $1/d_k$ 缩放，才能让超参数在不同宽度间迁移。Cerebras-GPT 等采用 μP 的工作使用了 $1/d_{head}$。在**固定 head 维度**（如 128）、只扩展头数的常见设置下，两者只差一个常数因子。

**Q：Q、K 的分布不满足假设怎么办？**

A：初始化时大致满足，训练过程中会偏离。大模型训练中 Q、K 的范数可能持续增长，导致 attention logit 过大、训练发散。这正是 **QK-Norm** 要解决的问题（第 3 章 Q8）：直接对 Q、K 做归一化，不再依赖初始化来保证尺度。

**Q：$1/\sqrt{d_k}$ 和温度是什么关系？**

A：$1/\sqrt{d_k}$ 就是一个固定的温度系数。有的工作把它换成可学习的标量（如 Henry et al. 2020 的 QK-Norm 中的 $g$）。长上下文外推时，也有方法根据长度调整 attention 温度（例如 YaRN 中的温度修正，→ 第 2 章）。

### ⚠️ 常见陷阱

- 要说出推导的前提（独立、零均值、单位方差），并知道这个前提在训练中会被打破。
- 不要说"√d_k 有严格证明是最优的"：原论文自己说的是"suspect"。

### 📚 出处
- Vaswani et al. 2017, §3.2.1 及脚注
- Yang et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*. 2022. arXiv:2203.03466
- Dey et al. *Cerebras-GPT*. 2023. arXiv:2304.03208

---

## Q5：Self-Attention 的计算复杂度是多少？ ⭐⭐⭐⭐

### 🎯 一句话标答

> 序列长度 $n$、隐藏维度 $d$ 时，单层 attention 的时间复杂度为 $O(nd^2 + n^2d)$（投影 + 分数与加权），朴素实现需要 $O(n^2)$ 的额外显存存放注意力矩阵。FlashAttention 不改变 FLOPs，但通过分块计算把额外显存降到 $O(n)$，并大幅减少显存读写。

### 📐 分解（单头视角，$d_k = d$）

| 步骤 | 操作 | FLOPs（乘加记 2） | 中间结果 |
|---|---|---|---|
| QKV 投影 | $(n,d)\times(d,3d)$ | $6nd^2$ | $O(nd)$ |
| $QK^T$ | $(n,d)\times(d,n)$ | $2n^2d$ | $O(n^2)$ |
| softmax | 逐行 | $O(n^2)$ | $O(n^2)$ |
| $AV$ | $(n,n)\times(n,d)$ | $2n^2d$ | $O(nd)$ |
| 输出投影 | $(n,d)\times(d,d)$ | $2nd^2$ | $O(nd)$ |

🧮 合计约 $8nd^2 + 4n^2d$。所以 **$n > 2d$ 时注意力部分开始主导**。例如 $d=4096$ 时，序列超过约 8K token，$n^2$ 项才超过投影项。如果再算上 FFN 的 $16nd^2$（$d_{ff}=4d$），整层中 attention 的二次项占主导需要更长的序列。这就是为什么在常见的 4K–8K 训练长度下，**大部分计算其实在矩阵投影和 FFN 上**。

> ⚠️ 加上多头后结论不变：$h$ 个头，每头 $d/h$ 维，二次项合计仍为 $4n^2d$。

### 🔥 高频追问

**Q：FlashAttention 降低了什么？**

A：📄 Dao et al. 2022：FlashAttention 是 **IO 感知**的精确注意力算法。它把 Q、K、V 分块加载到 GPU 的片上 SRAM，用 online softmax 逐块累积结果，**从不在显存（HBM）中写出完整的 $n\times n$ 矩阵**。
- FLOPs：不变（反向传播时为了省显存会重算一部分，FLOPs 略有增加）。
- 额外显存：从 $O(n^2)$ 降到 $O(n)$。
- 速度：因为 attention 主要受显存带宽限制，减少 HBM 读写带来了明显的实际加速。
- 后续：FlashAttention-2（更好的并行划分）、FlashAttention-3（利用 Hopper 架构的异步和 FP8）。

**Q：怎么把 $O(n^2)$ 降下来？**

A：几条路线：
- **稀疏注意力**：📄 Longformer（滑窗 + 全局 token）、BigBird（滑窗 + 全局 + 随机），复杂度 $O(n)$。现代 LLM 中常见的是**滑动窗口与全局注意力交替**（如 Mistral 7B 的滑窗、Gemma 2/3 的局部-全局交替）。
- **线性注意力 / SSM**：$O(n)$，并且推理时状态大小固定（Q10）。
- **可训练的稀疏注意力**：📄 DeepSeek 的 NSA（Native Sparse Attention, 2025）在训练时就学习稀疏模式。
- **FlashAttention**：不改变算法复杂度，只优化实现。

**Q：带 KV Cache 的推理复杂度是多少？**

A：生成第 $t$ 个 token 时，新 token 的 query 只需和 $t$ 个缓存的 K、V 计算：每步 attention 为 $O(td)$，生成 $n$ 个 token 共 $O(n^2d)$。更关键的是 decode 阶段每步都要**读完整个 KV Cache**，所以是**显存带宽受限**的（→ 第 12 章）。

### ⚠️ 常见陷阱

- 区分 FLOPs 和显存：FlashAttention 不减少 FLOPs。
- 区分训练（并行处理整条序列）和推理（逐 token、受带宽限制）。
- 不要以为短序列时 attention 是主要开销。

### 📚 出处
- Dao et al. *FlashAttention*. NeurIPS 2022. arXiv:2205.14135；*FlashAttention-2*. arXiv:2307.08691；Shah et al. *FlashAttention-3*. arXiv:2407.08608
- Beltagy et al. *Longformer*. arXiv:2004.05150；Zaheer et al. *BigBird*. arXiv:2007.14062
- Yuan et al. *Native Sparse Attention*. ACL 2025. arXiv:2502.11089
- Tay et al. *Efficient Transformers: A Survey*. arXiv:2009.06732

---

## Q6：Multi-Head Attention 的作用，为什么不用单头？ ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 📄 原论文的说法是：多头让模型"**能同时关注来自不同表示子空间、不同位置的信息**"，单头会因为平均化而抑制这种能力。在把 $d_{model}$ 均分给各头的设置下，参数量和 FLOPs 与单头相同。但它不是完全免费的：每头维度变小会带来**低秩瓶颈**，而且训练好的模型中很多头是冗余的。

### 🗣️ 30 秒口语版

"一个 softmax 只能产生一种加权分布。如果一个位置既要关注相邻的词，又要关注远处的指代对象，单头只能把这些需求平均在一起。多头把 d_model 拆成 h 份，每份独立做一次 attention，再拼接起来投影回去，这样不同的头可以学习不同的关注模式。

把维度均分时，参数量和计算量都和单头一样。不过这里有代价：每头维度变成 d/h，注意力矩阵的秩被限制在 d/h 以内，Bhojanapalli 等人 2020 年专门分析过这个低秩瓶颈。另外，Voita 和 Michel 在 2019 年都发现，训练好的模型可以剪掉大量的头而性能下降很小，说明多头之间有大量冗余。"

### 📐 公式

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O, \qquad \text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$

$W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{model}\times d_k}$，$d_k = d_{model}/h$，$W^O \in \mathbb{R}^{d_{model}\times d_{model}}$。

🧮 参数量：Q/K/V 投影共 $3 \cdot h \cdot d_{model} \cdot d_k = 3d_{model}^2$，输出投影 $d_{model}^2$，与单头完全相同。

### 📄 证据：多头到底学到了什么？

- **头的分工**：📄 Voita et al. 2019（*Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned*）在机器翻译模型中识别出几类重要的头：关注相邻位置的"位置头"、对应句法依存的"句法头"、关注低频词的头。用可微的剪枝方法去掉大部分头之后，BLEU 下降很小。
- **冗余**：📄 Michel et al. 2019（*Are Sixteen Heads Really Better than One?*）发现，测试时很多层可以只保留一个头而性能基本不变；但训练时多头仍然有用。
- **低秩瓶颈**：📄 Bhojanapalli et al. 2020（*Low-Rank Bottleneck in Multi-head Attention Models*）指出，当每头维度 $d_k < n$ 时，每个头的注意力矩阵表达能力受限；他们建议让 head 维度和序列长度相关，而不是固定为 $d_{model}/h$。
- **机制可解释性**：📄 Olsson et al. 2022（Anthropic, *In-context Learning and Induction Heads*）发现了"归纳头"（induction head）：两层注意力配合实现"在前文中找到当前 token 上次出现的位置，复制它后面的 token"。他们提供证据表明，归纳头可能是 in-context learning 的重要机制之一。

### 🔥 高频追问

**Q：头数是不是越多越好？head 维度一般取多少？**

A：不是。头数增加意味着每头维度减小，低秩瓶颈更严重。常见的 head 维度：

| 模型 | head 维度 |
|---|---|
| GPT-2 | 64 |
| GPT-3 175B（$d=12288$，96 头） | 128 |
| LLaMA 系列 | 128 |
| Gemma（7B）/ Gemma 2 部分尺寸 | 256 |
| DeepSeek-V3（MLA） | Q/K 为 128 + 64（RoPE 部分），V 为 128 |

所以"大于 128 是浪费"并不成立。head 维度和头数是需要根据模型规模和推理开销（KV Cache 大小与头数成正比）来权衡的超参数。

**Q：MQA 只有一个 KV 头，是不是退化成了单头？**

A：不是。MQA 仍然有 $h$ 个 query 头，每个头用自己的 query 在同一组 K、V 上产生**不同的注意力分布**。损失的是 K、V 表示的多样性，query 的多样性保留了。

### ⚠️ 常见陷阱

1. 说清楚"不同子空间、不同位置"，并能给出 Voita / Olsson 的证据。
2. 不要说"多头是免费的午餐"：有低秩瓶颈，也有冗余。
3. 注意力可视化不等于解释：📄 Jain & Wallace 2019（*Attention is not Explanation*）和 Wiegreffe & Pinter 2019（*Attention is not not Explanation*）对此有过争论。

### 📚 出处
- Voita et al. ACL 2019. arXiv:1905.09418
- Michel et al. NeurIPS 2019. arXiv:1905.10650
- Bhojanapalli et al. ICML 2020. arXiv:2002.07028
- Olsson et al. *In-context Learning and Induction Heads*. Transformer Circuits Thread, 2022. arXiv:2209.11895
- Jain & Wallace. NAACL 2019. arXiv:1902.10186；Wiegreffe & Pinter. EMNLP 2019. arXiv:1908.04626

---

## Q7：MHA → MQA → GQA → MLA 的演化与对比 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 这条演化线的目标是**减小 KV Cache**，因为 decode 阶段每步都要读一遍 KV Cache，它决定了推理的带宽开销和可支持的并发量。MQA 让所有 query 头共享一组 KV；GQA 分组共享，是现在的主流；MLA（DeepSeek-V2）把 KV 压缩成一个低维潜在向量，推理时通过**权重吸收**直接在潜在空间中计算注意力。

### 🗣️ 30 秒口语版

"MHA 每个头都有自己的 K、V，KV Cache 和头数成正比。

MQA 是 Shazeer 2019 年提出的，所有 query 头共享一组 K、V，KV Cache 缩小为 1/h，但有一定的质量损失，训练也可能不稳定。PaLM 用了 MQA。

GQA 是 2023 年 Google 的工作，把 query 头分成 g 组，每组共享一组 KV，是 MHA 和 MQA 之间的插值。论文还提出可以用大约 5% 的原始预训练计算量，把 MHA checkpoint 'uptrain' 成 GQA。LLaMA-2 70B 用 64 个 query 头配 8 个 KV 头，现在 GQA 基本是标配。

MLA 是 DeepSeek-V2 提出的，思路不是减少头数，而是把每个 token 的 K、V 联合压缩成一个低维向量 c，只缓存 c。推理时把 K 的上投影矩阵吸收进 Q 的投影，把 V 的上投影吸收进输出投影，就可以直接在低维空间算注意力。难点是 RoPE 会破坏这种吸收，所以 DeepSeek 额外拆出一小段带 RoPE 的 key，所有头共享。DeepSeek-V2 报告说，和 DeepSeek 67B 相比，KV Cache 减少了 93.3%。"

### 📐 KV Cache 对比

设 $L=32$ 层、$n_h=32$ 个 query 头、$d_h=128$、序列长度 4096、FP16（2 字节）。每个 token 每层缓存的元素数和总量：

| 方法 | 每 token 每层缓存元素数 | 4096 token 总量 | 相对 MHA |
|---|---|---|---|
| MHA | $2 n_h d_h = 8192$ | 2 GiB | 100% |
| GQA（$g=8$） | $2 g d_h = 2048$ | 512 MiB | 25% |
| MQA | $2 d_h = 256$ | 64 MiB | 3.1% |
| MLA（DeepSeek-V2 维度） | $d_c + d_h^R = 512 + 64 = 576$ | 144 MiB | 7.0% |

🧮 计算方法：总字节数 = 元素数 × 2 字节 × 层数 × token 数。例如 MHA：$8192 \times 2 \times 32 \times 4096 = 2^{31}$ 字节 = 2 GiB。

> 注：MLA 这一行把 DeepSeek-V2 的 $d_c=512$、$d_h^R=64$ 套到了这个假想模型上，只用于比较量级。📄 DeepSeek-V2 报告中"减少 93.3%"是**相对于 DeepSeek 67B** 的比较。

### 🧮 MLA 推导（重点）

📄 DeepSeek-V2 的 MLA（省略 query 侧的压缩）：

$$c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}, \qquad k_t^{C} = W^{UK} c_t^{KV}, \qquad v_t^{C} = W^{UV} c_t^{KV}$$

推理时只缓存 $c_t^{KV}$。

**权重吸收**：对第 $i$ 个头，
$$q_{t,i}^T k_{s,i}^{C} = (W_i^{Q} h_t)^T (W_i^{UK} c_s^{KV}) = h_t^T \underbrace{(W_i^{Q})^T W_i^{UK}}_{\text{可预先合并}} c_s^{KV}$$
所以 query 可以直接投影到 $d_c$ 维的潜在空间，与缓存的 $c_s^{KV}$ 做点积，**不需要还原出完整的 K**。同理，$W^{UV}$ 可以吸收进输出投影 $W^O$。

**RoPE 带来的问题**：RoPE 在 $q$ 和 $k$ 上分别乘以和位置有关的旋转矩阵 $R_t, R_s$：
$$q_t^T R_t^T R_s k_s = h_t^T (W^Q)^T R_{s-t}\, W^{UK} c_s$$
$R_{s-t}$ 夹在 $W^Q$ 和 $W^{UK}$ 中间，并且随相对位置变化，所以两个矩阵**无法预先合并**。

**解耦 RoPE（Decoupled RoPE）**：额外引入一小段专门承载位置信息的向量：
- 每个头有一个 RoPE query $q_{t,i}^R$；
- 所有头**共享**一个 RoPE key $k_t^R = \text{RoPE}(W^{KR} h_t) \in \mathbb{R}^{d_h^R}$，它和 $c_t^{KV}$ 一起缓存；
- 最终分数是"内容部分"（经过吸收的低秩点积）和"位置部分"（RoPE 点积）之和。

详细讨论见第 2 章 Q8。

📄 **MLA 的效果**：DeepSeek-V2 报告的消融实验显示，MLA 的效果**优于** MHA，同时 KV Cache 远小于 MHA。

> 💡 这也解释了第 3 章 Q8 的问题：QK-Norm 需要对完整的 $k$ 做逐头归一化，这会破坏权重吸收，所以 MLA 不能直接使用 QK-Norm。

### 🔥 高频追问

**Q：为什么 GQA 的质量损失比 MQA 小？**

A：📄 Ainslie et al. 2023 报告，GQA 的质量接近 MHA，速度接近 MQA。💭 直观上，$g$ 组 KV 的表示容量是 MQA 的 $g$ 倍，不同组的 query 可以匹配不同的 K、V 子空间。

**Q：MLA 为什么能在压缩的同时保持质量？**

A：💭 MLA 不是简单地减少 KV 的维度，而是**缓存一个共享的低维潜在向量，再为每个头单独上投影**。上投影后每个头的 K、V 仍然不同，所以它比 GQA 更灵活。📄 TransMLA（Meng et al. 2025）从理论上论证：在 KV Cache 大小相同时，GQA 都可以表示成 MLA，反之则不成立，所以 MLA 的表达能力更强；他们还提出了把 GQA 模型转换成 MLA 的方法。

**Q：除了 MLA，还有哪些减小 KV Cache 的方向？**

A：
- **跨层共享 KV**：📄 Cross-Layer Attention（Brandon et al. 2024）让相邻层共享 KV；📄 YOCO（Sun et al. 2024）只缓存一次全局 KV。
- **KV Cache 量化**：INT8 / INT4 / FP8（→ 第 13 章）。
- **KV 淘汰与压缩**：H2O、StreamingLLM 的 attention sink、SnapKV 等（→ 第 12 章）。
- **滑动窗口 / 局部注意力**：部分层只缓存最近的 token（Mistral、Gemma 2/3）。
- **混合架构**：大部分层用固定大小状态的线性注意力，只有少数层用完整注意力（Q10）。

### ⚠️ 常见陷阱

1. MQA 是 $g=1$ 的 GQA，MHA 是 $g=h$ 的 GQA。
2. 讲 MLA 必须讲清**权重吸收**和**RoPE 为什么会破坏吸收**，只说"压缩到低维"是不够的。
3. "减少 93.3%"是相对 DeepSeek 67B 的数字，不是相对任意 MHA 模型。

### 📚 出处
- Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need*（MQA）. 2019. arXiv:1911.02150
- Ainslie et al. *GQA*. EMNLP 2023. arXiv:2305.13245
- DeepSeek-AI. *DeepSeek-V2*. 2024. arXiv:2405.04434
- Meng et al. *TransMLA: Multi-Head Latent Attention Is All You Need*. 2025. arXiv:2502.07864
- Brandon et al. *Reducing Transformer Key-Value Cache Size with Cross-Layer Attention*. 2024. arXiv:2405.12981
- Sun et al. *You Only Cache Once*（YOCO）. 2024. arXiv:2405.05254

---

## Q8：Causal Mask 是什么？为什么需要？怎么实现？ ⭐⭐⭐

### 🎯 一句话标答

> Causal mask 在 softmax 之前把位置 $i$ 对所有 $j > i$ 的 logit 设为 $-\infty$，即**只允许关注下三角（含对角线）**。它让一次前向就能并行计算所有位置的下一个 token 预测（teacher forcing），同时保证每个位置看不到未来，训练和自回归推理的条件一致。

### 📐 示例（n=4）

```
加 mask 后的 logit（行 = query，列 = key）：
       t1    t2    t3    t4
  t1 [s11   -∞    -∞    -∞ ]
  t2 [s21   s22   -∞    -∞ ]
  t3 [s31   s32   s33   -∞ ]
  t4 [s41   s42   s43   s44]
```

softmax 后，$-\infty$ 的位置权重为 0。

### 💻 实现

```python
n = scores.size(-1)
allowed = torch.ones(n, n, dtype=torch.bool, device=scores.device).tril()
scores = scores.masked_fill(~allowed, float("-inf"))
attn = scores.softmax(dim=-1)

# 实际工程：直接用 is_causal，让 kernel 自己处理
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### 🔥 高频追问

**Q：训练时一半的注意力矩阵都被 mask 掉了，是不是浪费？**

A：朴素实现会先算完整个矩阵再 mask，确实浪费约一半。📄 但 FlashAttention 等分块 kernel 会**直接跳过完全位于上三角的块**，所以 causal attention 的实际计算量约为非 causal 的一半。

**Q：推理时还需要 causal mask 吗？**

A：要分阶段：
- **Decode 阶段**（每次只算一个新 token）：新 token 的 query 本来就只能看到已缓存的历史，不需要 mask。
- **Prefill 阶段**（一次性处理整个 prompt）：和训练一样，**需要** causal mask。
- **投机解码的验证、chunked prefill、多序列打包**：一次处理多个新 token，需要 causal mask，打包时还需要防止不同序列互相关注的块对角 mask（→ 第 14、15 章）。

> ❌ 旧版说"推理时严格来说不需要 mask"，只对 decode 阶段成立。

**Q：padding mask 和 causal mask 怎么配合？会有什么坑？**

A：两者取逻辑与。一个常见的坑：如果某一行**所有位置都被 mask**（例如左 padding 时 padding 位置自己作为 query），softmax 的分母为 0，会产生 NaN。常见的处理方式：用一个很大的负数代替 $-\infty$，或者确保每个 query 至少能看到自己。

**Q：序列打包（packing）训练时 mask 怎么处理？**

A：把多条短样本拼成一条长序列以减少 padding 时，如果只用普通的 causal mask，后面的样本会关注到前面不相关的样本。严格的做法是使用**块对角的 causal mask**（每个样本内部是 causal 的，样本之间完全屏蔽），FlashAttention 的变长接口（`cu_seqlens`）支持这种方式；位置编码也要在每个样本处重新从 0 开始。是否严格隔离对效果的影响，不同工作的结论并不一致。

### ⚠️ 常见陷阱

1. mask 加在 logit 上，不是加在概率上。
2. 推理分 prefill 和 decode 两个阶段，不能一概而论。
3. 全屏蔽的行会产生 NaN。

---

## Q9：Encoder-Only / Decoder-Only / Encoder-Decoder 分别适合什么任务？ ⭐⭐⭐

### 🎯 一句话标答

> Encoder-Only（双向）适合只需要理解的任务，如分类、序列标注和轻量检索；Decoder-Only（因果）适合通用生成和对话，是当前 LLM 的主流；Encoder-Decoder 适合"读入一段输入、生成一段输出"且两者差异较大的任务，如翻译、语音识别和摘要。三者之间的界限正在变得模糊：decoder LLM 也能做 embedding，decoder 也能改造成 encoder-decoder。

### 📐 对比

| 维度 | Encoder-Only | Decoder-Only | Encoder-Decoder |
|---|---|---|---|
| 注意力 | 双向 | 因果 | Encoder 双向 + Decoder 因果 + cross-attn |
| 典型目标 | MLM | 下一个 token 预测 | Span corruption（T5）/ 去噪（BART）/ UL2 混合 |
| 代表模型 | BERT、RoBERTa、ModernBERT | GPT、LLaMA、Qwen、DeepSeek | T5、BART、Whisper、T5Gemma |
| 典型用途 | 分类、NER、重排序 | 生成、对话、Agent、Embedding（新趋势） | 翻译、ASR、摘要 |

还有一个中间形态：**Prefix LM**（非因果 decoder），对输入前缀用双向注意力，对输出部分用因果注意力。📄 UL2（Tay et al. 2022）把多种去噪目标统一起来，同时支持这几种模式。

### 🔥 高频追问

**Q：多模态模型是 Encoder-Decoder 吗？**

A：要区分：
- 📄 **Flamingo**（Alayrac et al. 2022）：在冻结的 decoder LM 中插入 gated cross-attention 层去读取视觉特征，形式上接近 encoder-decoder。
- 📄 **LLaVA**（Liu et al. 2023）、Qwen-VL 等目前主流的 VLM：视觉编码器（如 ViT）的输出经过投影层，**作为 token 直接拼接到 decoder-only LLM 的输入序列中**，没有 cross-attention。这不是经典的 Encoder-Decoder，而是"视觉编码器 + Decoder-Only"。
- **Whisper**：标准的 Encoder-Decoder。

所以"多模态让 Encoder-Decoder 复兴"不准确，更准确的说法是"多模态普遍需要模态编码器，但和语言模型的连接方式有多种"。

**Q：如果让你做一个翻译系统，选哪种？**

A：取决于约束：
- 需要覆盖大量语言、追求通用质量：通用的 decoder-only LLM 往往已经很强，还能顺便处理术语、风格等指令。
- 对延迟和成本敏感、语言对固定：专用的较小 Encoder-Decoder 模型（例如 📄 Meta 的 NLLB-200, 2022）性价比更高。
- 语音翻译：一般是语音 encoder + 文本 decoder。

> ❌ 旧版说"DeepL 基于 mBART/NLLB"没有出处，已删除。

### 📚 出处
- Tay et al. *UL2: Unifying Language Learning Paradigms*. ICLR 2023. arXiv:2205.05131
- Alayrac et al. *Flamingo*. NeurIPS 2022. arXiv:2204.14198
- Liu et al. *Visual Instruction Tuning*（LLaVA）. NeurIPS 2023. arXiv:2304.08485
- Radford et al. *Robust Speech Recognition via Large-Scale Weak Supervision*（Whisper）. 2022. arXiv:2212.04356
- NLLB Team. *No Language Left Behind*. 2022. arXiv:2207.04672

---

## Q10：研究前沿：Attention 之后是什么？ ⭐⭐⭐⭐⭐

> 不是面试必答题。目的是帮你看清"哪些问题还没有解决"。

### 方向 1：线性复杂度的序列模型与混合架构

- 📄 **Mamba**（Gu & Dao 2023）：选择性状态空间模型，参数随输入变化，训练时可并行扫描，推理时状态大小固定。📄 **Mamba-2**（Dao & Gu 2024）提出"状态空间对偶"（SSD），说明 SSM 和一类线性注意力在数学上是等价的。
- 📄 **门控线性注意力与 Delta 规则**：GLA（Yang et al. 2023）、DeltaNet（Yang et al. 2024）、Gated DeltaNet（Yang, Kautz, Hatamizadeh, ICLR 2025）。Delta 规则把状态更新看成在线学习一个键值映射，"先删除旧值再写入新值"，改善了线性注意力的检索能力。
- **工业界的混合架构**（大部分层用线性注意力，少数层保留完整注意力）：📄 MiniMax-01 的 Lightning Attention（2025）、Qwen3-Next（Gated DeltaNet + gated attention 混合）、Kimi Linear 等。
- ❓ **开放问题**：线性/状态空间模型在**精确检索和复制**类任务上有理论上的劣势（固定大小的状态容量有限，📄 参见 Jelassi et al. 2024, *Repeat After Me: Transformers are Better than State Space Models at Copying*）。混合比例应该如何设计？哪些层应该保留完整注意力？

### 方向 2：注意力本身的改进

- 📄 **Differential Transformer**（Ye et al. 2024, Microsoft）：用两个 softmax 注意力图的差来消除"注意力噪声"。
- 📄 **Gated Attention**（Qwen 团队 2025）：在注意力输出后加一个与输入相关的 sigmoid 门，报告可以缓解 attention sink 并提升训练稳定性（Qwen3-Next 采用）。
- 📄 **Attention sink**（Xiao et al. 2023, StreamingLLM）：很多头会把大量注意力分配给第一个 token。这是 softmax"权重必须和为 1"带来的副作用，还是模型有意学出的功能？这个问题仍在讨论中（另见第 3 章 Q9）。

### 方向 3：机制可解释性

- 📄 Anthropic 的 *A Mathematical Framework for Transformer Circuits*（Elhage et al. 2021）把 Transformer 分解为对残差流的读写操作，从电路的角度分析注意力头的组合方式（如 QK 电路、OV 电路），归纳头就是在这个框架下发现的。
- ❓ **开放问题**：这些在小模型上发现的电路，在大模型中是否以同样的形式存在？

### 🧪 动手验证思路

1. 实现一个带 causal mask 的多头注意力，和 `F.scaled_dot_product_attention` 的输出对比，误差应在 1e-5 以内。
2. 在一个 2 层的 attention-only 小模型上训练重复序列任务，观察是否出现归纳头（参照 Olsson et al. 2022）。
3. 实现 MLA 的权重吸收：分别用"还原完整 K"和"在潜在空间计算"两种方式算注意力分数，验证两者数值一致；再加入 RoPE，观察吸收为什么不再成立。

### 📚 出处
- Gu & Dao. *Mamba*. 2023. arXiv:2312.00752；Dao & Gu. *Transformers are SSMs*（Mamba-2）. ICML 2024. arXiv:2405.21060
- Yang et al. *Gated Linear Attention Transformers*. ICML 2024. arXiv:2312.06635
- Yang et al. *Gated Delta Networks*. ICLR 2025. arXiv:2412.06464
- MiniMax. *MiniMax-01*. 2025. arXiv:2501.08313
- Jelassi et al. *Repeat After Me*. ICML 2024. arXiv:2402.01032
- Ye et al. *Differential Transformer*. ICLR 2025. arXiv:2410.05258
- Qiu et al.（Qwen）. *Gated Attention for Large Language Models*. 2025. arXiv:2505.06708
- Xiao et al. *Efficient Streaming Language Models with Attention Sinks*. ICLR 2024. arXiv:2309.17453
- Elhage et al. *A Mathematical Framework for Transformer Circuits*. Transformer Circuits Thread, 2021

---

## 📝 本章小结

| 关键点 | 你必须能脱口而出 | 性质 |
|---|---|---|
| 架构 | 原始：Enc-Dec + Post-Norm；现代：Decoder-Only + Pre-Norm + RMSNorm + SwiGLU + RoPE + GQA | 📄 |
| 参数量 | 每层约 $12d^2$（attn $4d^2$ + FFN $8d^2$） | 🧮 |
| Decoder-Only | 纯无监督预训练后零样本最好（Wang 2022），但依赖训练流程 | 📄 |
| Attention 公式 | $\text{softmax}(QK^T/\sqrt{d_k})V$，按行 softmax | 🧮 |
| √d_k | 独立假设下方差为 $d_k$；原论文说"suspect"；μP 用 $1/d$ | 🧮 + 📄 |
| 复杂度 | $8nd^2 + 4n^2d$；FlashAttention 不减 FLOPs，减显存读写 | 🧮 + 📄 |
| 多头 | 不同子空间/位置；有低秩瓶颈和冗余；归纳头 | 📄 |
| GQA / MLA | GQA 分组共享 KV；MLA 缓存低维 latent + 权重吸收 + 解耦 RoPE | 🧮 + 📄 |
| Causal mask | 下三角允许；prefill 需要、decode 不需要；全屏蔽行会 NaN | 🧮 |

## ✅ 自测题

1. 写出原始 Transformer 和 LLaMA 的单层结构，并指出 5 处不同。
2. 用 $12Ld^2$ 估算 LLaMA-2 7B（$L=32, d=4096$）的非嵌入参数量，并解释和 7B 的差距来自哪里（提示：SwiGLU 的 $d_{ff}=11008$、词表嵌入）。
3. 推导 $\text{Var}(q\cdot k)=d_k$，并说明在什么情况下这个结论不成立。
4. 写出 MLA 的权重吸收公式，并说明 RoPE 为什么会破坏吸收。
5. Wang et al. 2022 的对照实验得出了什么结论？它对"为什么 LLM 是 Decoder-Only"的回答有什么影响？
6. 实现一个支持 padding mask + causal mask 的多头注意力，并处理全屏蔽行的 NaN 问题。

---

> **下一章预告**：第 2 章｜位置编码专题——从 Sinusoidal 到 RoPE，再到长上下文外推的 PI / NTK / YaRN，以及 MLA 中解耦 RoPE 的完整推导。
