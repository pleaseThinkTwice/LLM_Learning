# 第 1 章｜Transformer 深度解析

> **本章定位**：LLM 面试的"地基题"，几乎 100% 会被问到。一面用来开场试水，二面挖数学细节，三面看你能不能讲清楚演化脉络。读完本章，你应该能做到：**遇到任何 Transformer 相关的题，张口就有 3 个层次的答案**——30 秒应付一面、3 分钟支撑二面、追问能聊到论文细节。

> **配套章节**：位置编码 → 第 2 章 / 归一化与激活 → 第 3 章 / 手撕 Attention 代码 → 第 9 章。本章只讲架构与注意力。

---

## 📌 本章导航：高频题分布

| 序号 | 题目 | 难度 | 频率（字节/阿里/腾讯/Meta） | 类型 |
|---|---|---|---|---|
| Q1 | 描述 Transformer 整体架构 | ⭐⭐⭐ | 8 / 5 / 4 / 3 | 一面开场 |
| Q2 | 为什么 LLM 都是 Decoder-Only？ | ⭐⭐⭐⭐⭐ | 6 / 2 / 1 / 4 | 概念深度 |
| Q3 | 手推 Self-Attention 计算过程 | ⭐⭐⭐⭐⭐ | **11 / 8 / 6 / 5** | 必问 |
| Q4 | 为什么除以 √d_k？ | ⭐⭐⭐⭐ | 7 / 3 / 2 / 2 | 数学敏感度 |
| Q5 | Self-Attention 复杂度 | ⭐⭐⭐⭐ | 5 / 3 / 2 / 4 | 系统能力 |
| Q6 | Multi-Head 的作用，为什么不用单头？ | ⭐⭐⭐⭐⭐ | 6 / 3 / 2 / 3 | 易答错 |
| Q7 | MHA → MQA → GQA → MLA 演化 | ⭐⭐⭐⭐⭐ | 7 / 4 / 3 / 2 | 加分题 |
| Q8 | Causal Mask 原理与实现 | ⭐⭐⭐ | 4 / 2 / 2 / 1 | 实现细节 |
| Q9 | Encoder/Decoder/Enc-Dec 选型 | ⭐⭐⭐ | 2 / 3 / 2 / 1 | 应用题 |

---

## Q1：请描述 Transformer 的整体架构 ⭐⭐⭐

### 🎯 一句话标答

> Transformer 是一个**完全基于注意力机制**的序列建模架构，由 Encoder 和 Decoder 两半组成，核心组件是 Self-Attention + FFN，靠残差连接和 LayerNorm 撑住深度。

### 🗣️ 30 秒口语版（一面开场用）

"好的。Transformer 整体可以看成两半——Encoder 负责把输入读懂，Decoder 负责把输出写出来。每一半都是同一种层堆 N 次，原论文是 6 层，现在的大模型一般 30 到 100 层都有。

每一层里头其实就两个核心模块：一个是 **多头自注意力**，让序列中的 token 之间互相'看见';另一个是 **前馈网络 FFN**，做特征变换。这两个模块外面都套着一层残差连接和 LayerNorm，保证训练能稳住。

Decoder 比 Encoder 多一个东西——**Cross-Attention**，用来让 Decoder 在生成时去看 Encoder 的输出。另外 Decoder 的 Self-Attention 是带 Causal Mask 的，保证生成时只能看到前面，不能偷看后面。

不过现在主流的 LLM——GPT、LLaMA、DeepSeek 这些——已经把 Encoder 砍掉了，只保留 Decoder，这个我们后面可以展开讲。"

### 📐 3 分钟深度版（二面用）

**先讲结构，再讲信息流**。

**结构上**，原始 Transformer 是这样的：

```
                ┌──────────────────┐
   输入序列  →  │   Encoder × N    │  → 上下文表示
                └──────────────────┘
                         ↓ (cross attention)
                ┌──────────────────┐
   目标序列  →  │   Decoder × N    │  → 输出概率分布
                └──────────────────┘
                         ↓
                       Softmax
                         ↓
                      下一个 token
```

**Encoder 单层结构**：
```
x → MHA(LN(x)) + x → FFN(LN(x)) + x  (Pre-Norm 写法)
```

**Decoder 单层结构**多一步 Cross-Attention：
```
x → Masked-MHA → +x → LN → Cross-Attn(with Enc out) → +x → LN → FFN → +x
```

**信息流的关键点有四个**：

第一，**Embedding + 位置编码**。Token 通过 Embedding 矩阵变成 d 维向量，但纯 Embedding 是位置无关的，所以要加上位置信息——原论文用的是正弦余弦，现在主流用 RoPE（这个第 2 章细讲）。

第二，**Self-Attention 是核心创新**。它替代了 RNN 的"逐步传递信息"，改成"全连接式地一次性建模所有 pair-wise 关系"，这就是为什么能并行。

第三，**残差 + Norm**。这两个不是装饰品，是必需品。残差解决梯度消失，让 100 层的网络也能训；Norm 让激活值分布稳定，避免训崩。

第四，**FFN 占了大头参数**。FFN 中间维度通常是 4d，所以一层里 FFN 的参数量大概是 Attention 的两倍。LLaMA 用 SwiGLU 后中间维度变成 8d/3，参数量等价。

### 🔥 高频追问 Top 3

**Q：Transformer 和 RNN 比，优势是什么？为什么能取代 RNN？**

A：核心就两点——**并行 + 长程依赖**。RNN 是按时间步串行的，没法并行；而且远距离信息要经过很多步传递，容易梯度消失。Self-Attention 一次性看完整个序列，任意两个 token 之间都是 O(1) 的距离，训练能并行，长距离建模也好得多。代价是 O(n²) 的复杂度，所以长序列又成了问题，这是后来 Flash Attention 这些工作要解决的。

**Q：FFN 是干什么的？只有 Attention 不行吗？**

A：Attention 本质是**信息的混合**——让 token 之间互相交换信息。但单纯的混合没有非线性变换能力，所以 FFN 负责**逐 token 的特征变换**。可以理解为 Attention 横向通信、FFN 纵向加工，两者交替才能让网络学到复杂表示。有论文做过消融实验，去掉 FFN 性能大幅下降。

**Q：为什么是堆 N 层而不是设计得更复杂？**

A：这是深度学习的通用经验——同样参数量下，**深而窄** 通常比 **浅而宽** 效果好。多层能学到层次化的表示：浅层学语法、中层学语义、深层学逻辑。但层数也不是越多越好，会有边际收益递减和训练不稳定的问题，所以实践中是个平衡。

### ⚠️ 常见陷阱

1. **混淆"Transformer 架构"和"现代 LLM 架构"**：原始 Transformer 是 Encoder-Decoder（用于翻译），现代 LLM 是 Decoder-Only。很多人答题时直接套原论文图，但面试官想听的是"你知道演化"。
2. **忘了说残差和 Norm**：这两个是训练能成功的关键，没讲到会被扣分。
3. **把 FFN 说成是"分类头"**：FFN 是中间层，不是输出层。分类头在最顶上，是 Linear + Softmax。

### 🏢 大厂偏好

- **字节**：常追问"如果让你改进 Transformer 你会怎么做"，看你对架构的理解深度
- **阿里**：喜欢从"Transformer 怎么用于推荐场景"切入
- **Meta**：喜欢追问 LLaMA 相比原始 Transformer 改了什么（答：Pre-Norm、RMSNorm、SwiGLU、RoPE、GQA）

### 📚 延伸阅读
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) §3.1-§3.3 必读
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 经典图解

---

## Q2：为什么现在的 LLM 都是 Decoder-Only？ ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 因为 Decoder-Only 在**统一性、Scaling 友好度、推理效率、数据利用率**这四个维度上全面胜出，而 GPT 系列实证证明了它能做几乎所有任务。

### 🗣️ 30 秒口语版

"这个问题挺有意思的。我觉得可以从四个角度回答。

第一，**任务统一**。Decoder-Only 把所有任务——分类、翻译、问答、生成——都统一成'下一个 token 预测'，模型不用为不同任务设计不同的头，非常优雅。

第二，**Scaling 更友好**。OpenAI 的 Scaling Law 论文是基于 Decoder-Only 做的，工业界投钱时倾向于一条被验证过的路线，所以越投越多，飞轮就转起来了。

第三，**推理效率高**。Decoder 天然是因果的，KV Cache 可以直接用；Encoder-Decoder 还要做 Cross-Attention，工程上麻烦。

第四，**数据利用率高**。Decoder-Only 训练时每个 token 都是监督信号；BERT 那种 Masked LM 只有 15% 的 token 在产生梯度，效率差了一截。

加起来，Decoder-Only 就成了现在的事实标准。"

### 📐 3 分钟深度版

如果让我说得更细一点，每条都能展开：

**1. 任务统一性（最根本的原因）**

GPT-3 之前的 NLP 是"百花齐放"——分类用 BERT、生成用 BART、问答用 T5，每种任务都有自己的范式。GPT-3 证明了一件事：**只要模型够大，next-token prediction 这一个范式就能 cover 所有任务**。

```
情感分类  → 输入"评价：这个手机很好用，情感是：" → 模型续写"正面"
翻译     → 输入"中文：你好  英文：" → 模型续写"Hello"
代码生成 → 输入"# 求斐波那契数列\ndef fib(n):" → 模型续写函数体
```

任务统一带来的好处是：训练数据无限大（互联网上的任何文本都能用），不用专门标注。

**2. Scaling Law 的偏爱**

OpenAI 2020 年的 Kaplan 论文和后来的 Chinchilla 都是基于 Decoder-Only 推导的。这些规律告诉我们："参数、数据、算力按特定比例增加，loss 会平滑下降"——这给工业界投资提供了确定性。一旦确定性建立，资源就会涌入，进一步放大优势。

Encoder-Decoder 架构（如 T5）当然也能 scaling，但因为 GPT 路线先跑通，生态、工具链、数据集都集中在 Decoder-Only，后来者很难翻盘。

**3. In-Context Learning 涌现**

GPT-3 论文里发现：当 Decoder-Only 模型超过 60B 参数后，**zero-shot 和 few-shot 能力突然涌现**。给几个例子就能学会新任务，不用微调。这个能力在 Encoder-Decoder 上也有，但 Decoder-Only 的形式更自然——把示例和目标拼成一个序列即可。

**4. 推理工程的友好**

这点经常被忽略，但工业落地非常重要。Decoder-Only 是**单向因果**的，所以：
- KV Cache 实现简单（每次只 append，不重算）
- 投机解码、并行解码这些加速技术天然适配
- Streaming 输出体验好

Encoder-Decoder 推理时要先跑 Encoder 一次，再 Decoder 自回归生成，工程上更复杂，KV Cache 也分成两部分。

**5. 数据利用率**

| 训练范式 | 监督信号密度 |
|---|---|
| BERT (MLM) | 只有 15% 的 token 被 mask，其他都"白送" |
| Decoder-Only (CLM) | 每个 token 都贡献 loss |

同样训练 1 万亿 token，Decoder-Only 拿到的有效梯度信号是 BERT 的 6 倍多。

### 🔥 高频追问 Top 3

**Q：那 BERT 是不是就过时了？**

A：不完全是。BERT 这种 Encoder-Only 模型在**需要双向理解、不需要生成**的任务上还是有优势的，比如稠密检索（Dense Retrieval）、序列标注、句子表示。现在的 embedding 模型大多还是 Encoder 架构，比如 BGE、E5。但纯生成任务的赛道，BERT 确实已经退场了。

**Q：那 Encoder-Decoder 还有存在的必要吗？**

A：在**输入输出严格对齐**的任务上还是有价值的，比如机器翻译、文档摘要、语音识别。Whisper 就是 Encoder-Decoder，因为音频和文字是两种模态，分开处理更合理。但通用对话场景，Decoder-Only 已经赢了。

**Q：Decoder-Only 有什么缺点吗？**

A：有的。最明显的是**双向理解的天然劣势**——比如完形填空类任务，Encoder 的双向 Attention 理论上更适合。但实践中 Decoder-Only 靠规模和数据补回来了。另外，Decoder-Only 对**长输入的理解**不如 Encoder（因为只能从左到右看），所以现在很多检索增强系统会先用 Encoder 模型做 embedding。

### ⚠️ 常见陷阱

- 不要只答"因为 GPT 跑通了"——这是结果，不是原因。要从架构层面给出技术理由。
- 不要忘了提 In-Context Learning，这是 Decoder-Only 最神奇的涌现能力。
- 别把 Decoder-Only 说得太完美——双向理解的劣势要承认，体现你的客观性。

### 🏢 大厂偏好

- **字节**：会追问"如果让你设计下一代架构，你怎么改 Decoder-Only？"——往 MoE、SSM、混合架构方向答
- **Anthropic / Meta**：偏爱讨论 BERT 退场的本质原因，看你对训练范式的理解

### 📚 延伸阅读
- [GPT-3 Paper](https://arxiv.org/abs/2005.14165) - In-context learning 的起源
- [What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?](https://arxiv.org/abs/2204.05832) - 直接对比三种架构

---

## Q3：请详细推导 Self-Attention 的计算过程 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> Self-Attention = **"用 Q 去查 K，得到权重；用权重对 V 加权求和"**，完整公式是 `softmax(QKᵀ/√d_k) · V`。

### 🗣️ 30 秒口语版

"好的，我从直觉到公式讲一下。

直觉上，Self-Attention 干的事情就是——对序列中的每一个 token，让它**回头看一遍其他所有 token，按相关性加权汇总信息**。

具体怎么做呢？每个 token 的 embedding 会被映射成三个东西：**Q 是 Query，代表'我要查什么'**；**K 是 Key，代表'我能被怎么查'**；**V 是 Value，代表'我的实际内容'**。

然后用 Q 和所有 K 做点积，得到一组相似度分数——分数越高代表越相关。这组分数除以 √d_k 做缩放，再过 Softmax 归一化成概率分布，最后用这个概率分布去加权 V，就得到了这个 token 的新表示。

公式就是这个：**softmax(QKᵀ/√d_k) · V**。"

### 📐 3 分钟深度版（含完整推导）

**Step 0：输入定义**

设输入序列长度为 n，每个 token 的 embedding 维度为 d_model：
$$X \in \mathbb{R}^{n \times d_{model}}$$

**Step 1：三次线性变换得到 Q, K, V**

引入三个可学习的权重矩阵 $W^Q, W^K \in \mathbb{R}^{d_{model} \times d_k}$ 和 $W^V \in \mathbb{R}^{d_{model} \times d_v}$：

$$Q = XW^Q \in \mathbb{R}^{n \times d_k}$$
$$K = XW^K \in \mathbb{R}^{n \times d_k}$$
$$V = XW^V \in \mathbb{R}^{n \times d_v}$$

通常 $d_k = d_v = d_{model}$（在 MHA 里会被拆成多头）。

**Step 2：计算注意力分数矩阵**

$$S = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times n}$$

矩阵 S 的 $(i, j)$ 位置表示 **token i 对 token j 的关注程度**（除以 √d_k 的原因看 Q4）。

**Step 3：Softmax 归一化（按行做）**

$$A = \text{softmax}(S), \quad A_{ij} = \frac{\exp(S_{ij})}{\sum_{k=1}^{n} \exp(S_{ik})}$$

注意是**按行 softmax**——每一行加起来等于 1，代表 token i 对所有 token 的关注权重分布。

**Step 4：加权聚合 V**

$$\text{Output} = AV \in \mathbb{R}^{n \times d_v}$$

每一行 Output[i] = $\sum_j A_{ij} \cdot V_j$，就是按权重把所有 token 的 value 加起来。

**最终公式合一**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 💻 关键代码片段

```python
import torch
import torch.nn.functional as F
import math

def self_attention(X, W_Q, W_K, W_V, mask=None):
    Q = X @ W_Q                                    # (n, d_k)
    K = X @ W_K                                    # (n, d_k)
    V = X @ W_V                                    # (n, d_v)
    
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (n, n)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)               # (n, n)，按行归一化
    return attn @ V                                # (n, d_v)
```

**⚠️ 实现要点**（面试常考）：
1. `transpose(-2, -1)` 是把最后两维转置，**支持任意 batch 维度**
2. mask 一定要在 softmax **之前** 加，加在之后就无效了
3. softmax 必须 `dim=-1`，不能写错

### 🔥 高频追问 Top 3

**Q：为什么是 Q、K、V 三个矩阵而不是一个？直接 X·Xᵀ 不行吗？**

A：这是个好问题。直接用 X·Xᵀ 也能算相似度，但有两个致命问题：
- **表达能力受限**——一个 token 在"查询"和"被查"两种角色下，应该呈现不同的特征。比如 "is" 这个词作为 query 想找主语，作为 key 想被动词找到，两种角色的表示应该是不同的。Q、K 分开允许模型学到这种角色化的表示。
- **对称性问题**——X·Xᵀ 是对称矩阵，token A 对 B 的注意力 = B 对 A 的注意力，这显然不合理。引入 Q、K 后矩阵不再对称，能建模有向关系。
- **V 单独存在**是因为：注意力计算的是"该看谁"，但真正传递的内容可以不一样。比如计算时关注的是"语法相关性"，但传递的可能是"语义内容"。

**Q：Softmax 之前的注意力分数矩阵 QKᵀ 表达了什么？**

A：表达的是 token 之间的**未归一化相似度**——本质上是把每个 token 的 query 和所有 token 的 key 做内积，内积大说明它们在某个子空间里方向接近，相关性高。后面 softmax 把它转成概率分布，方便加权。

**Q：如果不用 Softmax，能不能用别的归一化方式？**

A：可以，但有取舍。Linear Attention（用 ReLU 之类）能把复杂度从 O(n²) 降到 O(n)，代价是性能略降；最近的工作如 Linformer、Performer 都在探索。但 Softmax 有几个独特优点：① 严格的概率解释；② 数值稳定（最大值减去）；③ 梯度性质好。所以目前主流还是 Softmax。

### ⚠️ 常见陷阱

1. **顺序写反**：必须先除以 √d_k 再做 softmax，反了就没意义
2. **mask 加在 softmax 之后**：那时候已经是概率了，乘以 0 不会让其他位置重新归一化
3. **忘了 softmax 是按行的**：按列就完全错了
4. **混淆 d_k 和 d_model**：在 MHA 里 d_k = d_model / n_heads

### 🏢 大厂偏好

- **字节**：必考的手撕题，会要求**白板写出来 + 解释维度变化**
- **阿里**：会追问"如果序列特别长怎么办"，引出 Flash Attention / 线性注意力
- **Meta**：喜欢追问"Attention 的本质是 kernel smoother，你怎么看"——这是个偏理论的角度

### 📚 延伸阅读
- 原论文 §3.2 - 注意力公式
- [Transformers from Scratch](https://e2eml.school/transformers.html) - 从零推导
- [Attention as Soft Dictionary Lookup](https://blog.research.google/2022/04/learning-from-weakly-supervised-data.html) - 把 attention 理解成软查字典

---

## Q4：为什么 Self-Attention 要除以 √d_k？ ⭐⭐⭐⭐

### 🎯 一句话标答

> 因为不除会让 QKᵀ 的方差随 d_k 线性增长，导致 softmax 进入饱和区，梯度趋近于零，**根本训不动**。

### 🗣️ 30 秒口语版

"这个问题其实是个**数值稳定性问题**。

假设 Q 和 K 的每个元素都是均值 0、方差 1 的独立分布，那它们点积 q·k 的方差就等于 d_k——维度越大，方差越大。如果 d_k 是 64，那点积的方差就是 64，标准差 8，意味着分数可能跑到 ±20、±30 这种很大的范围。

这时候 softmax 会怎么样？指数函数 e^x 在 x=20 和 x=10 之间差了一万倍，**softmax 会变成接近 one-hot**——一个位置接近 1，其他全接近 0。

后果就是**梯度消失**——softmax 在饱和区导数接近 0，反向传播时梯度传不下去，模型训不动。

除以 √d_k 之后，方差被拉回 1，softmax 分布平滑，梯度正常。"

### 📐 数学推导（二面会要求写）

**前提假设**：Q 和 K 的每个元素 $q_i, k_i$ 都是 i.i.d.，均值为 0，方差为 1。

**计算点积的方差**：

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

因为 $q_i, k_i$ 独立且均值为 0：
$$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i] \mathbb{E}[k_i] = 0$$
$$\text{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - 0 = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1 \cdot 1 = 1$$

由独立性：
$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

**所以除以 √d_k 后**：
$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

方差被拉回 1，达到稳定。

### 🔥 高频追问 Top 3

**Q：为什么是 √d_k 不是 d_k 或者 log(d_k)？**

A：因为我们要做的是**让标准差归一**，方差是平方关系，所以开根号。如果除以 d_k，标准差变成 1/√d_k，又太小了，分布会过于平坦，softmax 退化成均匀分布，模型学不到有用的注意力 pattern。

**Q：如果 Q、K 不是均值 0 方差 1 的分布呢？**

A：实际训练中靠初始化（Xavier/Kaiming）保证大致满足。如果用了不合适的初始化或者归一化策略，这个假设可能会被破坏。这也是为什么现代 LLM 都很重视 Pre-Norm 和初始化方案，确保 QK 的分布稳定。

**Q：除以 √d_k 是温度系数吗？**

A：可以这么理解！1/√d_k 本质上就是一个**固定的温度系数**。如果我们想要更尖锐的注意力分布，可以再除一个更大的数（相当于降温）；想要更平滑就乘一个大于 1 的数（升温）。一些工作就是这么做的，比如 logit lens 分析时会调节温度。

### ⚠️ 常见陷阱

- **不要只回答"数值稳定性"就完事**——面试官想听完整推导
- **不要把 √d_k 解释成"经验值"**——它有严格的数学推导
- **不要忘了独立性假设**——这是推导的前提

### 🏢 大厂偏好

- **字节**：必问，要求白板推导
- **微软 / Google**：会追问"如果不假设独立性，方差怎么算"——这是测概率论功底

### 📚 延伸阅读
- 原论文 §3.2.1 末尾的脚注

---

## Q5：Self-Attention 的计算复杂度是多少？ ⭐⭐⭐⭐

### 🎯 一句话标答

> 时间复杂度 **O(n²·d)**，空间复杂度 **O(n²)**，瓶颈在 n×n 的注意力矩阵。

### 🗣️ 30 秒口语版

"Self-Attention 的复杂度分两块看。

**时间上**，主要开销是两次大矩阵乘法：QKᵀ 是 (n,d)×(d,n) 得到 (n,n)，复杂度 O(n²d)；后面 A·V 又是 (n,n)×(n,d)，又是 O(n²d)。中间的 softmax 是 O(n²)。所以**总时间复杂度 O(n²d)**。

**空间上**，最大的开销是中间的 n×n 注意力矩阵。这是 Flash Attention 要解决的核心问题——它通过分块计算，避免完整存这个矩阵，把空间复杂度从 O(n²) 降到 O(n)。

序列变长时，n²d 的 n² 项很快主导。所以长上下文场景下，怎么优化这个二次复杂度是个核心问题，催生了 Linear Attention、Flash Attention、稀疏注意力等一系列工作。"

### 📐 详细分解

| 步骤 | 操作 | 时间复杂度 | 空间复杂度 |
|---|---|---|---|
| 1. 三次线性投影 | (n,d)×(d,d) | O(nd²) | O(nd) |
| 2. QKᵀ | (n,d)×(d,n) | O(n²d) | O(n²) ← 瓶颈 |
| 3. 除以 √d_k | 逐元素 | O(n²) | O(n²) |
| 4. Softmax | 逐行 | O(n²) | O(n²) |
| 5. A·V | (n,n)×(n,d) | O(n²d) | O(nd) |
| **合计** | | **O(n²d + nd²)** | **O(n²)** |

**何时 n²d 主导，何时 nd² 主导？**
- 当 **n > d** 时（长序列），n²d 主导
- 当 **n < d** 时（短序列），nd² 主导（线性投影开销大）

实际 LLM 一般是 n > d 或 n ≈ d，所以认为是 **O(n²d)**。

### 🔥 高频追问 Top 3

**Q：MHA 的复杂度和单头一样吗？**

A：**一样**。MHA 把 d 拆成 h 个 d/h 的头并行做，每个头是 O(n²·(d/h))，h 个头加起来还是 O(n²d)。所以多头免费提升了模型能力，没有额外计算开销。

**Q：怎么把这个 O(n²) 降下来？**

A：主流方法有三类——
- **稀疏注意力**：只计算部分位置，比如 Longformer 的滑窗、BigBird 的局部+全局，复杂度降到 O(n)
- **线性注意力**：用 kernel trick 把 softmax 拆解，比如 Linformer、Performer，复杂度 O(n)
- **Flash Attention**：不改算法，改实现——通过 IO-aware 的 tiling 把空间降到 O(n)，时间还是 O(n²d) 但常数小很多

**Q：为什么 KV Cache 之后 Decoder 推理是 O(n) 的？**

A：因为推理时每次只生成一个新 token，新 token 的 Q 只需要和缓存的所有 K、V 算注意力——是 1×n 的运算，而不是 n×n。所以单步推理是 O(n·d)，生成 n 个 token 总共 O(n²·d)。这里的"O(n)"指的是单步，不是总体。

### ⚠️ 常见陷阱

- **混淆训练和推理的复杂度**：训练是 O(n²d) per step（一次性算所有 token），推理（带 KV Cache）是 O(nd) per token。
- **忘了空间复杂度**：很多人只答时间，但工程上空间往往是真正的瓶颈。

### 🏢 大厂偏好

- **字节 / 阿里**：必问，会追问怎么优化
- **AI Infra 岗**：会要求详细到 FLOPs 量级的估算

### 📚 延伸阅读
- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)

---

## Q6：Multi-Head Attention 的作用，为什么不用单头？ ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 多头允许模型在**不同子空间并行学习不同类型的注意力模式**，参数量和计算量与单头相同，但表达能力更强。

### 🗣️ 30 秒口语版

"好问题。先讲个直觉——单头注意力只能学**一种**关注模式，但语言里其实有很多种关系：语法上的主谓关系、语义上的指代关系、长距离的话题关系。指望一个头同时学好所有这些，太难了。

多头的做法是：把 d_model 维拆成 h 份，每份是 d_model/h，然后**并行做 h 次注意力**。每个头在自己的子空间里学一种关系——有的头可能专门学相邻词关系，有的学远距离指代，有的学语法依存。最后拼起来。

关键是——**多头并不增加参数量和计算量**。因为每头维度变成 d/h，总维度还是 d，矩阵乘法的复杂度不变。所以多头是个'免费午餐'，凭白多了表达能力。

实证上，可视化注意力 pattern 确实能看到不同头学到了不同的模式。"

### 📐 公式与维度分析

**单头**：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
其中 $Q, K, V \in \mathbb{R}^{n \times d_{model}}$。

**多头**：
$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$

其中 $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{model} \times d_k}$，$d_k = d_{model}/h$。

**参数量对比**（设 d_model = 512，h = 8）：

| | 单头 | 多头 |
|---|---|---|
| Q/K/V 投影 | 3 × 512² = 786K | 8 × (3 × 512 × 64) = 786K ✓ 相同 |
| 输出投影 | 512² = 262K | 512² = 262K ✓ 相同 |

**复杂度**：每头 O(n²·d/h)，h 头加起来还是 O(n²d)。

### 💡 直觉理解

可以类比成 CNN 的**多通道**：
- 单头 Attention ≈ 单卷积核
- 多头 Attention ≈ 多卷积核，每个卷积核学一种特征模式

或者类比成**集成学习**：多个"弱的注意力"投票得到一个"强的表示"。

### 🔥 高频追问 Top 3

**Q：多头是不是越多越好？**

A：**不是**。头数太多每头维度太小，单头能力变弱；太少又失去多头的优势。**经验上 8、16、32 是常见值**，且 d_k 通常保持在 32-128 之间。最近也有研究表明，训练好的模型中**很多头是冗余的，可以剪掉**——比如 Voita 等人的 "Analyzing Multi-Head Self-Attention" 发现剪掉一半头性能不降。

**Q：MQA 把 KV 头变成 1 个，那不就回到单头了吗？**

A：**不一样**。MQA 仍然有多个 Q 头（h 个），只是它们共享同一组 K、V。所以 Q 还是能从 h 个不同子空间发起查询，只不过查的"内容"是同一个。这相当于"用 h 双不同的眼睛看同一本书"，损失了 K、V 的多样性，但保留了 Q 的多样性，所以效果只是略降。

**Q：为什么每头维度通常是 32-128？**

A：太小 (<32) 表达能力不够，QK 点积的"区分度"差；太大 (>128) 又会浪费——因为单头不需要那么强。32-128 是一个工程上的甜点。LLaMA 用 128，GPT-3 用 96，DeepSeek 用 128。

### ⚠️ 常见陷阱

1. **不要说"多头能学更多东西"——要说清楚是"不同子空间学不同模式"**
2. **不要忘了说"参数量不变"——这是关键的工程优势**
3. **可视化 attention map 是好的证据**，但要承认"可视化解释不完全可靠"，避免被反问

### 🏢 大厂偏好

- **字节 / 阿里**：经常追问"如果让你设计单头版的 LLM，怎么补回多头的能力？"——开放题，可以答 mixture of experts、动态路由
- **OpenAI / Anthropic**：会从理论角度追问"多头的数学解释"

### 📚 延伸阅读
- [Analyzing Multi-Head Self-Attention](https://arxiv.org/abs/1905.09418) - 多头剪枝
- [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650) - 头数实验

---

## Q7：MHA → MQA → GQA → MLA 的演化与对比 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 这条演化主线是**为了减小 KV Cache、加速推理**——MHA 每头独立 KV，MQA 所有头共享 1 组 KV，GQA 分组共享，MLA 把 KV 压缩到低维潜在空间。**核心权衡是质量 vs. 显存**。

### 🗣️ 30 秒口语版

"这是个能讲出深度的好题。

**MHA** 是原始版本——h 个 Q 头对应 h 个 K 头、h 个 V 头，独立完整。问题是 KV Cache 太大了，长上下文场景下显存爆炸。

**MQA** 是个激进方案——所有 Q 头共享 1 个 K 头、1 个 V 头。KV Cache 直接缩到 1/h，推理飞快。但代价是质量明显下降。代表模型 PaLM。

**GQA** 是个折中——把 h 个 Q 头分成 g 组，每组共享 1 个 K/V 头。KV Cache 是 MHA 的 g/h，质量基本无损。LLaMA-2 70B 用 8 个 KV 头对 64 个 Q 头，是个甜点。**现在是事实标准**。

**MLA** 是 DeepSeek-V2 的创新——不是减少头数，而是把 KV 整个**压缩到一个低维潜在空间**，推理时再投影回来。KV Cache 减小 93%，质量还能保持。这是目前最优雅的方案。

整体演化逻辑是：从'砍数量'到'压维度'，越来越聪明。"

### 📐 详细对比

```
                 ┌─────────────────────────────────────────┐
                 │              Q 头数 = h (都一样)          │
                 └─────────────────────────────────────────┘

MHA:   Q₁ Q₂ ... Q_h     K₁ K₂ ... K_h     V₁ V₂ ... V_h
       │  │       │       │  │       │       │  │       │
       └──┴───────┘       └──┴───────┘       └──┴───────┘
       (h 头)             (h 头独立)         (h 头独立)


MQA:   Q₁ Q₂ ... Q_h     K (单一)           V (单一)
       │  │       │       │                  │
       └──┴───────┘       └─ 所有 Q 共享 ──┘  └─ 所有 Q 共享 ──┘


GQA:   Q₁ Q₂ ... Q_h     K₁ K₂ ... K_g      V₁ V₂ ... V_g
       │  │       │       │                   │
       └──┘   ... 每组 Q 共享一组 KV (g < h)


MLA:   Q₁ Q₂ ... Q_h     ┌─────────┐         ┌─────────┐
       │  │       │       │ 低维 c_t │ ←压缩─→│ 低维 c_t │
       └──┴───────┘       └─────────┘         └─────────┘
                          推理时 c_t 再投影回 K、V
```

**详细参数表**（假设 h=32, d_head=128, L=32 层, s=4096 序列, FP16）：

| 方法 | KV 头数 | 单 token KV 维度 | KV Cache 总量 | 相对 MHA |
|---|---|---|---|---|
| MHA | 32 | 2 × 32 × 128 = 8192 | 2.0 GB | 100% |
| MQA | 1 | 2 × 1 × 128 = 256 | 64 MB | 3.1% |
| GQA (g=8) | 8 | 2 × 8 × 128 = 2048 | 512 MB | 25% |
| MLA | - | ~576（DeepSeek-V2） | ~140 MB | ~7% |

### 🔍 MLA 深度展开（DeepSeek 核心创新，重点）

MLA 不属于"减少头数"思路，是另起炉灶。核心思想是**低秩压缩**：

```
传统 KV：每个 token 存 K_t = X_t · W_K  和  V_t = X_t · W_V

MLA：每个 token 只存压缩向量 c_t = X_t · W_DKV  (低维, ~512)
     需要时再用 W_UK, W_UV 投影回 K、V
```

**KV Cache 公式**：
- MHA：2 × n_h × d_head 维
- MLA：d_c 维（约 4~5 倍 d_head，但不乘 n_h）

**额外加 RoPE 时的小麻烦**：因为 MLA 的 K 是动态投影出来的，没法直接乘 RoPE 矩阵。DeepSeek 的解法是把 K 拆成两部分——一部分参与压缩（c_t），另一部分独立加 RoPE，最后拼接。

### 🔥 高频追问 Top 3

**Q：MQA 质量下降为什么 GQA 不下降？**

A：因为 GQA 仍然保留了多组 KV，每组负责一部分 Q 头，**信息容量**比 MQA 大得多。MQA 是 32 个 Q 头共用 1 组 KV，相当于让 1 组 KV"伺候"32 种不同的查询，太挤了；GQA 是 4 个 Q 头一组共用，压力小得多。

**Q：MLA 为什么能保持质量？**

A：因为 MLA 不是在"剪枝"，而是在"重新参数化"。它假设 KV 在低维潜在空间里就有足够的表达力——这个假设通过实验验证是成立的。可以类比 LoRA——不是直接减少参数，而是用低秩分解表达高维变换。

**Q：能不能比 MLA 更优？**

A：理论上可以继续探索。最近有些方向：① 进一步压缩 d_c；② 在 KV Cache 上做量化（INT8、INT4），可以叠加；③ 跨层共享 KV，DeepSeek 在某些版本中尝试过。但 MLA + KV 量化已经接近物理极限。

### ⚠️ 常见陷阱

1. **不要把 MQA 和 GQA 混淆**——MQA 是 g=1 的极端版 GQA
2. **不要忘了 MLA 的 RoPE 兼容性问题**——这是面试官追问的高频点
3. **不要只讲原理不讲数字**——KV Cache 节省多少、相对 MHA 多少，要能信手拈来

### 🏢 大厂偏好

- **字节 / DeepSeek 出身的面试官**：必问 MLA，会要求画图
- **AI Infra 岗**：追问 KV Cache 具体计算和量化方案

### 📚 延伸阅读
- [MQA Paper](https://arxiv.org/abs/1911.02150)
- [GQA Paper](https://arxiv.org/abs/2305.13245)
- [DeepSeek-V2 Paper](https://arxiv.org/abs/2405.04434) - MLA 详细介绍

---

## Q8：Causal Mask 是什么？为什么需要？怎么实现？ ⭐⭐⭐

### 🎯 一句话标答

> Causal Mask 是个**上三角矩阵**，在 softmax 之前把"未来位置"的注意力分数设为 -∞，保证 Decoder 在训练时不会"偷看"后面的 token。

### 🗣️ 30 秒口语版

"Decoder-Only 的 LLM 是自回归生成的——预测第 t 个 token 时，只能看 1 到 t-1。但如果训练时不加约束，Self-Attention 会让每个 token 都看到序列里所有 token，包括未来的——这就是'信息泄露'，模型会作弊，推理时就崩了。

Causal Mask 就是干这件事的。它是个**上三角矩阵**，在注意力分数矩阵的'未来'位置上加一个 -∞，让 softmax 之后这些位置的权重变成 0。这样每个 token 真的就只能看到自己和前面的 token。

实现上很简单，PyTorch 一行 `torch.tril(torch.ones(n, n))` 就能造出来，然后用 `masked_fill(mask == 0, -inf)` 应用。"

### 📐 矩阵示例（n=4）

不加 mask 的注意力分数矩阵（每行是一个 token 对所有 token 的注意力）：
```
       t1   t2   t3   t4
  t1 [s11  s12  s13  s14]    ← t1 不应该看到 t2,t3,t4
  t2 [s21  s22  s23  s24]    ← t2 不应该看到 t3,t4
  t3 [s31  s32  s33  s34]    ← t3 不应该看到 t4
  t4 [s41  s42  s43  s44]
```

加 Causal Mask 后：
```
       t1   t2   t3   t4
  t1 [s11  -∞   -∞   -∞ ]
  t2 [s21  s22  -∞   -∞ ]
  t3 [s31  s32  s33  -∞ ]
  t4 [s41  s42  s43  s44]
```

经过 softmax 后，-∞ 位置变成 0，每个 token 只对自己和前面的 token 有非零注意力。

### 💻 PyTorch 实现

```python
import torch

def causal_mask(seq_len):
    """生成下三角形 mask，True 表示允许，False 表示屏蔽"""
    return torch.tril(torch.ones(seq_len, seq_len)).bool()

# 应用 mask
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (n, n)
mask = causal_mask(scores.size(-1))                  # (n, n)
scores = scores.masked_fill(~mask, float('-inf'))    # 屏蔽未来位置
attn = F.softmax(scores, dim=-1)
```

### 🔥 高频追问 Top 3

**Q：训练时加 Causal Mask，那训练效率怎么样？**

A：表面上看，n×n 矩阵里有一半（上三角）是没用的，是浪费。但实际上**仍然是高效的**——因为 GPU 矩阵乘法不擅长跳过特定位置，**整体算完再 mask 掉，比稀疏计算更快**。这就是为什么 Transformer 训练能并行——所有 token 的预测可以一次性算出来，每个位置的 loss 都是有效梯度信号。

**Q：推理时还需要 Causal Mask 吗？**

A：**严格来说不需要**。因为推理是自回归的，每一步只算一个新 token 对所有历史的注意力——本来就只有"过去"，没有"未来"，物理上不存在泄露。所以推理时的 Q 只有一行（当前 token），KV 是所有历史，KV Cache 直接拼上去即可，不需要 mask。

**Q：Cross-Attention 也需要 mask 吗？**

A：**通常不需要**。Cross-Attention 是 Decoder 看 Encoder 的输出，而 Encoder 是双向的，输出包含完整序列信息。Decoder 在生成时看完整 Encoder 是合理的。除非有特殊场景（比如流式翻译，Encoder 也是边生成边喂），才需要给 Cross-Attention 也加 mask。

### ⚠️ 常见陷阱

1. **混淆 padding mask 和 causal mask**：padding mask 是屏蔽 padding token（不让它们参与/被关注），causal mask 是屏蔽未来。**两者要一起用**，通常用按位与组合。
2. **mask 加错位置**：必须在 softmax **之前**，加在 logits 上，不能加在概率上。
3. **训练用 mask，推理不用 mask** 这个区分要清楚。

### 🏢 大厂偏好

- 这是个**实现题**，更多在手撕代码环节考察。常见错误是把 mask 加在 softmax 后面，被面试官现场捉包。

### 📚 延伸阅读
- 见第 9 章手撕代码

---

## Q9：Encoder-Only / Decoder-Only / Encoder-Decoder 分别适合什么任务？ ⭐⭐⭐

### 🎯 一句话标答

> Encoder-Only 适合**理解类任务（分类、检索）**；Decoder-Only 适合**生成与对话**；Encoder-Decoder 适合**输入输出强对齐的任务（翻译、摘要）**。

### 🗣️ 30 秒口语版

"我从两个维度来回答——结构和任务的匹配度。

**Encoder-Only**——BERT 这类，注意力是**双向**的，能同时看左右上下文，所以理解能力强。适合分类、命名实体识别、稠密检索这种'看懂就行，不用生成'的任务。

**Decoder-Only**——GPT/LLaMA 这类，注意力是**单向的因果**，天然是生成式的。适合对话、续写、代码生成、agent 这种需要逐 token 输出的场景。现在 LLM 几乎全是这一派。

**Encoder-Decoder**——T5/BART/Whisper 这类，**两边解耦**，Encoder 充分理解输入，Decoder 自由生成输出。适合输入输出明显是两个不同空间的任务，比如翻译（中文到英文）、语音识别（音频到文字）、摘要（长文到短文）。

简单记忆口诀：**只读 → Encoder，只写 → Decoder，读完再写 → Encoder-Decoder**。"

### 📐 详细对比

| 维度 | Encoder-Only | Decoder-Only | Encoder-Decoder |
|---|---|---|---|
| 注意力方向 | 双向 | 单向因果 | Enc 双向 + Dec 单向 |
| 训练目标 | MLM (15% mask) | CLM (全量 token) | Span Corruption / Seq2Seq |
| 数据利用率 | 低（15%） | 高（100%） | 中 |
| In-Context Learning | 弱 | 强 | 中 |
| 代表模型 | BERT, RoBERTa, BGE | GPT, LLaMA, DeepSeek | T5, BART, Whisper |
| 适合任务 | 分类、NER、Embedding | 生成、对话、Agent | 翻译、摘要、ASR |
| 当前地位 | 仍是 Embedding 主流 | 通用 LLM 主流 | 多模态/对齐场景 |

### 🔥 高频追问 Top 3

**Q：为什么 Embedding 模型大多还是 Encoder？**

A：因为 Embedding 需要**对整句的双向理解**。一个句子的语义不是"从左到右逐渐累积"出来的，而是整句协同的产物。Encoder 的双向注意力天然适合这个——CLS token 或 mean pooling 能融合全局信息。Decoder-Only 也能做 Embedding，但需要技巧（比如 SGPT、LLM2Vec），且通常不如同等规模的 Encoder。

**Q：未来 Encoder-Decoder 还会复兴吗？**

A：在**多模态**领域已经在复兴了。比如 Flamingo、BLIP、Whisper——视觉编码、音频编码这种"异构输入"，用 Encoder 处理更优雅，输出再交给 Decoder 生成文本。所以"纯文本场景 Decoder-Only 赢"和"多模态场景 Encoder-Decoder 仍有空间"并不矛盾。

**Q：如果让你做一个翻译系统，你选哪种？**

A：要看场景——
- 通用聊天里的翻译能力：直接用 Decoder-Only 大模型（GPT-4），效果好且省事
- 专用翻译产品（如 DeepL）：Encoder-Decoder（基于 mBART/NLLB）仍是更优解，因为输入输出对齐更好，质量更可控
- 多模态翻译（图片翻译、语音翻译）：必然是 Encoder-Decoder

### ⚠️ 常见陷阱

1. **不要绝对化**——Decoder-Only 也能做分类、Encoder 也能做生成，只是各有优劣
2. **不要忘记 Encoder 在多模态中的复兴**——这是个加分点

### 🏢 大厂偏好

- **业务岗**（推荐、搜索）：会追问"我们的场景该选哪种架构"，要结合具体任务回答
- **算法岗**：会从训练范式角度追问数据利用率

### 📚 延伸阅读
- [What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?](https://arxiv.org/abs/2204.05832)

---

## 📝 本章小结

| 关键点 | 你必须能脱口而出 |
|---|---|
| Transformer 架构 | Encoder-Decoder × N，每层 = Attn + FFN + Res + Norm |
| Decoder-Only 胜出原因 | 任务统一 / Scaling 友好 / 推理高效 / 数据利用率 |
| Self-Attention 公式 | softmax(QKᵀ/√d_k)·V，按行 softmax |
| 除 √d_k 原因 | QKᵀ 方差为 d_k，不缩放 softmax 会饱和 |
| Self-Attention 复杂度 | 时间 O(n²d)，空间 O(n²) |
| Multi-Head 作用 | 不同子空间学不同模式，参数量不变 |
| MHA→MLA 演化 | 减头数 → 减组数 → 压维度 |
| Causal Mask | 上三角 -∞，加在 softmax 前 |
| 三类架构选型 | 只读 Enc / 只写 Dec / 读完写 Enc-Dec |

## ✅ 自测题

回答这 5 个问题，如果能流畅讲清楚，本章过关：

1. 当面试官问"什么是 Self-Attention"，你能在 30 秒内说清直觉 + 公式 + 一个细节吗？
2. 为什么 √d_k 而不是别的数？现场推导一遍。
3. 画图说清楚 MHA、MQA、GQA、MLA 的区别，并说出各自的 KV Cache 大小。
4. 用 50 行 Python 实现一个带 Causal Mask 的 Self-Attention（不查文档）。
5. 如果面试官问"如果让你改进 Transformer"，你能给出 3 个有理有据的方向吗？

---

> **下一章预告**：第 2 章｜位置编码专题——从 Sinusoidal 讲到 RoPE，再到长文本外推的 NTK / YaRN。位置编码是字节面试的另一个高频区，DeepSeek-V3 的 MLA 涉及 RoPE 的特殊处理也会在那里详谈。
