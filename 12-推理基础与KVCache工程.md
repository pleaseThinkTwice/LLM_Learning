# 第 12 章｜推理基础与 KV Cache 工程

> **本章定位**：第三部分（推理与部署）的地基。核心只有一句话：**Prefill 是算力问题，Decode 是带宽问题，KV Cache 把 Decode 从"重算"变成了"重读"**。由此推出三件事：KV Cache 有多大（Q2）、Decode 的速度上限在哪（Q3）、以及围绕 KV Cache 的三类系统优化：显存怎么放（PagedAttention，Q4）、请求怎么调度（Continuous Batching、Chunked Prefill，Q5、Q6）、前缀怎么复用（Prefix Caching，Q7）。

> **配套章节**：MHA → MQA → GQA → MLA → 第 1 章 Q7 / RoPE 与 MLA 的解耦 RoPE → 第 2 章 Q8 / 量化 → 第 13 章 / 投机解码 → 第 14 章 / FlashAttention、引擎选型、PD 分离 → 第 15 章 / 长上下文训练 → 第 24 章

---

## 📏 标注约定

| 标记 | 含义 |
|---|---|
| 📄 **[论文]** | 论文、技术报告或官方博客的结论（附出处） |
| 🧮 **[推导]** | 可推导或可计算。本章的时间、吞吐数字大多是 Roofline 推算的**理论上界**，实际值会因 kernel 效率和调度开销更低 |
| 💭 **[直觉]** | 经验、常见解释或有争议 |
| ❓ **[开放]** | 尚无共识 |

---

## 📌 本章导航

| 序号 | 题目 | 难度 | 类型 |
|---|---|---|---|
| Q1 | **Prefill vs Decode：两阶段的本质区别** | ⭐⭐⭐⭐⭐ | 必问 |
| Q2 | **KV Cache 原理与显存估算** | ⭐⭐⭐⭐⭐ | 必问 + 计算 |
| Q3 | **Decode 为什么是 Memory-Bound（Roofline）** | ⭐⭐⭐⭐⭐ | 必问 + 推导 |
| Q4 | **PagedAttention 与显存碎片** | ⭐⭐⭐⭐⭐ | 必问 |
| Q5 | **Continuous Batching** | ⭐⭐⭐⭐⭐ | 必问 |
| Q6 | Chunked Prefill 与 Decode 卡顿 | ⭐⭐⭐⭐ | 进阶 |
| Q7 | Prefix Caching / RadixAttention | ⭐⭐⭐⭐ | 应用 |

---

## Q1：Prefill vs Decode——两阶段的本质区别 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> LLM 推理分两阶段：**Prefill** 把整个 prompt 一次性并行前向，是矩阵乘矩阵（GEMM），**compute-bound**，决定首 token 延迟 **TTFT**，同时产出 prompt 所有位置的 KV Cache；**Decode** 逐 token 自回归生成，每步只处理 1 个新 token，是向量乘矩阵（GEMV），**memory-bound**，决定每个输出 token 的时间 **TPOT**。两者瓶颈不同，所以任何推理优化都要先问"优化的是哪一段"。

### 🗣️ 30 秒口语版

"Prefill 时 prompt 的所有 token 都已知，causal mask 保证每个位置只看左边，一次前向就能并行算完，GPU 算力能吃满。Decode 时第 $n$ 个 token 依赖第 $n-1$ 个，只能一步一个；每步计算量很小，但权重还是要完整读一遍，所以时间主要花在从显存搬权重上。一句话：Prefill 拼算力，Decode 拼带宽。"

### 📐 两阶段的计算形态

设参数量为 $P$，prompt 长度为 $n$，已生成长度为 $t$：

| | Prefill | Decode（第 $t$ 步） |
|---|---|---|
| 输入形状 | $[n, d]$ | $[1, d]$ |
| K/V | 当场算 $n$ 行 | **从 Cache 读** $t$ 行，当场算 1 行 |
| Attention 矩阵 | $[n, n]$ | $[1, t]$ |
| 线性层 | GEMM | GEMV |
| 每步 FLOPs | $\approx 2Pn$ | $\approx 2P$ |
| 每步读权重 | $P \times$ 字节数 | $P \times$ 字节数（**一样多**） |
| 瓶颈 / 指标 | 算力 / TTFT | 带宽 / TPOT |

🧮 Decode 每步的计算量只有 Prefill 的 $1/n$，读权重的量却一样。这是全部矛盾的来源。

### 📐 时间感（🧮 估算）

LLaMA-2 7B、A100-80G、FP16，prompt 1000 token、输出 500 token：
- Prefill： $2 \times 7 \times 10^9 \times 1000 = 1.4 \times 10^{13}$ FLOPs；按 312 TFLOPS 峰值、50% 利用率，约 90 ms。
- Decode：每步读 14 GB 权重，A100 带宽约 2 TB/s，每步约 7 ms；500 步约 3.5 s。

💭 在这个设定下总耗时绝大部分在 Decode，所以"推理优化"很多时候就是"Decode 优化"。prompt 很长、输出很短（RAG、分类）时比例会反过来。

### 🔥 高频追问

**Q：Prefill 能靠加 batch 提高吞吐吗？**

A：收益很小。Prefill 已经接近 compute-bound，再堆 batch 只会让每个请求排队更久，TTFT 变差。对 Prefill 有意义的是：少算（Prefix Caching，Q7）、算得快（FlashAttention，第 15 章 Q2）、切开算以免拖累 Decode（Chunked Prefill，Q6）。

**Q：两阶段性质不同，为什么放在同一张卡上？**

A：这就是 **PD 分离**的动机：Prefill 集群追求 TTFT，Decode 集群追求吞吐，中间传 KV Cache。📄 代表工作有 DistServe（OSDI 2024）、Splitwise（ISCA 2024）、Mooncake 等。代价是 KV Cache 的跨机传输，所以 MLA 这类小 KV 的架构在 PD 分离下更有优势（第 15 章 Q5）。

**Q：Decode 的 attention 为什么要单独写 kernel？**

A：形状不同。Prefill 的 attention 是 $[n,n]$ 矩阵，FlashAttention 通过分块让它不落 HBM。Decode 的 Q 只有一行，没有大矩阵，瓶颈是读 KV Cache；batch 小时 GPU 的 SM 大量闲置。📄 Flash-Decoding（Dao et al. 2023，博客）把 KV 的序列维切成多段并行计算再归并，提高长上下文 decode 的并行度。

### ⚠️ 常见陷阱

- 说"Decode 慢是因为要算 attention"。batch 小、上下文不长时，主要时间在读**权重**；长上下文、大 batch 时读 **KV Cache** 才成为主要矛盾。
- 把"推理延迟"混着说，不拆成 TTFT 和 TPOT。
- 忘记 Prefill 的另一个产出：prompt 所有位置的 KV Cache。

### 📚 出处

- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023. arXiv:2309.06180（§2 是两阶段的好综述）
- Zhong et al. *DistServe*. OSDI 2024. arXiv:2401.09670
- Patel et al. *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*. ISCA 2024. arXiv:2311.18677
- Dao et al. *Flash-Decoding for long-context inference*. 2023（PyTorch / Stanford CRFM 博客）

---

## Q2：KV Cache 原理与显存估算 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> KV Cache 把历史 token 的 K、V 存下来，让每步 Decode 只需计算新 token 的那一行，**用显存换计算**。每 token 的大小是 $2 \times L \times n_{kv} \times d_h \times$ 字节数（ $n_{kv}$ 是 **KV 头数**，GQA 下不等于 Q 头数）。🧮 LLaMA-2 7B（MHA）是 512 KB/token，LLaMA-2 70B（GQA-8）只有 320 KB/token，**比 7B 还小**：KV 大小取决于层数 × KV 头数 × 头维度，与参数量没有直接关系。

### 📐 为什么 K、V 能缓存，Q 不用

🧮 第 $i$ 个位置的 $K_i = x_i W^K$、 $V_i = x_i W^V$ 只依赖该位置的输入（以及它左边的上下文，经由前面的层），而 causal mask 保证后来的 token 不会改变它们，所以算出一次即可复用。第 $t$ 步只需要新的 $Q_t$：

$$\text{Attn}(Q_t, K_{1:t}, V_{1:t}) = \text{softmax}\left(\frac{Q_t K_{1:t}^\top}{\sqrt{d_h}}\right) V_{1:t}$$

$Q$ 只有下标 $t$， $K, V$ 是 $1{:}t$：谁需要历史，谁就缓存。

🧮 复杂度：不用 Cache，生成第 $t$ 个 token 要重算前面所有位置，总代价随长度三次方增长；用 Cache 后每步只算一行、读 $t$ 行，总代价降一阶。

### 📐 显存公式与常见模型

$$\text{KVCache}_ {\text{bytes}} = 2 \times B \times S \times L \times n_{kv} \times d_h \times \text{sizeof(dtype)}$$

| 模型（FP16） | 层数 | KV 头数 | $d_h$ | 每 token | 4K 上下文 |
|---|---|---|---|---|---|
| LLaMA-2 7B（MHA） | 32 | 32 | 128 | 512 KB | 2.0 GB |
| LLaMA-2 13B（MHA） | 40 | 40 | 128 | 800 KB | 3.1 GB |
| LLaMA-2 70B（GQA-8） | 80 | 8 | 128 | 320 KB | 1.25 GB |
| LLaMA-3 8B（GQA-8） | 32 | 8 | 128 | 128 KB | 0.5 GB |
| DeepSeek-V3（MLA） | 61 | — | — | 约 69 KB | 约 0.27 GB |

🧮 以上都可由公式直接算出。DeepSeek-V3 每层缓存 $d_c + d_h^R = 512 + 64 = 576$ 个元素（第 2 章 Q8）， $61 \times 576 \times 2 \approx 69$ KB。

🧮 **并发预算**：A100-80G 放 7B FP16 权重 14 GB，再留约 2 GB 给激活和临时缓冲，剩约 64 GB 给 KV Cache。每条 4K 请求 2 GB，理论上最多约 32 条；换成 LLaMA-3 8B（128 KB/token）则约 128 条。KV 大小决定并发，并发决定吞吐，吞吐决定单位成本。

### 🔥 高频追问

**Q：KV Cache 会不会比重算还慢？**

A：🧮 对标准 decoder 几乎不会。7B FP16 在 A100 上读一个 token 的 K、V（512 KB）约 0.26 µs；重新得到这个 token 在所有层的 K、V 要过一遍整个网络，约 $2 \times 7\text{B} = 14$ GFLOP，峰值算力下约 45 µs，贵两个数量级以上。

> ❌ 旧版写"序列很长时读 KV Cache 会比重算更慢，而且不罕见"，不成立。

真正值得分析的是**读 KV Cache 什么时候超过读权重**：batch=1 的 7B 模型，读权重 14 GB，读 KV $S \times 512$ KB。🧮 $S \approx 14\text{ GB} / 512\text{ KB} \approx 2.8$ 万时两者相等，再长下去 KV 成为 Decode 的主要瓶颈。这解释了长上下文下 Decode 变慢、GQA/MLA 的价值被放大，以及 KV 量化和压缩的必要性。

💭 "重算更划算"的情况确实存在，但在别处：抢占后 KV 已被换出到 CPU 时，重新 prefill 可能比从 PCIe 搬回来快（Q4）。

**Q：能不能只保留一部分 KV？**

A：可以，前提是注意力稀疏。主要几派：
- 📄 **StreamingLLM**（Xiao et al., ICLR 2024）：发现 **attention sink**（开头几个 token 获得大量注意力），只保留开头几个 token 和最近的滑动窗口，可无限长流式生成，但会丢中间信息。
- 📄 **H2O**（Zhang et al., NeurIPS 2023）：按累积注意力识别"heavy hitter"，驱逐其余 token。
- 📄 **SnapKV**（Li et al., NeurIPS 2024）、📄 **PyramidKV**（Cai et al. 2024）：在 Prefill 结束时一次性压缩，不同层保留不同比例。
- 📄 **量化而非丢弃**：KIVI（Liu et al., ICML 2024，Key 按通道、Value 按 token 量化到 2 bit）、KVQuant（Hooper et al., NeurIPS 2024），见第 13 章 Q7。

💭 这些都是**有损**的，生产中要说清在什么场景愿意接受（长文摘要可以，精确检索不行）。第 24 章 Q6 的检索头研究提供了另一个思路：只对少数负责检索的头保留完整 KV。

**Q：为什么主流是 GQA 而不是更狠的 MQA？**

A：📄 GQA 论文（Ainslie et al., EMNLP 2023）报告 GQA 质量接近 MHA、速度接近 MQA。🧮 从 32 头的 MHA 到 GQA-8，KV 降 4 倍；从 GQA-8 到 MQA 再降 8 倍，但质量下降更明显。工程上，MQA 只有 1 个 KV 头，TP=8 时无法切分，只能每卡复制一份；💭 GQA-8 正好在 TP=8 时每卡一个 KV 头，常被认为是这一配置流行的原因之一。MLA 走另一条路：不减头数而是压缩维度（第 1 章 Q7）。

### ⚠️ 常见陷阱

- 公式里用 $n_q$ 而不是 $n_{kv}$；忘了乘 2；不说 dtype（FP8 KV 时减半）。
- 说"KV Cache 存的是 attention 结果"。存的是投影后的 K、V。

### 📚 出处

- Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need*（MQA）. 2019. arXiv:1911.02150
- Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. EMNLP 2023. arXiv:2305.13245
- Xiao et al. *Efficient Streaming Language Models with Attention Sinks*. ICLR 2024. arXiv:2309.17453
- Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*. NeurIPS 2023. arXiv:2306.14048
- Li et al. *SnapKV: LLM Knows What You are Looking for Before Generation*. NeurIPS 2024. arXiv:2404.14469
- Cai et al. *PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling*. 2024. arXiv:2406.02069
- Liu et al. *KIVI*. ICML 2024. arXiv:2402.02750；Hooper et al. *KVQuant*. NeurIPS 2024. arXiv:2401.18079

---

## Q3：Decode 为什么是 Memory-Bound（Roofline） ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> **算术强度** = FLOPs ÷ 访存字节数。Decode 时每个 FP16 参数（2 字节）只参与 2 次浮点运算，batch=1 时算术强度约为 **1 FLOP/byte**；🧮 A100 的平衡点是 $312 \text{ TFLOPS} / 2 \text{ TB/s} \approx 150$ FLOP/byte。低两个数量级，所以算力几乎全在空转。直接推论：**batch=1 时 TPOT 的下界 = 模型字节数 ÷ HBM 带宽**，7B FP16 在 A100 上约 7 ms，即约 143 tok/s。

### 📐 机器平衡点

$$\text{Machine Balance} = \frac{\text{Peak FLOPS}}{\text{Memory Bandwidth}}$$

| GPU | FP16/BF16 稠密峰值 | HBM 带宽 | 平衡点（🧮） |
|---|---|---|---|
| A100-80G | 312 TFLOPS | 2039 GB/s | 约 153 |
| H100-SXM | 989 TFLOPS | 3350 GB/s | 约 295 |
| L40S | 362 TFLOPS | 864 GB/s | 约 419 |

🧮 H100 相对 A100 算力约 3.2 倍、带宽约 1.6 倍，平衡点上升，"内存墙"更明显。

### 📐 算术强度约等于 batch

📄 batch 为 $B$ 时，FLOPs 约 $2PB$，而权重只读一次（全 batch 共享），访存约 $2P$ 字节：

$$\text{Intensity} \approx \frac{2PB}{2P} = B \ \text{FLOP/byte}$$

这一分析在 📄 Pope et al.（*Efficiently Scaling Transformer Inference*, MLSys 2023）中有系统推导。

| batch | 状态（A100） | 吞吐上界（🧮 7B，忽略 KV 读取） |
|---|---|---|
| 1 | 严重 memory-bound | 约 140 tok/s |
| 32 | memory-bound | 约 4500 tok/s |
| 约 150 | 接近平衡点 | 接近算力上限 |

🧮 但 batch 到 150 之前 KV Cache 通常先爆显存：7B、4K 上下文、batch 150 需要约 300 GB KV。**在很多场景里，限制吞吐的是 KV Cache 显存而不是算力**，这正是 Q4 的动机。

### 📐 三条优化路线

$$\text{TPOT}_ {\min} \approx \frac{\text{每步读取的字节}}{\text{HBM 带宽}}$$

| 路线 | 改变什么 | 章节 |
|---|---|---|
| 减少字节：量化 | 分子。🧮 7B INT4 约 3.5 GB → 约 1.75 ms/token | 第 13 章 |
| 摊薄字节：加大 batch | 每字节服务更多请求 | Q4、Q5 |
| 每步多出 token：投机解码 | 每次读权重产出多个 token | 第 14 章 |

🧮 70B FP16 在 8 卡 TP 上，每卡读 $140/8$ GB，带宽并行叠加，下界约 8.75 ms：**TP 在推理端的价值不只是装得下，还在于叠加带宽**（第 15 章 Q4）。

### 🔥 高频追问

**Q：attention 部分的算术强度也随 batch 上升吗？**

A：🧮 不会。线性层的权重被全 batch 共享，而每条序列有**自己的** KV Cache，batch 加大时 KV 的访存同步增加，attention 部分的算术强度不随 batch 改善（约为常数，与 GQA 的分组数有关）。所以长上下文、大 batch 时 attention 逐渐成为 Decode 的主导瓶颈，batching 救不了它，只能靠 GQA/MLA、KV 量化和压缩。

**Q：MoE 的算术强度怎么算？**

A：🧮 batch=1 时只读被激活的专家，访存少，这是 MoE 推理友好的原因；batch 大时不同 token 路由到不同专家，几乎所有专家都被读到，但每个专家只分到约 $kB/N$ 个 token，算术强度被稀释。📄 DeepSeek-V3 decode 阶段以 320 卡为最小部署单元、EP320，目的之一就是让每个专家攒到足够大的 batch（第 15 章 Q6、第 21 章）。

**Q：用带宽更高但算力弱的卡是不是更划算？**

A：💭 对纯 Decode（PD 分离后的 D 集群）是的，所以推理卡在堆带宽和显存；混合部署时 Prefill 仍需算力。用 SRAM 替代 HBM 的架构（如 Groq、Cerebras）也是在拆内存墙，代价是单芯片容量小，需要大量芯片拼一个模型。

### ⚠️ 常见陷阱

- 只说"Decode 慢因为不能并行"，没讲到 memory-bound。
- 把 batch 无限吹，不提 KV Cache 显存的约束。
- 不区分线性层和 attention 的算术强度。

### 📚 出处

- Williams et al. *Roofline: An Insightful Visual Performance Model for Multicore Architectures*. CACM 2009
- Pope et al. *Efficiently Scaling Transformer Inference*. MLSys 2023. arXiv:2211.05102
- Chen. *Transformer Inference Arithmetic*. 2022（kipp.ly 博客）

---

## Q4：PagedAttention 与显存碎片 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 📄 vLLM 论文（Kwon et al., SOSP 2023）发现，已有系统按 max_seq_len 为每条请求预分配连续显存，**只有 20.4%–38.2% 的 KV Cache 显存真正存放了 token 状态**，其余浪费在内部碎片、外部碎片和无法共享上。PagedAttention 借鉴操作系统的**虚拟内存分页**：KV Cache 切成固定大小的 block，物理上不连续，用 block table 映射，把浪费降到 **4% 以下**；相同延迟下吞吐相对 FasterTransformer 和 Orca 提升 **2–4 倍**。

### 📐 三种浪费

根因是**事先不知道会生成多少 token**，而早期实现要求每条序列的 KV 连续存放，只能按最坏情况预分配。
1. **内部碎片**：预留 2048 个位置，实际只生成 100 个。
2. **外部碎片**：不同大小的预留块释放后留下空洞，新请求放不进去。
3. **无法共享**：同一 system prompt、并行采样（ $n > 1$）、beam search 的相同前缀只能各存一份。

### 📐 分页的做法

- KV Cache 切成固定大小的 block（vLLM 默认 16 个 token 一块）。
- 每条序列有一张 **block table**（逻辑块 → 物理块），物理块可以完全不连续。
- **按需分配**：写满一块才申请下一块，🧮 内部碎片最多一块。
- **共享**：多张 block table 指向同一物理块，用**引用计数 + copy-on-write**，写时才复制被修改的那一块。并行采样和 beam search 的分叉因此是零拷贝的，这也是 Prefix Caching（Q7）的基础。

代价：attention kernel 不能再假设 KV 连续，要先查 block table 再 gather，分块计算并用 online softmax 增量合并（与 FlashAttention 同一套数学）。核心工作量在 kernel 里，所以叫 Paged**Attention**。

📄 block 大小的权衡：太小则 GPU 并行度低、查表开销大；太大则内部碎片回来。vLLM 论文的实验中 16 在不同负载下表现较好。

### 🔥 高频追问

**Q：显存用完了怎么办？**

A：**抢占**。两种恢复方式：
- **Swapping**：把被抢占序列的 KV 拷到 CPU 内存，之后再拷回。🧮 PCIe 4.0 x16 约 32 GB/s，比 HBM 慢约 60 倍。
- **Recomputation**：直接丢弃，恢复时重新 prefill。重算是 compute-bound 的并行计算。

📄 vLLM 论文的测量：block 较小时，swap 需要大量小块 PCIe 传输，开销大，重算更划算；block 较大时 swap 才更有优势。vLLM 抢占时通常选择最晚到达的序列，并且整条序列一起处理。

**Q：PagedAttention 有开销，为什么整体更快？**

A：📄 vLLM 论文测得 attention kernel 本身比连续布局的实现慢约 20%–26%，但显存利用率提升后能同时运行的序列多得多，而 Decode 是 memory-bound，🧮 batch 翻倍近似吞吐翻倍（Q3）。这是"牺牲单点效率、换取系统吞吐"。

**Q：PagedAttention 和 MLA 冲突吗？**

A：不冲突，只是 block 里存的东西变了：MLA 下存的是压缩后的潜向量 $c_t$ 和解耦 RoPE 部分（第 2 章 Q8）。分页、共享、抢占逻辑都复用，只是 paged kernel 要专门适配。🧮 MLA 减小每 token 字节数，PagedAttention 让显存能被用满，两者效果相乘。

### ⚠️ 常见陷阱

- 只说"分页"，不说为什么必须分页（不知道生成长度 → 只能按最坏情况预分配）。
- 忘了共享与 copy-on-write。
- 说"PagedAttention 让 attention 算得更快"。kernel 更慢，赢在并发。
- 把 PagedAttention 和 Continuous Batching 混为一谈：前者管显存，后者管调度。

### 📚 出处

- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023. arXiv:2309.06180（§3、§6、§7）

---

## Q5：Continuous Batching ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> **Static batching** 要等一批里生成最长的请求结束才能换下一批，早结束的槽位空转；📄 **Continuous batching**（Orca, OSDI 2022 的 iteration-level scheduling）把调度粒度降到**每一次前向迭代**：任何序列一结束就移出、立刻补入新请求。根因是 **LLM 的输出长度不可预测**、方差很大。📄 Anyscale 的基准报告在其设置下相对 static batching 最高约 23 倍吞吐；💭 收益大小主要取决于输出长度的方差。

### 📐 静态 vs 连续

🧮 示意：batch=4，四条请求分别生成 3、12、5、4 个 token。静态批处理要跑 12 步，有效计算 $(3+12+5+4)/(4 \times 12) = 50$%，而且期间新请求只能排队。连续批处理在第 4 步就把结束的槽位补上新请求，有效计算接近 100%。

### 📐 Selective batching 与变长打包

📄 Orca 的观察：Transformer 里不是所有算子都要求"矩形 batch"。线性层、LayerNorm 可以把不同长度的序列拉平成 $[\sum_i S_i, d]$ 一起算；attention 每条序列的 KV 长度不同，需要逐序列计算。现代实现（FlashAttention 的 varlen 接口、vLLM 的 paged kernel）把这一点做进了 kernel：输入是一维拉平的 token 序列加上累积长度索引，没有 padding。

### 📐 调度器每一步做什么

每一次前向前：① 移出已结束的序列，释放其 KV block，立即返回结果；② 显存允许时从等待队列拉入新请求（先做 prefill）；③ 显存不够时抢占（Q4）；④ 前向一步，prefill 和 decode 的序列可以混在同一个 batch 里。

### 🔥 高频追问

**Q：Continuous batching 和 dynamic batching 是一回事吗？**

A：不是。Dynamic batching（Triton 等推理服务的经典功能）在一个时间窗口内**攒批**，一批开始后成员不变，解决的是"请求到达时间不齐"。Continuous batching 的成员在运行中动态进出，解决的是"请求**结束**时间不齐"。💭 CV 模型每个请求计算量固定，LLM 的计算量取决于生成多少 token，事先不知道，这是两者的根本区别。

**Q：它和 PagedAttention 是什么关系？**

A：正交但互为前提。Continuous batching 频繁地进出序列，如果显存是连续预分配的，会立即制造外部碎片；反过来 PagedAttention 省下的显存，如果调度还是静态的，也填不满。📄 Orca 有 iteration-level scheduling 但按最大长度预留 KV；vLLM 补上了显存层。

**Q：有什么代价？**

A：
1. **尾延迟与公平性**：被抢占的请求延迟暴涨，调度策略显著影响 P99。
2. **TTFT 与 TPOT 冲突**：新请求的 prefill 插进来会卡住正在 decode 的序列（Q6）。
3. **CPU 调度开销**：每一步都要调度，decode 一步只有几毫秒时，CPU 侧的调度和准备可能占相当比例。💭 这是推理引擎把调度与执行解耦、做 CPU-GPU 重叠的动机之一（第 15 章 Q3）。

### ⚠️ 常见陷阱

- 与 dynamic batching 混淆；不提"输出长度不可预测"这个根因。
- 只说好处不说代价（抢占伤 P99、prefill 卡 decode）。

### 📚 出处

- Yu et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models*. OSDI 2022
- Anyscale. *How continuous batching enables 23x throughput in LLM inference while reducing p50 latency*. 2023（博客）
- Kwon et al. vLLM，arXiv:2309.06180

---

## Q6：Chunked Prefill 与 Decode 卡顿 ⭐⭐⭐⭐

### 🎯 一句话标答

> Continuous batching 下，一个长 prompt 的 prefill 会让某一步迭代从几毫秒变成几百毫秒，所有正在 decode 的用户一起卡住（**generation stall**）。📄 **Chunked prefill**（Sarathi，以及系统化的 Sarathi-Serve, OSDI 2024）把长 prefill 切成固定 token 预算的小块，与 decode **捎带（piggyback）**在同一个 batch 里执行：decode 是 memory-bound、prefill 是 compute-bound，**两者的瓶颈互补**，混合 batch 共享同一次权重读取，GPU 利用率反而更高。代价是长 prompt 自身的 TTFT 略变差。

### 📐 问题的量化（🧮 示意）

32 条序列在 decode，每步约 7 ms。插入一个 8K token 的 prefill：

- 不切块：这一步变成约 500 ms，32 个用户都卡住半秒；ITL 的 P99 由这类步骤决定。
- 切成 512 token 一块：每步约 30–40 ms，分十几步完成。ITL 平滑；长 prompt 的 TTFT 只比一次算完略长，因为每一步的算力被更充分地利用了。

数值依模型、硬件和实现而定，这里只演示量级关系。

### 📐 为什么混合 batch 更高效

🧮 回到 Q3：纯 decode（32 条）算术强度约 32，memory-bound；纯 prefill（8K token）约 8000，compute-bound；32 条 decode 加 512 token 的 prefill 块，约 544，接近平衡点。权重只读一次，decode 和 prefill 共享；prefill 的计算填满了 decode 闲置的算力。

### 📐 关键参数

一步的 token 预算（vLLM 中为 `max_num_batched_tokens`）是 decode token 与 prefill 块之和。调小：每步快、ITL 平滑、长 prompt 的 TTFT 变差，适合聊天和流式 Agent；调大：TTFT 好、但会卡 decode，适合离线批处理。vLLM 的调度是 **decode 优先**：先排入所有 running 序列的 decode token，剩余预算才给 prefill。💭 起点可以从几百到两千左右试，按实测的 ITL P99 与 TTFT 曲线确定。

### 🔥 高频追问

**Q：切块计算和一次计算结果一样吗？**

A：🧮 数学上等价：causal mask 下第 $i$ 个位置的 KV 只依赖 $\le i$ 的位置，先算前 512 个再算后 512 个（后者可以读到已算好的前者的 KV），依赖关系完全相同。浮点归约顺序不同可能带来极小的数值差异，不影响质量，但会让结果不能逐位复现，做回归测试时要注意。这是调度层面的优化，属于**无损**，和量化、KV 驱逐（有损）性质不同。

**Q：Chunked prefill 和 PD 分离是竞争关系吗？**

A：是同一问题（prefill 与 decode 相互干扰）的两种解法：前者**融合**，让两者互补；后者**隔离**，各自优化到极致，可以用异构硬件，代价是 KV 跨机传输和部署复杂度（第 15 章 Q5）。💭 中小规模用 chunked prefill 通常足够；大规模可以两者叠加。

### ⚠️ 常见陷阱

- 说"chunked prefill 是为了让 prefill 更快"。它主要是不让 prefill 拖累 decode。
- 只说"削峰"，不说"瓶颈互补"。

### 📚 出处

- Agrawal et al. *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*. 2023. arXiv:2308.16369
- Agrawal et al. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*. OSDI 2024. arXiv:2403.02310

---

## Q7：Prefix Caching / RadixAttention ⭐⭐⭐⭐

### 🎯 一句话标答

> 不同请求经常共享相同的前缀（system prompt、工具定义、few-shot 示例、多轮对话历史、同一篇检索文档）。Prefix caching 把前缀的 KV Cache **跨请求复用**，命中部分**完全跳过 prefill**，大幅降低 TTFT。🧮 能复用的根据是 causal mask：前缀的 KV 只依赖前缀本身，与后面接什么无关。vLLM 用**按块的前缀哈希 + 引用计数**实现；📄 SGLang 的 **RadixAttention** 用基数树（radix tree）管理缓存，支持树形共享和 LRU 驱逐。**它只改善 TTFT，不改善 TPOT。**

### 📐 命中率从哪来

| 场景 | 共享的前缀 |
|---|---|
| 多轮对话 | 前面所有轮次的历史 |
| Agent（ReAct 循环） | system、工具 schema、之前所有的 thought/action/observation（每轮只在末尾追加） |
| Few-shot 批量任务 | system 与示例 |
| RAG | system；同一文档被多个问题检索到时，文档也能共享 |
| 无状态单轮问答 | 只有 system prompt |

🧮 Agent 第 $k$ 轮的 prompt 是第 $k-1$ 轮加上新的 observation。不开 prefix caching， $K$ 轮的总 prefill 量随 $K$ 二次增长（每轮重算全部历史）；开了之后只算每轮新增的部分。

### 📐 vLLM：前缀哈希

每个满块的哈希必须包含**它前面所有块的哈希**，即 $h_i = \text{hash}(h_{i-1}, \text{块 } i \text{ 的 token})$。否则"内容相同、前缀不同"的块会被错误复用，而它们的 KV 其实不同，结果会出错。匹配从头开始，**一旦某块不命中，之后全部作废**，这是"前缀"的字面含义。未写满的块不缓存，因为内容还会变。

### 📐 SGLang：RadixAttention

📄 SGLang（Zheng et al., NeurIPS 2024）用基数树管理所有请求的 KV：节点保存一段 token 及其 KV，匹配就是从根往下走。优点是天然支持树形共享（多候选采样、Agent 的分支探索），并可以从叶子开始做 LRU 驱逐，保留被更多请求依赖的公共前缀（第 15 章 Q3）。

### 🔥 高频追问

**Q：有什么风险？**

A：
1. **安全与隐私**：跨用户共享 KV 意味着存在**时序侧信道**：攻击者通过 TTFT 是否异常地快，判断某个前缀是否被别人用过。📄 Song et al.（2024）演示了针对推理系统缓存的这类攻击；📄 PrefixWall（Pennas et al. 2026）提出了防御。对策包括按租户隔离缓存（例如 vLLM 的 `cache_salt`，只有相同 salt 的请求才共享）。
2. **显存**：缓存占的显存不能用于运行中的请求，命中率低时得不偿失，需要 LRU 和预算控制。
3. **可复现性**：同一前缀在不同 batch 组合下算出的 KV 可能有极小的浮点差异。

**Q：怎么组织 prompt 以提高命中率？**

A：**稳定的放前面，变化的放后面**：system → 工具定义（固定顺序）→ few-shot（固定）→ 对话历史（只追加）→ 新输入或新 observation。常见的坑：开头放时间戳或用户 ID；按相关性动态排序工具；JSON 序列化不固定 key 顺序（要用 `sort_keys=True`）；用滑动窗口截断对话历史（每轮改动前缀，缓存全废）。💭 需要压缩历史时，在固定的检查点压缩，而不是每轮滑一格（第 18 章 Q5）。

### ⚠️ 常见陷阱

- 块哈希不包含前缀，导致错误复用（正确性问题，不是性能问题）。
- 以为 prefix caching 能改善 TPOT。
- 不提跨租户的侧信道风险。

### 📚 出处

- Zheng et al. *SGLang: Efficient Execution of Structured Language Model Programs*. NeurIPS 2024. arXiv:2312.07104
- vLLM 文档. *Automatic Prefix Caching*（设计文档）
- Gim et al. *Prompt Cache: Modular Attention Reuse for Low-Latency Inference*. MLSys 2024. arXiv:2311.04934（复用非前缀的模块化片段）
- Song et al. *The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems*. 2024. arXiv:2409.20002
- Pennas et al. *PrefixWall: Mitigating Prefix Caching Side Channels in Shared LLM Systems*. 2026. arXiv:2603.10726

---

## 🔬 研究前沿与开放问题

- ❓ **KV 压缩的可靠评估**：H2O、SnapKV、PyramidKV 等在常见基准上近似无损，但在多轮对话（后续问题依赖之前被丢弃的 token）和多跳检索上的损失缺少系统评估。哪些任务对 KV 驱逐最敏感？
- ❓ **缓存与压缩的统一**：上下文压缩会改写前缀、破坏 prefix caching（Q7、第 18 章 Q5）。能否只在 KV 层面做压缩或合并，不改 token 前缀？
- ❓ **调度的最优性**：在 TTFT/TPOT 双 SLO 下，chunk 大小、batch 组成、抢占策略的联合优化目前多是启发式（第 15 章）。
- ❓ **注意力稀疏的机制**：attention sink、检索头（第 24 章 Q6）解释了为什么少量 token 承担了大部分注意力；这能否指导"该保留哪些 KV"的有原则的方法？

🧪 **实验思路（文字描述）**：
- 用公式计算几个开源模型（MHA、GQA、MLA）在 4K、32K、128K 下的每 token KV 和最大并发，再与 vLLM 启动日志中报告的 KV block 数对照。
- 在同一模型上分别以 batch 1、8、32、128 测 decode 吞吐，画出吞吐-batch 曲线，标出 KV 显存耗尽的位置，验证 Q3 的"算术强度约等于 batch"和"显存先于算力成为瓶颈"。
- 构造一个多轮 Agent 轨迹，分别在 prompt 开头加入与不加入时间戳，测 prefix cache 命中率和 TTFT。
- 在长文档问答上比较 StreamingLLM、H2O、SnapKV 在不同保留比例下的准确率，再追加一个依赖被丢弃信息的追问。

---

## 📝 本章小结

| 关键点 | 结论 | 性质 |
|---|---|---|
| 两阶段 | Prefill = compute-bound = TTFT；Decode = memory-bound = TPOT | 🧮 |
| KV Cache | $2 \times L \times n_{kv} \times d_h \times$ 字节数 / token；7B 512 KB，70B（GQA-8）320 KB | 🧮 |
| Roofline | Decode 算术强度约等于 batch；A100 平衡点约 150；batch=1 时 TPOT 下界 = 模型字节 ÷ 带宽 | 🧮 + 📄 |
| 三条路线 | 量化 / 加大 batch / 投机解码 | 🧮 |
| attention 的强度 | 每条序列独享 KV，不随 batch 改善；长上下文时成为主瓶颈 | 🧮 |
| PagedAttention | 有效利用率 20%–38% → 浪费 < 4%；kernel 慢约 20%–26%，吞吐 2–4 倍 | 📄 |
| Continuous batching | 调度粒度降到迭代；根因是输出长度不可预测 | 📄 |
| Chunked prefill | prefill 与 decode 瓶颈互补；无损；略伤长 prompt 的 TTFT | 📄 + 🧮 |
| Prefix caching | 救 TTFT 不救 TPOT；哈希要包含前缀；变化的放最后；有侧信道风险 | 📄 + 🧮 |

## ✅ 自测题

1. A100-80G 部署 LLaMA-3 70B（80 层，GQA-8， $d_h = 128$），FP16，32K 上下文：每条请求的 KV 多大？需要几张卡？最多几并发？
2. 用 Roofline 解释为什么 batch=1 时 7B FP16 在 A100 上难以超过约 140 tok/s。换 H100 呢？量化到 INT4 呢？
3. 为什么说 batching 能提高线性层的算术强度，却提高不了 attention 的？这对长上下文意味着什么？
4. 读 KV Cache 在多长的上下文时超过读权重（7B，batch=1）？KV Cache 会不会比重算还慢？
5. PagedAttention 让 attention kernel 变慢，为什么整体吞吐反而提高？
6. Continuous batching 和 dynamic batching 的区别是什么？为什么后者对 LLM 基本没用？
7. 为什么 chunked prefill 能同时改善 ITL 尾部和 GPU 利用率？为什么说它是无损的？
8. Prefix caching 的块哈希为什么必须包含前缀？为什么滑动窗口截断历史会让缓存失效？
9. 你的 Agent 线上反馈"每次工具调用后要等很久才出下一个字"，给出三个可能原因和排查方法。

---

> **下一章预告**：第 13 章｜量化专题——PTQ 与 QAT、GPTQ（从 OBS 到 Hessian 近似）、AWQ（为什么"激活感知"是对的）、SmoothQuant，以及 W4A16 / W8A8 / FP8 的选型。本章说"减少每步读取的字节"是三条优化路线之一，第 13 章就是这条路线。
