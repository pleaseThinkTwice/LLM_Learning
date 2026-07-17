# 第 12 章｜推理基础与 KV Cache 工程

> **本章定位**：这是**第三部分（推理与部署）的地基**，也是应用岗 / Agent 岗面试的第一道分水岭。训练岗聊 Scaling Law，应用岗聊 **TTFT 和显存**。本章的核心只有一句话：**Prefill 是算力问题，Decode 是带宽问题，而 KV Cache 是把 Decode 从"重算"变成"重读"的那个交易**。能把这条主线讲透，PagedAttention、Continuous Batching、Chunked Prefill 全都是它的推论。

> **配套章节**：KV Cache 的**架构侧**（MHA→MQA→GQA→MLA）→ 第 1 章 / 长上下文外推（PI/NTK/YaRN）→ 第 2 章 / 量化 → 第 13 章 / 投机解码 → 第 14 章 / 推理引擎选型与服务化指标 → 第 15 章

---

## 📌 本章导航

| 序号 | 题目 | 难度 | 频率（字节/阿里/Meta/DeepSeek） | 类型 |
|---|---|---|---|---|
| Q1 | **Prefill vs Decode：两阶段的本质区别** | ⭐⭐⭐⭐⭐ | **9 / 7 / 6 / 6** | 必问 |
| Q2 | **KV Cache 原理与显存估算** | ⭐⭐⭐⭐⭐ | **9 / 8 / 6 / 7** | 必问·可能手撕 |
| Q3 | **Decode 为什么是 Memory-Bound（Roofline）** | ⭐⭐⭐⭐⭐ | **8 / 5 / 6 / 6** | 必问 |
| Q4 | **PagedAttention 与显存碎片** | ⭐⭐⭐⭐⭐ | **9 / 7 / 5 / 5** | 必问 |
| Q5 | **Continuous Batching** | ⭐⭐⭐⭐⭐ | **8 / 7 / 5 / 4** | 必问 |
| Q6 | Chunked Prefill 与 Decode 卡顿 | ⭐⭐⭐⭐ | 6 / 4 / 3 / 4 | 进阶 |
| Q7 | Prefix Caching / RadixAttention | ⭐⭐⭐⭐ | 6 / 5 / 3 / 3 | 应用 |

---

## Q1：Prefill vs Decode——两阶段的本质区别 ⭐⭐⭐⭐⭐（必问）

### 🎯 一句话标答

> LLM 推理分两阶段：**Prefill 把整个 prompt 一次性并行前向，是 compute-bound，决定 TTFT**；**Decode 逐 token 自回归生成，每步只算 1 个 token，是 memory-bound，决定 TPOT**——两者的瓶颈完全不同，所以几乎所有推理优化都要先问"你优化的是哪一段"。

### 🗣️ 30 秒口语版

"LLM 推理天然是两个阶段，性质截然不同。

**Prefill（预填充）**：用户输入一段 prompt，比如 1000 个 token。这 1000 个 token **可以一次性并行送进模型**——因为它们全都是已知的，causal mask 保证每个位置只看自己左边，一次前向就能把所有位置算完。这一步产出两样东西：第一个输出 token，以及**这 1000 个位置的 KV Cache**。

这个阶段是**矩阵乘矩阵**（1000 × d 的激活乘权重），GPU 的算力被喂饱，是 **compute-bound**。它决定了用户的**首 token 延迟 TTFT**——也就是"我按了回车之后要等多久才看到第一个字"。

**Decode（解码）**：从第二个 token 开始，每一步只输入**上一步刚生成的那 1 个 token**，算出下一个。这是自回归的本质——第 n 个 token 依赖第 n-1 个，**没法并行**。

这个阶段是**向量乘矩阵**（1 × d 乘权重）。算力上只有 prefill 的千分之一，但**权重还是要完整读一遍**——所以 GPU 在疯狂搬数据、算力却闲着，是 **memory-bound**。它决定了 **TPOT**（每个输出 token 的时间），也就是"字往外蹦得快不快"。

**一句总结**：Prefill 是短跑冲刺，拼算力；Decode 是长跑，拼显存带宽。**Batch 只能救 Decode，救不了 Prefill**——这是后面所有优化的出发点。"

### 📐 详细拆解

#### 两阶段的计算形态

设 hidden dim = $d$，prompt 长度 = $n$，已生成长度 = $t$。

| | Prefill | Decode（第 t 步） |
|---|---|---|
| 输入形状 | $[n, d]$ | $[1, d]$ |
| Q 的形状 | $[n, d]$ | $[1, d]$ |
| K/V 的形状 | $[n, d]$（当场算） | $[t, d]$（**从 Cache 读** + 当场算 1 行） |
| Attention 矩阵 | $[n, n]$ | $[1, t]$ |
| FFN 计算 | 矩阵 × 矩阵 (GEMM) | **向量 × 矩阵 (GEMV)** |
| 每步 FLOPs | $\approx 2 \cdot P \cdot n$ | $\approx 2 \cdot P$ |
| 每步读权重 | $P \times$ dtype | $P \times$ dtype（**一样多！**） |
| 瓶颈 | 算力 | 带宽 |
| 对应指标 | **TTFT** | **TPOT / ITL** |

> $P$ = 模型参数量。注意最后两行：**Decode 每步的计算量只有 Prefill 的 $1/n$，但读权重的量一模一样**。这就是全部矛盾的来源。

#### 一个具体的时间感

以 LLaMA-2 7B / A100-80G / fp16 为例，prompt 1000 token、输出 500 token：

```
Prefill:  1000 token 一次前向
          FLOPs ≈ 2 × 7e9 × 1000 = 1.4e13 = 14 TFLOP
          A100 fp16 峰值 ≈ 312 TFLOPS，按 50% MFU 算
          → 约 90 ms          ← 这就是 TTFT

Decode:   500 步，每步读一遍 14 GB 权重
          A100 HBM 带宽 ≈ 2000 GB/s
          → 每步 ≈ 14/2000 = 7 ms
          → 500 步 ≈ 3.5 s     ← 这就是用户看到的"打字速度"
```

**结论**：总耗时里 **Decode 占 97%**。所以"推理优化"绝大部分时候等于"Decode 优化"。

### 💡 直觉理解

可以这样类比——

**Prefill** 像是**开卷考试的审题阶段**：题目一次性发到你手里，你可以一眼扫完全部，速度取决于你脑子转多快（算力）。

**Decode** 像是**一个字一个字往外挤答案**：每写一个字，你都要**把整本教科书从头翻一遍**（读全部权重），但实际只用到其中一句话（算 1 个 token）。翻书的时间远大于思考时间——所以速度取决于你翻书多快（带宽），跟你脑子多聪明没关系。

**那怎么办？**——**同时给 32 个人答题，共用一次翻书**。这就是 batching：翻书成本被 32 个人摊薄，算力才被用起来。这也解释了为什么 **batch 对 Decode 有奇效、对 Prefill 几乎无效**（Prefill 本来就已经吃满算力了）。

### 🔥 高频追问 Top 3

**Q：Prefill 阶段能不能也用 batch 提高吞吐？**

A：**收益很小，而且有害**。因为 Prefill 已经是 compute-bound——GPU 算力已经跑满了，再堆 batch 只是让每个请求排队更久，总吞吐不变，但 **TTFT 变差**。

真正对 Prefill 有意义的优化是另一个方向：
- **减少要算的 token**：Prefix Caching（Q7）——命中的部分直接跳过
- **算得更快**：FlashAttention（避免 $[n,n]$ 矩阵的 HBM 读写）、更好的 kernel
- **切开来算**：Chunked Prefill（Q6）——不是为了 Prefill 快，是为了不让它拖累 Decode

**Q：既然 Prefill 和 Decode 性质完全不同，为什么要放在同一张卡上跑？**

A：**这正是当前的前沿方向——PD 分离（Disaggregated Prefill/Decode）**。

放一起的问题：一个是 compute-bound、一个是 memory-bound，混在一起两边都不最优；而且长 prompt 的 Prefill 会**卡住**正在 decode 的请求（Q6 会详谈）。

PD 分离的做法：
- **Prefill 集群**：用算力强的卡，追求 TTFT，可以用大 TP
- **Decode 集群**：用显存大、带宽高的卡，追求吞吐，可以用大 batch
- 中间通过高速网络**传 KV Cache**

代表工作：**DistServe、Splitwise、Mooncake（月之暗面）、DeepSeek-V3 的推理部署**。代价是 KV Cache 跨机传输的带宽压力——所以 MLA 这种小 KV Cache 的架构在 PD 分离下优势更大。**这个点能答出来是加分项**，第 15 章会展开。

**Q：Decode 阶段的 Attention 计算，和 Prefill 的 Attention 有什么不同？为什么要单独写 kernel？**

A：形状完全不同，所以最优 kernel 不同。

- **Prefill**：Q 是 $[n, d]$，算 $[n, n]$ 的 attention 矩阵 → 是个标准 GEMM，**FlashAttention** 通过 tiling 把 $[n,n]$ 矩阵留在 SRAM 里、不落 HBM，收益巨大。
- **Decode**：Q 只有 $[1, d]$，算 $[1, t]$ → 本质是 **GEMV**，attention 矩阵只有一行，根本不存在"$[n,n]$ 矩阵太大"的问题。FlashAttention 那套 tiling 意义不大。

Decode 的真瓶颈是**读 KV Cache**（$t$ 越长读得越多）。所以有专门的 **FlashDecoding / FlashDecoding++**：把 KV 序列维度**切成多段并行**，再 reduce 合并——因为 batch=1 时 GPU 的 SM 大量闲置，切 KV 维度才能把并行度提上来。

vLLM 的 `paged_attention` kernel、FlashInfer 都是为 Decode 专门写的。**面试时能说清"Prefill 用 FlashAttention、Decode 用 FlashDecoding，因为一个是 GEMM 一个是 GEMV"，是很强的信号**。

### ⚠️ 常见陷阱

1. **不要说"Decode 慢是因为要算 attention"** —— 错。Decode 慢主要是**读权重**（batch 小时占大头），KV Cache 的读取在**长上下文 + 大 batch** 时才成为主要矛盾。要分清。
2. **不要把 TTFT 和 TPOT 混着说** —— "推理延迟"这个词在面试里是模糊的，主动拆成 TTFT / TPOT 是专业度的体现。
3. **不要忘了 Prefill 也产出 KV Cache** —— 很多人只说"Prefill 算出第一个 token"，漏掉了"顺便把 n 个位置的 KV 存下来"这个更重要的产出。

### 🏢 大厂偏好

- **字节 / 阿里（推理引擎、AI Infra 岗）**：必问，且会追问 PD 分离
- **应用岗 / Agent 岗**：会从业务角度问——"用户抱怨慢，你怎么定位是 TTFT 还是 TPOT 的问题"
- **DeepSeek**：会把这题引向 MLA 和 PD 分离的关系

### 📚 延伸阅读
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - vLLM 论文 §2 背景部分是最好的两阶段综述
- [DistServe](https://arxiv.org/abs/2401.09670) - PD 分离的代表工作
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)

---

## Q2：KV Cache 原理与显存估算 ⭐⭐⭐⭐⭐（必问，可能手撕）

### 🎯 一句话标答

> KV Cache 是**用显存换算力**——把历史 token 的 K、V 存下来，让每步 Decode 从 $O(t^2)$ 重算降到 $O(t)$ 读取；代价是显存占用随 **batch × 序列长度**线性增长，公式是 $2 \times L \times n_{kv} \times d_h \times \text{dtype}$ 字节/token。

### 🗣️ 30 秒口语版

"KV Cache 的动机很直接——**避免重复计算**。

自回归生成第 t 个 token 时，需要当前的 Q 去和**前面所有 token 的 K、V** 做 attention。而前面那些 token 的 K、V，在之前的步骤里**已经算过一模一样的了**——因为 causal mask 保证了它们不受后面 token 影响，**永远不会变**。

所以把它们缓存起来，每步只算新 token 的那**一行** K、V，追加到 Cache 后面。

**收益**：不用 Cache 的话，生成 n 个 token 总复杂度是 $O(n^3 d)$（每步重算全部）；用了 Cache 是 $O(n^2 d)$。差一个数量级。

**代价**：显存。公式我可以直接写——

**每 token 每层** = 2（K 和 V）× num_kv_heads × head_dim × dtype_bytes
**总量** = 上面这个 × num_layers × batch × seq_len

举个数：**LLaMA-2 7B，fp16，MHA**——32 层、32 个 KV 头、head_dim 128：
2 × 32 × 128 × 2 = 16 KB/层/token，× 32 层 = **512 KB/token**。

4096 长度的单条请求就要 **2 GB**。A100-80G 装完 14 GB 权重还剩 66 GB，理论上只能塞 **33 条并发**。这就是为什么 KV Cache 是 LLM serving 的**头号显存杀手**，也是为什么会有 GQA、MLA、KV 量化、PagedAttention 这一整个技术族。"

### 📐 完整推导

#### 为什么 K、V 可以缓存，Q 不行？

关键在 causal mask。第 $i$ 个位置的 K、V 只依赖 $x_i$ 本身（$K_i = x_i W^K$），**不依赖任何后来的 token**——所以一旦算出，永远不变，可以缓存。

而 Q 呢？第 $t$ 步只需要 $Q_t$ 这一个新的 query，历史的 $Q_1 ... Q_{t-1}$ **再也用不到了**——它们的输出早就算完输出了。所以 **Q 不需要缓存，缓存了也没用**。

$$\text{Attn}(Q_t, K_{1:t}, V_{1:t}) = \text{softmax}\left(\frac{Q_t K_{1:t}^\top}{\sqrt{d_h}}\right) V_{1:t}$$

看这个式子：$Q$ 下标只有 $t$，$K, V$ 下标是 $1{:}t$ ——**谁需要历史，谁就要缓存**。

#### 显存公式（必须能默写）

$$\text{KVCache}_{\text{bytes}} = 2 \times B \times S \times L \times n_{kv} \times d_h \times \text{sizeof(dtype)}$$

| 符号 | 含义 |
|---|---|
| 2 | K 和 V 两份 |
| $B$ | batch size |
| $S$ | 序列长度（prompt + 已生成） |
| $L$ | 层数 |
| $n_{kv}$ | **KV 头数**（注意不是 Q 头数！GQA 下两者不同） |
| $d_h$ | head_dim |

> **最容易出错的地方**：GQA 下 $n_{kv} \neq n_q$。面试官很喜欢用 LLaMA-2 70B 挖这个坑。

#### 主流模型实测（fp16，每 token）

| 模型 | 层数 | KV 头数 | head_dim | KV 维度/层 | **每 token** | 4K 上下文 |
|---|---|---|---|---|---|---|
| LLaMA-2 7B (MHA) | 32 | 32 | 128 | 4096 | **512 KB** | 2.0 GB |
| LLaMA-2 13B (MHA) | 40 | 40 | 128 | 5120 | **800 KB** | 3.1 GB |
| **LLaMA-2 70B (GQA-8)** | 80 | **8** | 128 | 1024 | **320 KB** | 1.25 GB |
| LLaMA-3 8B (GQA-8) | 32 | 8 | 128 | 1024 | **128 KB** | 0.5 GB |
| DeepSeek-V3 (MLA) | 61 | — | — | 576 元素/层 | **≈ 70 KB** | 0.27 GB |

> 🚨 **注意第 3 行的反直觉结论**：**70B 的 KV Cache 比 7B 还小**（320 KB vs 512 KB）！因为 70B 用了 GQA-8。这是个绝佳的面试素材——它说明 **KV Cache 大小和模型参数量没有直接关系，只和"KV 头数 × 层数"有关**。
>
> MLA 那一行的 576 见第 2 章的推导（$d_c=512$ 压缩向量 + $d_h^R=64$ 解耦 RoPE 部分）。**MLA 相对 MHA 是两个数量级的差距**——这就是 DeepSeek 能开出极低 API 价格的底层原因之一。

#### 显存全景：KV Cache 到底占多少？

A100-80G 跑 LLaMA-2 7B fp16：

```
模型权重          : 14 GB   （固定）
激活 + 临时 buffer:  2 GB   （固定，看 batch）
─────────────────────────────
剩给 KV Cache     : 64 GB   ← 这就是你的"并发预算"

单条 4K 请求      : 2 GB
→ 理论最大并发    : 32 条

如果换成 GQA-8 的 LLaMA-3 8B（128 KB/token）:
单条 4K 请求      : 0.5 GB
→ 理论最大并发    : 128 条   （4 倍！）
```

**这就是"KV Cache 决定并发数、并发数决定吞吐、吞吐决定单位成本"的完整链条。** 面试时能把这条链讲出来，比背公式强十倍。

### 💻 关键代码片段

**（一）显存估算函数——手撕高频**

```python
def kv_cache_bytes(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_kv_heads: int,   # 注意：GQA 下 != num_attention_heads
    head_dim: int,
    dtype_bytes: int = 2,  # fp16/bf16=2, fp8/int8=1, fp32=4
) -> int:
    """KV Cache 显存占用（字节）"""
    per_token_per_layer = 2 * num_kv_heads * head_dim * dtype_bytes
    return per_token_per_layer * num_layers * batch_size * seq_len


# LLaMA-2 7B, batch=1, 4K 上下文
n = kv_cache_bytes(1, 4096, 32, 32, 128)
print(f"{n / 1024**3:.2f} GB")   # → 2.00 GB

# LLaMA-2 70B (GQA-8), 同样条件
n = kv_cache_bytes(1, 4096, 80, 8, 128)
print(f"{n / 1024**3:.2f} GB")   # → 1.25 GB  ← 比 7B 还小
```

**（二）最朴素的 KV Cache 实现（体会"追加"这个动作）**

```python
class NaiveKVCache:
    def __init__(self):
        self.k = None   # [B, n_kv, S, d_h]
        self.v = None

    def update(self, new_k, new_v):
        """new_k/new_v: [B, n_kv, 1, d_h]  —— decode 时只有 1 个新 token"""
        if self.k is None:
            self.k, self.v = new_k, new_v
        else:
            # ⚠️ 这里就是问题所在：cat 会重新分配显存并拷贝整个 Cache
            self.k = torch.cat([self.k, new_k], dim=2)
            self.v = torch.cat([self.v, new_v], dim=2)
        return self.k, self.v


def decode_step(x, cache, W_q, W_k, W_v):
    q = x @ W_q                       # [B, 1, d]  只算新 token 的 Q
    new_k, new_v = x @ W_k, x @ W_v   # [B, 1, d]  只算新 token 的 K/V
    k, v = cache.update(new_k, new_v) # 拿到 [B, t, d] 的完整历史
    attn = softmax(q @ k.transpose(-1, -2) / d_h**0.5) @ v
    return attn
```

> **这段代码里藏着两个坑，正好引出后面两题**：
> 1. `torch.cat` **每步都重新分配 + 全量拷贝** → 所以真实实现要**预分配**
> 2. 但预分配要按 `max_seq_len` 来 → **巨量浪费** → 所以有了 **PagedAttention**（Q4）

### 🔥 高频追问 Top 3

**Q：为什么 KV Cache 是"用显存换算力"，不是纯粹的优化？有没有反而变慢的情况？**

A：有，而且不罕见。KV Cache 的收益是**省算力**，成本是**多读显存**。当序列非常长、batch 非常大时，**读 KV Cache 的时间会超过重算的时间**。

粗略的临界点分析（batch=1，7B 模型）：
- 读权重：14 GB（固定）
- 读 KV Cache：$S \times 512\text{KB}$

当 $S = 28{,}000$ 时，读 KV Cache 的量就等于读权重了。再长下去，**KV Cache 成为 Decode 的主要瓶颈**，而不是权重。

这解释了三件事：
1. 为什么**长上下文场景**（128K）下 Decode 会明显变慢——不是算力不够，是在搬 KV
2. 为什么**长上下文场景下 GQA/MLA 的价值被放大**——省的不只是显存，更是带宽
3. 为什么会有 **KV Cache 量化**（第 13 章）、**KV Cache 稀疏化 / 驱逐**（H2O、StreamingLLM、SnapKV）这些技术

**Q：KV Cache 能不能不存全部？可以丢掉一部分吗？**

A：可以，这是一个活跃的研究方向，核心思路是"**attention 是稀疏的，大部分历史 token 拿不到权重**"。

主流几派——
- **StreamingLLM**：发现有 **attention sink** 现象（开头几个 token 拿走大量注意力，即使语义无关）。方案：**只保留最开头的几个 token + 最近的滑动窗口**，中间全丢。能无限长流式生成，但**真的会丢中间信息**——适合聊天，不适合长文档 QA。
- **H2O (Heavy Hitter Oracle)**：用累积 attention 分数识别"重要 token"，动态驱逐低分的。
- **SnapKV / PyramidKV**：在 Prefill 结束时一次性压缩，且**不同层保留不同比例**（浅层保留多、深层保留少）。
- **量化而非丢弃**：KIVI、KVQuant——把 KV Cache 压到 int4/int2，不丢 token 只丢精度。

**面试时的正确姿态**：这些都是**有损**的，工业上默认不开。要说清"**在什么场景下我愿意接受这个损失**"——比如超长上下文的摘要任务可以，法律文档精确检索不行。这种权衡意识比背名词重要。

**Q：MQA / GQA / MLA 都是为了压 KV Cache，为什么最后主流是 GQA 而不是更狠的 MQA？**

A：这题是第 1 章的内容，但在推理场景下有个新角度——**MQA 压得太狠，质量掉得明显，而且收益边际递减**。

从 MHA（32 头）→ GQA-8：KV Cache 降 **4 倍**，质量几乎无损。
从 GQA-8 → MQA（1 头）：再降 **8 倍**，但质量明显下降。

而且有个工程细节：**TP（张量并行）下 MQA 很尴尬**——只有 1 个 KV 头，没法在 8 张卡之间切分，只能每张卡复制一份，白白浪费显存。**GQA-8 正好可以在 TP=8 时每卡一个头**，这个"凑巧"是 GQA 被广泛采用的隐藏原因之一。

**MLA 是另一条路**：不减头数，而是**压维度**，所以能同时拿到"KV Cache 极小"和"等效多头质量"。代价是实现复杂（RoPE 解耦，见第 2 章）。

### ⚠️ 常见陷阱

1. **公式里用 $n_q$ 而不是 $n_{kv}$** —— 最高频的翻车点。面试官问 LLaMA-2 70B 的 KV Cache，你要是按 64 个头算就露馅了（实际是 GQA-8）。
2. **忘了乘 2** —— K 和 V 是两份。
3. **说"KV Cache 存的是 attention 结果"** —— 错。存的是 **K 和 V 这两个投影后的中间张量**，不是 attention 输出，也不是 attention 权重矩阵。
4. **忽略 dtype** —— fp16 是 2 字节，但如果引擎开了 fp8 KV Cache，就减半了。答的时候把 dtype 说出来是专业度。
5. **说"KV Cache 是推理的优化技巧"** —— 更准确的说法是：**它是自回归推理的必需品**，不用它慢一个数量级。真正的"技巧"是怎么**管理**它。

### 🏢 大厂偏好

- **所有大厂**：**这是推理部分的必考第一题**，且**极大概率要求现场算数**
- **字节 / 阿里 Infra**：会给具体配置让你算显存和最大并发（上面"显存全景"那段就是标准答法）
- **DeepSeek**：一定会引向 MLA
- **应用岗**：会问"我们要支持 100 并发 × 32K 上下文，需要几张卡"——这题的实战形态

### 📚 延伸阅读
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) - MQA 原论文，KV Cache 瓶颈的最早系统性分析
- [GQA](https://arxiv.org/abs/2305.13245)
- [StreamingLLM](https://arxiv.org/abs/2309.17453) - attention sink 现象
- [SnapKV](https://arxiv.org/abs/2404.14469)

---

## Q3：Decode 为什么是 Memory-Bound？（Roofline 分析）⭐⭐⭐⭐⭐（必问）

### 🎯 一句话标答

> 因为 Decode 每步的**算术强度（Arithmetic Intensity）约等于 1 FLOP/byte**——每读 2 字节权重只做 2 次浮点运算——而 A100 的机器平衡点是 **~156 FLOP/byte**。差了两个数量级，所以 GPU 的算力有 **99% 在空转**，全部时间都花在从 HBM 搬权重上。

### 🗣️ 30 秒口语版

"这题要用 **Roofline 模型**来答，答好了是很强的信号。

**算术强度**的定义：`总 FLOPs ÷ 总访存字节数`。这个比值决定了你是 compute-bound 还是 memory-bound。

**GPU 的机器平衡点**：A100 fp16 峰值算力约 312 TFLOPS，HBM 带宽约 2 TB/s。相除得到 **156 FLOP/byte**——意思是"**每从显存读 1 字节，你至少得做 156 次浮点运算，才配得上这张卡的算力**"。低于这个数，你就是在浪费算力。

**Decode 的算术强度**：
- 每个参数做 1 次乘 + 1 次加 = **2 FLOPs**
- 每个参数 fp16 占 **2 字节**
- 算术强度 = 2 / 2 = **1 FLOP/byte**

**1 vs 156——差 156 倍。** 所以 GPU 的 tensor core 基本在睡觉，全部时间在等 HBM。

**直接推论（这句一定要说）**：既然是 memory-bound，那 Decode 的速度就有个**硬上限**，跟你算力多强无关：

> **上限 = HBM 带宽 ÷ 模型大小**

7B fp16 = 14 GB，A100 带宽 2000 GB/s → **最快 143 token/s**。任何声称 batch=1 单卡跑 7B 超过这个数的，要么用了量化（减小模型）、要么用了投机解码（一次出多个 token）——**不可能有第三种**。

**这也直接给出了三条优化路线**：
1. **减小分子（模型大小）** → **量化**（第 13 章）
2. **提高算术强度** → **增大 batch**（Q5 continuous batching），batch=B 时强度变成 B FLOP/byte
3. **一次多出几个 token** → **投机解码**（第 14 章）

整个第三部分的技术地图，其实就是这个公式的三个变量。"

### 📐 Roofline 详细分析

#### 机器平衡点（Machine Balance）

$$\text{Machine Balance} = \frac{\text{Peak FLOPS}}{\text{Memory Bandwidth}}$$

| GPU | 峰值算力 (fp16/bf16) | HBM 带宽 | **平衡点** |
|---|---|---|---|
| A100-80G | 312 TFLOPS | 2039 GB/s | ~153 FLOP/byte |
| H100-SXM | 989 TFLOPS | 3350 GB/s | ~295 FLOP/byte |
| H800 | 989 TFLOPS | 3350 GB/s | ~295 FLOP/byte |
| L40S | 362 TFLOPS | 864 GB/s | ~419 FLOP/byte |

> 🚨 **注意趋势**：新卡的**算力涨得比带宽快**（H100 相对 A100，算力 3.2×，带宽只有 1.6×）。这意味着**平衡点越来越高，memory-bound 的问题越来越严重**。这是"内存墙"的具体体现，也是为什么推理优化的重要性逐年上升。**这个趋势能答出来是很强的加分**。

#### Decode 的算术强度（含 batch）

Batch = $B$ 时：
- FLOPs = $2 \cdot P \cdot B$（每条序列都要过一遍全部参数）
- 访存 = $P \times 2$ 字节（**权重只读一次，全 batch 共享！**）

$$\text{Intensity} = \frac{2PB}{2P} = B \ \text{FLOP/byte}$$

**极其漂亮的结论：算术强度 ≈ batch size。**

| batch | 算术强度 | A100 状态 | 吞吐（7B, token/s） |
|---|---|---|---|
| 1 | 1 | 严重 memory-bound（算力利用 <1%） | ~140 |
| 8 | 8 | memory-bound | ~1100 |
| 32 | 32 | memory-bound | ~4500 |
| **~150** | ~150 | **接近平衡点** | 接近算力上限 |
| 256 | 256 | compute-bound（但 KV Cache 可能已爆显存） | 不再线性增长 |

**这张表回答了两个面试题**：
1. **"为什么大 batch 能提高吞吐？"** —— 因为权重只读一次被全 batch 摊薄，算术强度线性上升。
2. **"batch 是不是越大越好？"** —— 不是。两个限制：**(a)** 超过平衡点后变 compute-bound，收益饱和；**(b)** 更现实的是 **KV Cache 先爆显存**——batch=150 × 4K 上下文的 7B 模型需要 300 GB KV Cache，A100 根本装不下。

> **所以真正的故事是**：理论上 batch 要到 150 才能吃满算力，但**显存只允许你到 32**。**KV Cache 才是限制吞吐的真凶**——这就是 PagedAttention 存在的全部意义（Q4）。这个逻辑闭环能讲出来，这一整章就通了。

#### 上界公式（记住它）

$$\text{TPOT}_{\min} = \frac{\text{Model Size (bytes)}}{\text{HBM Bandwidth}}$$

```
7B fp16 (14 GB) on A100 (2000 GB/s):  7.0 ms/token → 143 tok/s
7B int4 (3.5 GB) on A100:             1.75 ms/token → 571 tok/s   ← 量化 4× 加速的来源
70B fp16 (140 GB) on 8×A100 (TP=8):   140/8/2000 = 8.75 ms → 114 tok/s
```

> 第 3 行有个细节：TP=8 时每张卡只存 1/8 的权重，**8 张卡的带宽是并行叠加的**——这是 TP 在推理端的真正价值：**不只是为了装得下，更是为了把总带宽乘 8**。这个点第 15 章展开。

### 💡 直觉理解

**Roofline 就是"水管 vs 水泵"**。

GPU 的算力是水泵（每秒能处理多少水），HBM 带宽是水管（每秒能送多少水过来）。

- **Prefill**：一次送 1000 个 token 的活儿过来，水泵拼命抽 → **水泵是瓶颈**（compute-bound）
- **Decode batch=1**：水管里送来一整本教科书（14 GB 权重），水泵只用其中一句话就算完了，剩下时间在等下一批水 → **水管是瓶颈**（memory-bound）
- **Decode batch=32**：同样一本教科书送过来，水泵一次给 32 个人服务 → **水管成本被摊薄 32 倍**

所以"提高 batch"的本质不是"算得更多"，而是"**同样的搬运，干更多的活**"。

### 🔥 高频追问 Top 3

**Q：既然 Decode 是 memory-bound，那用算力更弱但带宽更高的卡是不是更划算？**

A：**方向对，这正是推理卡的设计哲学**，但要分两种情况说：

- **纯 Decode 场景**（PD 分离后的 D 集群）：确实应该选**高带宽、大显存**的卡。所以你会看到 H100 的 HBM3 (3.35 TB/s)、MI300X 的 5.3 TB/s、以及各种"推理专用卡"都在堆带宽和容量。
- **但混合部署时**：Prefill 还是要算力。所以通用卡才是主流。

另一个角度：**这也是 Groq / Cerebras 这类"用 SRAM 替代 HBM"的架构存在的理由**——SRAM 带宽是 HBM 的几十倍，直接把 memory wall 拆了，代价是容量小到必须用几百张卡拼一个模型。**能提到这个是很好的深度信号。**

**Q：MoE 模型的算术强度怎么算？和 Dense 有什么不同？**

A：**这是个陷阱题，答对了很加分**。

MoE 的 FLOPs 按**激活参数**算，但**访存要看情况**：
- **batch=1**：只激活 top-k 个专家，**只需要读这几个专家的权重**。所以 MoE 在小 batch 下 memory 访问也少，Decode 反而**比同总参数的 Dense 快很多**——这是 MoE 推理友好的核心原因。
- **batch 很大时**：不同序列会路由到**不同的专家**，batch 越大，被激活的专家并集越接近全部。极端情况下**要读全部专家的权重，但每个专家只服务少数几条序列**——算术强度不但没随 batch 线性上升，反而**塌了**。

所以 **MoE 的算术强度增长曲线比 Dense 差**，需要 **EP（专家并行）+ 大 batch + 良好的负载均衡** 才能跑好。这就是为什么 DeepSeek-V3 的推理部署要用**很大的 EP 规模**（部署时 EP 甚至到几百）——**为了让每个专家都攒够 batch**。见第 15 章和 Frontier Topics 的 MoE 部分。

**Q：算术强度这个分析对 Attention 部分也成立吗？**

A：不完全，要分开算——这也是为什么长上下文的行为不一样。

- **权重部分**（FFN + 投影）：算术强度 = $B$（batch 共享权重，如上）
- **Attention 部分**：每条序列有**自己的 KV Cache，不共享**！所以 batch 加大时，KV 的访存也线性加大 → **attention 部分的算术强度恒等于 ~1，不随 batch 改善**。

**推论**：随着 batch 和序列长度增大，**attention 部分逐渐成为 Decode 的主导瓶颈**，而且它是 batching 救不了的。这解释了：
1. 为什么长上下文 + 大 batch 下吞吐会"塌方"
2. 为什么 GQA / MLA 的价值在长上下文下被放大（**它们直接减少 attention 的访存**）
3. 为什么会有 FlashDecoding 这种专门优化 Decode attention 的 kernel

**"batching 能救权重，救不了 attention"——这句话能说出来，说明你是真懂了。**

### ⚠️ 常见陷阱

1. **不要只说"Decode 慢因为不能并行"** —— 这只答对了一半（自回归依赖），没答到硬件层面（memory-bound）。面试官要的是后者。
2. **不要把 batch 无限吹** —— 一定要主动说出"KV Cache 显存先爆"这个约束，否则显得没有工程经验。
3. **不要忽略 attention 和权重的算术强度不同** —— 这是区分"看过博客"和"真做过"的题眼。
4. **数字不要记死** —— 记 **A100 ≈ 150 FLOP/byte、7B fp16 单卡 ≈ 140 tok/s 上限** 这两个锚点就够，其他现场推。

### 🏢 大厂偏好

- **字节 / 阿里 AI Infra、推理引擎岗**：必问，且会现场让你算 Roofline
- **Meta（Production Engineer / Inference）**：会问 GPU 型号相关的具体数字
- **DeepSeek**：会引向 MoE 的算术强度和 EP 部署
- **应用岗**：不会问得这么深，但**能主动说出这套分析，会显著抬高面试官对你的评级**——因为这证明你不是只会调 API

### 📚 延伸阅读
- [Roofline: An Insightful Visual Performance Model](https://dl.acm.org/doi/10.1145/1498765.1498785) - Roofline 原论文（2009，非 LLM，但是思想源头）
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) - Google 的推理 scaling 分析，Roofline 用得最漂亮的一篇
- [LLM Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) - 最好的入门博客

---

## Q4：PagedAttention 与显存碎片 ⭐⭐⭐⭐⭐（必问）

### 🎯 一句话标答

> 传统实现要为每条请求**按 max_seq_len 预分配连续显存**，导致 **60–80% 的 KV Cache 显存被浪费**在内部碎片、外部碎片和无法共享上；PagedAttention 借鉴**操作系统虚拟内存的分页**思想，把 KV Cache 切成固定大小的 block、**物理不连续、用 block table 映射**，把浪费降到 **4% 以下**，从而把并发数（进而吞吐）提升 **2–4 倍**。

### 🗣️ 30 秒口语版

"要先讲清**问题**，PagedAttention 才有意义。

**问题的根源**：KV Cache 是**动态增长**的——你事先不知道用户会生成多少 token。但传统实现（HuggingFace、早期 FasterTransformer）要求 KV Cache 是**一块连续显存**（因为 attention kernel 要连续访问）。

于是只能**按最坏情况预分配**——比如 max_seq_len=2048，那不管用户实际生成多少，先占 2048 个 token 的显存。

**三种浪费**：
1. **内部碎片（Internal Fragmentation）**：预留了 2048，实际只生成了 100 个 token → **95% 白占**。这是最大头。
2. **外部碎片（External Fragmentation）**：不同请求的预留块大小不一，中间留下的空隙塞不进新请求 → 显存明明够，但"装不下"。
3. **无法共享**：两个请求用同一个 system prompt，或者并行采样（n=4）从同一个 prompt 分叉，本来 KV 完全一样，但因为要求连续，只能**各存一份**。

vLLM 论文测出来：**这三种浪费加起来，真正有效的 KV Cache 显存只有 20–40%**。

**PagedAttention 的解法——直接抄操作系统**：

操作系统怎么解决进程内存碎片的？**虚拟内存 + 分页**。PagedAttention 一模一样：
- KV Cache 切成固定大小的 **block**（vLLM 默认 **16 个 token 一块**）
- 每条序列有一张 **block table**（就是页表），记录"逻辑 block → 物理 block"的映射
- 物理 block 在显存里**可以完全不连续**
- **按需分配**：写满一个 block 才申请下一个

**收益**：
- **内部碎片 ≤ 1 个 block**（最多浪费 15 个 token 的空间）→ 浪费从 60-80% 降到 **<4%**
- **外部碎片彻底消失**（所有块一样大，任意可用块都能用）
- **能共享**：多个 block table 可以指向同一个物理 block → 加上 **copy-on-write**，并行采样、beam search、system prompt 全都能共享

**代价**：attention kernel 要重写——不能再假设 KV 连续，必须**先查 block table 再 gather**。这就是为什么它叫 Paged**Attention** 而不是 PagedCache——**核心工作量在 kernel 里**。

**最终效果**：vLLM 论文报告相对 HF Transformers 和 Orca，吞吐提升 **2–4 倍**。这就是 vLLM 一战成名的原因。"

### 📐 详细机制

#### 浪费从哪来——一张图说清

```
【传统：连续预分配，max_seq_len = 2048】

请求 A (实际生成 100 token):
┌──────┬───────────────────────────────────────┐
│ 已用  │        预留但永远用不上                │
│ 100  │              1948                     │   ← 内部碎片：95% 浪费
└──────┴───────────────────────────────────────┘

请求 B (已结束，释放):
┌──────────────────┐
│      空洞         │   ← 外部碎片：新请求要 2048 连续，这里只有 800，用不了
└──────────────────┘

请求 C, D 用同一个 system prompt (500 token):
C: [system 500][C 自己的...]     ┐
D: [system 500][D 自己的...]     ┘  ← 同样的 500 token 存了两份


【PagedAttention：分页，block = 16 token】

请求 A (实际 100 token) → 只分配 ⌈100/16⌉ = 7 个 block
  逻辑视图: [blk0][blk1][blk2][blk3][blk4][blk5][blk6]
  block table:  ↓     ↓     ↓     ↓     ↓     ↓     ↓
  物理显存:   #42   #7   #103  #15  #88   #3   #61   ← 完全不连续，无所谓
  浪费 = 7×16 - 100 = 12 个 token 的空间 (1.7%)   ← 内部碎片只剩最后一块

请求 C, D 共享 system prompt:
  C 的 block table: [#5][#6][#7]...[C 独有的块]
  D 的 block table: [#5][#6][#7]...[D 独有的块]
                     ↑ 同一批物理块，refcount=2，只存一份
```

#### Block Table 的核心数据结构

```python
# 概念示意（非 vLLM 真实代码，但结构等价）

BLOCK_SIZE = 16   # 每个 block 存 16 个 token 的 KV

class PhysicalBlock:
    block_id: int
    ref_count: int      # 被多少条序列引用 → copy-on-write 的关键

class Sequence:
    block_table: List[int]   # 逻辑块 i → 物理块 block_table[i]
    num_tokens: int

class BlockManager:
    def __init__(self, total_gpu_blocks: int):
        self.free_blocks = list(range(total_gpu_blocks))
        self.blocks = {}

    def append_token(self, seq: Sequence):
        """写入一个新 token，必要时才申请新块"""
        offset = seq.num_tokens % BLOCK_SIZE
        if offset == 0:
            # 当前块正好写满 → 申请新块（唯一的分配时机）
            if not self.free_blocks:
                raise OutOfMemory   # → 触发抢占：swap 或 recompute
            seq.block_table.append(self.free_blocks.pop())
        seq.num_tokens += 1

    def fork(self, parent: Sequence) -> Sequence:
        """并行采样 / beam search 分叉 —— 零拷贝"""
        child = Sequence(block_table=parent.block_table.copy(),
                         num_tokens=parent.num_tokens)
        for blk in child.block_table:
            self.blocks[blk].ref_count += 1   # 只加引用计数，不拷数据
        return child

    def write_with_cow(self, seq: Sequence, logical_idx: int):
        """Copy-on-Write：只有真要写共享块时才复制"""
        blk = seq.block_table[logical_idx]
        if self.blocks[blk].ref_count > 1:
            new_blk = self.free_blocks.pop()
            copy_block(src=blk, dst=new_blk)          # 只复制这一块，不是整个序列
            self.blocks[blk].ref_count -= 1
            seq.block_table[logical_idx] = new_blk
```

> **注意 `fork` 和 `write_with_cow` 这两个方法**——它们是"操作系统类比"最精彩的部分：`fork` 对应进程 fork，COW 对应写时复制。**面试时点出"vLLM 的并行采样是零拷贝的 fork + COW"，比只说"分页"深一层。**

#### PagedAttention Kernel 干了什么

普通 attention kernel：
```
K = kv_cache[seq_id]                 # 一次性拿到连续的 [t, d]
scores = Q @ K.T
```

Paged 版本：
```
for logical_blk in range(num_blocks):
    physical_blk = block_table[seq_id][logical_blk]   # ① 查页表
    K_blk = kv_cache_pool[physical_blk]               # ② gather 一块 [16, d]
    scores_blk = Q @ K_blk.T                          # ③ 算这一块
    # ④ online softmax 增量合并（和 FlashAttention 的 rescaling 同一套技术）
```

**这里有个精妙之处**：因为要**分块处理 + 增量合并 softmax**，PagedAttention 天然要用 **online softmax**——和 FlashAttention 是同一套数学。所以 PagedAttention 可以理解为 "**FlashAttention 的思想 + 分页的内存布局**"。

**Block size 的权衡**：
- 太小（如 1）：block table 巨大，查表开销高，kernel 里循环次数多
- 太大（如 256）：退化回连续分配，内部碎片重新变大
- **16 是 vLLM 的默认值**——经验最优点。这个数字最好记住。

### 🔥 高频追问 Top 3

**Q：显存真的用完了（free_blocks 空了）怎么办？**

A：**抢占（Preemption）**。vLLM 有两种策略：

1. **Swapping（换出到 CPU 内存）**：把某条序列的 KV block 拷到 host memory，让出显存；等有空间了再拷回来。
   - 优点：不用重算
   - 缺点：**PCIe 带宽只有 ~32 GB/s，比 HBM 慢 60 倍**，拷来拷去很痛
2. **Recomputation（丢弃后重算）**：直接丢掉 KV Cache，等恢复时重新 Prefill 一遍。
   - 优点：不占 PCIe
   - 缺点：白算一遍
   - **但注意**：重算是 Prefill（compute-bound、并行），而 swap 是搬数据（PCIe-bound）。**在多数配置下重算反而更快**——这个反直觉结论 vLLM 论文里实测过。

**抢占哪条序列？** vLLM 默认 **FCFS 的反向**——最晚来的先被踢（避免饿死老请求），且**整条序列一起踢**（因为一条序列的 block 是有依赖的，踢一半没意义）。

**Q：PagedAttention 有没有性能开销？为什么它还能更快？**

A：**有开销，但被收益淹没了**。

**开销**：
- 查 block table 的间接寻址（一次额外的内存访问）
- kernel 里非连续 gather，访存局部性差一点
- vLLM 论文测：**单条序列的 attention kernel 本身慢约 20–26%**

**收益**：
- 显存浪费从 60-80% → <4% → **能同时跑的序列多 2-4 倍**
- 而 Decode 是 memory-bound，**batch 翻 2-4 倍 ≈ 吞吐翻 2-4 倍**（见 Q3）

**净效果：吞吐 2-4 倍。** 这就是典型的"**牺牲单点效率、换取系统吞吐**"——用 20% 的 kernel 性能换 4 倍的并发，血赚。

**面试时能主动说"PagedAttention 的 kernel 其实更慢，赢在能塞更多 batch"，是非常强的信号**——说明你理解 Q3 的 Roofline，而不是把它当成一个"加速技巧"。

**Q：PagedAttention 和 MLA 冲突吗？DeepSeek 怎么做 paging？**

A：不冲突，但**block 的内容变了**。

- MHA/GQA 下：一个 block 存 16 个 token 的 **K 和 V**
- MLA 下：一个 block 存 16 个 token 的**压缩潜向量 $c_t$ + 解耦 RoPE 部分 $K^R$**（见第 2 章）

分页逻辑（block table、COW、抢占）**完全复用**，只是每个 block 的字节数从 16×512KB/32层 变成小得多的值。

**有意思的推论**：MLA 让每 token 的 KV 从 ~500 KB 降到 ~70 KB，意味着**同样的显存能装 7 倍的序列**。而 PagedAttention 让这些显存"真的能用满"。**两者是乘法关系，不是重复投入**——这也是 DeepSeek 推理成本极低的组合拳。

vLLM/SGLang 对 MLA 的支持经历过一轮专门适配（MLA 的 paged kernel 和 GQA 的不是同一个），这个工程细节在 Infra 岗会被问到。

### ⚠️ 常见陷阱

1. **只说"分页"，不说"为什么必须分页"** —— 必须先讲清**三种碎片**，尤其是"**因为不知道要生成多少 token，所以只能按 max_len 预分配**"这个根因。不讲根因就是背名词。
2. **忘了 Copy-on-Write / 共享** —— 这是 PagedAttention 的"第二半"，很多人只记得碎片，忘了共享。而共享正好是 Prefix Caching（Q7）的基础。
3. **说 PagedAttention 让 attention 算得更快** —— **反了**！它让 attention 算得**更慢**（-20%），但让系统吞吐更快（+2~4×）。这个点答反了会被扣分，答对了会加分。
4. **把 PagedAttention 和 Continuous Batching 混为一谈** —— 两个正交的技术：**PagedAttention 管显存怎么放**，**Continuous Batching 管请求怎么调度**。vLLM 两个都做了，但它们是不同的层。面试里经常一起问，要能分开讲。

### 🏢 大厂偏好

- **字节 / 阿里（推理引擎、AI Infra）**：**必问，且是深挖题**——会追问 block size 怎么选、抢占策略、kernel 实现
- **应用岗 / Agent 岗**：会问"为什么用 vLLM 不用 HF pipeline"——这题就是标准答案
- **Meta**：会从系统设计角度问 OS 类比
- **DeepSeek**：会问 MLA 下的 paging 适配

### 📚 延伸阅读
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - **vLLM 原论文，本章最该精读的一篇**，§3 是核心
- [vLLM 官方文档 - Optimization and Tuning](https://docs.vllm.ai/)

---

## Q5：Continuous Batching（连续批处理）⭐⭐⭐⭐⭐（必问）

### 🎯 一句话标答

> **Static Batching** 要等一批里最慢的请求（生成最长的那条）全部结束才能换下一批，导致早完成的位置**空转等待**；**Continuous Batching** 把调度粒度从"**批**"降到"**每一次前向迭代**"——任何序列一结束就立刻踢出、立刻补入新请求，**让 GPU 永远不留空位**，吞吐可提升数倍到十几倍。

### 🗣️ 30 秒口语版

"这题的关键是理解 **static batching 为什么烂**。

**Static Batching（传统做法）**：攒够 8 条请求，一起前向，一起 decode，**等 8 条全部生成完**，再换下一批。

问题在于——LLM 的**输出长度是不可预测的**，方差极大。同一批里可能有：
- 请求 A：回答"是的" → 3 个 token 就结束了
- 请求 B：写一篇文章 → 1000 个 token

那从第 4 步到第 1000 步，**A 的那个 batch 槽位就一直是废的**——填着 padding，白白占着显存、白白参与计算。而且这期间**新来的请求只能排队等着**，哪怕 GPU 明明有空位。

**Continuous Batching（也叫 Iteration-level Scheduling，Orca 论文提出）**：

调度粒度改成"**每一次前向**"。每一步 decode 之后，调度器都重新问一次：
- 有序列生成了 EOS 吗？→ **立刻踢出，立刻返回给用户**（不用等别人）
- 有空位了吗？→ **立刻从等待队列拉一个新请求进来**

**类比**：static batching 是**摆渡车**——坐满 8 人发车，到站全部下车，再回来接下一批。Continuous batching 是**扶梯**——有人下就有人上，永不停机。

**收益**：Anyscale 的 benchmark 报告过相对 static batching **最高 23 倍**的吞吐提升。差距的大小**直接取决于输出长度的方差**——方差越大，static batching 浪费越多，continuous batching 赢得越多。真实线上流量的长度方差**非常大**，所以收益非常真实。

**一个关键的技术前提**：新请求要**先做 Prefill** 才能加入 decode 队伍。而 Prefill 和 Decode 的计算形态完全不同（Q1）——怎么把它们混在一个 batch 里？Orca 的方案叫 **Selective Batching**：FFN 部分可以把不同长度的序列**拉平**成一维一起算，但 Attention 部分**必须逐序列单独算**（因为每条的 KV 长度不同）。

**但这里埋着一个新问题**——一个长 Prefill 插进来，会把这一步的耗时从 7ms 拉到 100ms，**所有正在 decode 的用户都会卡一下**。这就引出了 Chunked Prefill（下一题）。"

### 📐 详细对比

#### 时序图对比

```
【Static Batching】batch=4，横轴是 decode step
        step: 1  2  3  4  5  6  7  8  9  10 11 12
  Req A (3):  ■  ■  ■  ·  ·  ·  ·  ·  ·  ·  ·  ·    ← 3 步就完了，之后 9 步空转
  Req B (12): ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■
  Req C (5):  ■  ■  ■  ■  ■  ·  ·  ·  ·  ·  ·  ·    ← 7 步空转
  Req D (4):  ■  ■  ■  ■  ·  ·  ·  ·  ·  ·  ·  ·    ← 8 步空转
              └─ Req E,F,G 在门外干等，哪怕 step 5 之后有 3 个空位 ─┘

  有效计算 = (3+12+5+4) / (4×12) = 24/48 = 50%   ← 一半的算力喂了狗

【Continuous Batching】
        step: 1  2  3  4  5  6  7  8  9  10 11 12
  Req A (3):  ■  ■  ■  ✓
  Req B (12): ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■
  Req C (5):  ■  ■  ■  ■  ■  ✓
  Req D (4):  ■  ■  ■  ■  ✓
  Req E:            ▲  ■  ■  ■  ■  ■  ✓          ← A 一走 E 立刻补上
  Req F:                  ▲  ■  ■  ■  ■  ■  ■  ■
  Req G:                     ▲  ■  ■  ■  ■  ■  ■

  ▲ = 该请求的 Prefill 发生在这一步（注意这一步会变慢 → Q6 的问题）
  有效计算 ≈ 100%，且吞吐从 4 条/12步 变成 7 条/12步
```

#### Selective Batching：怎么把不同长度的序列混一起

Orca 的关键洞察：**Transformer 里不是所有算子都要求"矩形 batch"**。

| 算子 | 能不能直接 batch 不同长度？ | 做法 |
|---|---|---|
| Linear / FFN / LayerNorm | ✅ 能 | 把 $[B, S_i, d]$ **拉平**成 $[\sum S_i, d]$，当成一个大矩阵算 |
| **Attention** | ❌ **不能** | 每条序列的 KV 长度不同、不能互相看 → **必须逐序列算**（或用 varlen kernel） |

所以叫 "**Selective**" Batching——**选择性地**只对能 batch 的算子 batch。

现代实现（FlashAttention 的 `varlen` 接口、vLLM 的 paged kernel）已经把这个做进 kernel 里了：传一个 `cu_seqlens`（累积长度数组）进去，kernel 内部自己按序列切分。所以你在 vLLM 里看到的输入是**一维拉平的 token 序列 + 一个长度索引**，而不是 `[B, S]` 的矩形张量。

**这也解释了为什么 LLM 推理引擎不需要 padding** —— 拉平之后根本没有 padding 的概念。而 HF 的 `generate()` 要 padding，这本身就是它慢的原因之一。

#### 调度器每一步在干什么

```python
# 概念示意：continuous batching 的主循环
while True:
    # ① 回收：把结束的序列踢出，释放它的 KV block
    for seq in running:
        if seq.last_token == EOS or seq.length >= seq.max_tokens:
            running.remove(seq)
            block_manager.free(seq)
            stream_back_to_user(seq)      # ← 立刻返回，不等别人

    # ② 补位：显存够就拉新请求进来（先 prefill）
    while waiting and block_manager.can_allocate(waiting[0]):
        seq = waiting.popleft()
        block_manager.allocate(seq)
        running.append(seq)               # ← 这一步会做 prefill，很重 → Q6

    # ③ 抢占：显存不够就踢掉最新的（swap 或 recompute）
    while not block_manager.can_append_all(running):
        victim = running.pop()            # 最晚进来的先被踢
        preempt(victim)                   # swap out / 丢弃重算
        waiting.appendleft(victim)

    # ④ 前向一步（prefill 的序列和 decode 的序列可能混在一起）
    logits = model.forward(running)
    sample_next_tokens(running, logits)
```

> 注意 ①②③ **每一步 decode 都会执行一遍**——这就是 "iteration-level scheduling" 这个名字的字面意思。对比 static batching：这三件事**一个 batch 只做一次**。

### 🔥 高频追问 Top 3

**Q：Continuous Batching 和 Dynamic Batching 是一回事吗？**

A：**不是，这是个高频混淆点，能分清就是加分**。

- **Dynamic Batching**（TF Serving / Triton 的经典功能）：**攒批**——在一个时间窗口内（比如 10ms）等待请求到达，凑成一批一起算。粒度还是"**批**"，一批开始了就不能变。它解决的是"请求到达时间不齐"。
- **Continuous Batching**：粒度是"**迭代**"——批的**成员在运行过程中动态进出**。它解决的是"请求**结束**时间不齐"。

**Dynamic batching 对 LLM 基本没用**，因为 LLM 的痛点不是"请求来得不齐"，而是"**生成长度不可预测**"。这是 LLM 推理和 CV 推理的根本区别——**CV 模型的计算量是固定的（一张图就是一张图），LLM 的计算量取决于要生成多少 token，事先不知道**。

**这个对比能讲出来，说明你理解 LLM serving 的特殊性在哪。**

**Q：Continuous Batching 和 PagedAttention 是什么关系？必须一起用吗？**

A：**正交，但强协同——本质上互为前提**。

- **PagedAttention 管"显存怎么放"**，Continuous Batching 管"请求怎么调度"
- **但**：Continuous Batching 要频繁地"踢出一条、加入一条"，如果显存是**连续预分配**的，这个"进进出出"会立刻制造严重的外部碎片——**新请求要 2048 连续，刚释放的空洞是 800，塞不进去**。
- 反过来：PagedAttention 省下来的显存，如果调度器还是 static batching，那也**填不满**——省了显存没地方用。

所以 **vLLM 的成功 = PagedAttention（显存层）+ Continuous Batching（调度层）的组合**，缺一个另一个都发挥不出来。

历史上 **Orca 只有 continuous batching 没有 paging**（论文里它假设预分配），**vLLM 补上了显存层**——这是 vLLM 相对 Orca 的核心增量。**能说清这段技术史是很强的信号。**

**Q：Continuous Batching 有什么代价 / 副作用？**

A：三个，都很现实：

1. **公平性问题**：显存不够时要抢占，被抢占的请求延迟会**暴涨**（要重算或 swap）。所以调度策略（FCFS vs 优先级 vs 最短作业优先）会显著影响**尾延迟 P99**。生产上经常要为不同 SLA 的流量做优先级隔离。
2. **TTFT 与 TPOT 的冲突**：新请求的 Prefill 插进来会**卡住**正在 decode 的序列 → **牺牲已有用户的 TPOT，换新用户的 TTFT**。这个矛盾就是 **Q6 Chunked Prefill 要解决的问题**。
3. **调度开销**：每步都要跑一遍调度逻辑。Decode 一步才 7ms，如果调度器（Python 写的！）花 2ms，那就有 **28% 的时间浪费在 CPU 上**。这就是所谓的 **CPU overhead / scheduler bottleneck**——vLLM V1 架构的一大动机就是把调度和执行**解耦到不同进程**、做 CPU-GPU overlap。**这个点很少有人答得出来，答出来非常加分**。

### ⚠️ 常见陷阱

1. **把它和 Dynamic Batching 混为一谈** —— 见上面追问 1，这是最常见的错。
2. **不说"输出长度不可预测"这个根因** —— 这是 LLM serving 一切复杂性的源头。不提这个，你的答案就没有灵魂。
3. **把 Continuous Batching 和 PagedAttention 说成一个东西** —— 它们是两层。
4. **只说好处不说代价** —— 主动说出"抢占伤 P99"和"Prefill 卡 Decode"，会显得你真的看过线上指标。

### 🏢 大厂偏好

- **字节 / 阿里（推理、Infra）**：必问，会追问调度策略和公平性
- **应用岗**：会问"你们 QPS 上不去怎么办"——这题是第一层答案
- **Meta**：会问 SLA 分级和尾延迟
- **所有岗位**：**"vLLM 为什么快"这个问题的完整答案 = PagedAttention + Continuous Batching**，两个都要说

### 📚 延伸阅读
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) - **OSDI'22，continuous batching 的原论文**
- [How continuous batching enables 23x throughput in LLM inference](https://www.anyscale.com/blog/continuous-batching-llm-inference) - Anyscale 的经典博客，那个 23× 的数字出处

---

## Q6：Chunked Prefill 与 Decode 卡顿 ⭐⭐⭐⭐

### 🎯 一句话标答

> Continuous Batching 下，一个长 Prompt 的 Prefill 会把某一步 iteration 从 ~7ms 拉长到几百 ms，**让所有正在 decode 的用户集体卡顿（generation stall）**；Chunked Prefill 把长 Prefill **切成固定 token 预算的小块**，每块和 decode **捎带（piggyback）在同一个 batch 里**执行，用轻微牺牲 TTFT 换来**平滑的 TPOT** 和更高的吞吐。

### 🗣️ 30 秒口语版

"这题是 Q5 的自然延续，属于进阶题，答好了很显水平。

**问题**：Continuous Batching 说"有空位就拉新请求进来"。但新请求进来的第一件事是 **Prefill**。

假设当前有 32 条序列在 decode，每步 7ms，用户看到字在稳定地往外蹦。这时来了一个 8K token 的长 prompt——
- 这一步 iteration 要算 8000 个 token 的 prefill
- 耗时从 7ms 暴涨到 **~500ms**
- 结果：**32 个正在聊天的用户，屏幕集体卡住半秒**

这个现象叫 **generation stall**。在 RAG、长文档 QA、Agent（塞了一大堆工具定义和历史）这些**长 prompt 场景**下，卡顿会非常明显——而这恰恰是我们的业务场景。

**Chunked Prefill 的解法（Sarathi 论文）**：

既然一次算 8000 个太重，那就**切开**——每次只算 512 个 token 的 prefill，分 16 次算完。

关键在于"**捎带（piggyback）**"：每一步的 batch 里 = **所有正在 decode 的序列（每条 1 token）+ 一个 prefill chunk（512 token）**。这样：
- 每步耗时从 7ms 变成 ~35ms，**平滑可控**，而不是偶尔蹦到 500ms
- 而且——**这一步的算力利用率还提高了**！因为 decode 本来是 memory-bound（GPU 算力闲着），现在**顺手把 prefill 的计算塞进那些闲置的算力里**，等于**白嫖**。

**这是最精彩的地方**：Chunked Prefill 不只是"削峰"，它还**利用了 decode 的 memory-bound 特性**——decode 阶段 GPU 算力本来就在空转（Q3 说的算术强度=B），把 compute-bound 的 prefill 混进来，**两者的瓶颈正好互补**。所以它是**又平滑、又更快**，不是纯粹的权衡。

**代价**：长 prompt 的 TTFT 会略微变差（要等 16 步而不是 1 步）。所以这是个 **TTFT ↔ TPOT 的旋钮**——`max_num_batched_tokens` 调大偏 TTFT，调小偏 TPOT。"

### 📐 详细机制

#### 问题的量化

```
【无 Chunked Prefill】32 条 decode 中，插入一个 8K prefill

  step:  ... 98    99    100        101   102 ...
  耗时:      7ms   7ms   ██520ms██  7ms   7ms
                          ↑
                    8K prefill 独占这一步
                    32 个用户全部卡 0.5 秒
  P99 ITL(inter-token latency) = 520ms   ← 用户体验灾难

【有 Chunked Prefill】chunk = 512 token

  step:  ... 98    99    100   101   ...  115   116 ...
  耗时:      7ms   7ms   34ms  34ms  ...  34ms  7ms
                          └────── 16 步，每步捎带一个 chunk ──────┘
  P99 ITL = 34ms                        ← 平滑，可接受
  长 prompt 的 TTFT: 520ms → 16×34 = 544ms （只差 5%，因为算力被更好地利用了）
```

> **注意最后一行**：TTFT 几乎没变差！因为切块之后，**每一步的 GPU 利用率更高了**（decode 的空闲算力被 prefill 填满）。所以这个"权衡"其实非常便宜。

#### 为什么 Hybrid Batch 反而更高效——回到 Roofline

| batch 组成 | 算术强度 | GPU 状态 |
|---|---|---|
| 纯 decode（32 条） | ~32 FLOP/byte | **memory-bound**，算力利用 ~20% |
| 纯 prefill（8K token） | ~8000 FLOP/byte | **compute-bound**，带宽利用低 |
| **Hybrid（32 decode + 512 prefill chunk）** | **~544 FLOP/byte** | **接近平衡点**，算力和带宽都用上了 |

**这就是 Sarathi 的核心洞察**：prefill 和 decode 的瓶颈**互补**，混在一起跑，**比分开跑都更高效**。

- 权重只需要读**一遍**（decode 和 prefill 共享同一次权重读取！）
- prefill 的 512 个 token 把闲置的 tensor core 填满
- **1 + 1 > 2**

#### 关键参数

vLLM 里的开关（V1 已默认开启）：

```python
llm = LLM(
    model="meta-llama/Llama-3-8B",
    enable_chunked_prefill=True,
    max_num_batched_tokens=512,   # ← 核心旋钮：一步最多处理多少 token
                                   #   （decode 的 token + prefill chunk 的 token）
)
```

**`max_num_batched_tokens` 怎么调**：

| 取值 | 效果 | 适合场景 |
|---|---|---|
| 小（256-512） | 每步快、TPOT 平滑；但 prefill 要切很多块，TTFT 变差 | **聊天、Agent 流式输出**——用户对卡顿敏感 |
| 大（4096+） | prefill 一两步搞定，TTFT 好；但会卡 decode | **批处理、离线任务**——没人盯着屏幕 |
| 极大 | 退化成没开 chunked prefill | — |

**默认建议**：从 512 起调，看 P99 ITL 和 TTFT 的实测曲线定。

### 🔥 高频追问 Top 3

**Q：Chunked Prefill 会不会影响生成质量？切开算和一次算，结果一样吗？**

A：**数学上完全等价，结果 bit-level 上可能有微小差异**。

- **为什么等价**：Prefill 本质是"算出每个位置的 KV"。因为 causal mask，第 $i$ 个位置的 KV **只依赖 $\le i$ 的位置**。所以先算 0-511，再算 512-1023（这时 512-1023 能看到已经算好的 0-511 的 KV），和一次性算 0-1023 —— **完全一样的依赖关系，完全一样的结果**。
- **为什么可能有微小差异**：浮点加法不满足结合律，分块的 online softmax 归约顺序不同，可能有 1e-6 量级的数值差异。**不影响质量，但会让"结果不可 bit-wise 复现"** —— 做 A/B 测试或回归测试时要注意这点。

**关键点：Chunked Prefill 是"如何调度计算"的优化，不是"算什么"的改变——所以它是无损的。** 这和第 13 章的量化、Q2 追问里的 KV 驱逐**性质完全不同**（那些是有损的）。**能主动区分"无损优化"和"有损优化"，是很好的工程素养信号。**

**Q：Chunked Prefill 和 PD 分离（Disaggregation）是竞争关系吗？该选哪个？**

A：**是同一个问题的两种解法，选择取决于规模**。

问题都是"prefill 和 decode 互相干扰"：
- **Chunked Prefill**：**融合**——既然干扰，那就混得更彻底，让两者互补（1+1>2）
- **PD 分离**：**隔离**——既然干扰，那就物理分开，各自优化到极致

| | Chunked Prefill | PD 分离 |
|---|---|---|
| 部署复杂度 | **低**（一个开关） | **高**（两个集群 + KV 传输） |
| 适合规模 | 单机 / 小集群 | 大规模集群 |
| 硬件 | 同构 | **可异构**（P 用算力卡，D 用带宽卡） |
| 额外成本 | 无 | **KV Cache 跨机传输**（网络带宽） |
| 极致优化空间 | 中（两者仍互相约束） | **高**（P 和 D 可独立调 TP/batch/卡型） |

**实践**：中小规模（几台机器）用 Chunked Prefill 就够；超大规模（Mooncake、DeepSeek 的部署）上 PD 分离。**两者也可以叠加**——PD 分离后，P 集群内部仍可以用 chunked prefill 来平滑。

**Q：为什么 decode 的序列不需要"chunk"？**

A：因为 **decode 每条序列每步就 1 个 token，本来就是最小粒度了，没得切**。

Chunked Prefill 里被切的**只有 prefill 部分**。真正被限制的是"**一步 batch 里的总 token 预算**"——所以 `max_num_batched_tokens` 是这么分配的：

```
一步的 token 预算 = 512
  ├── decode 的序列：32 条 × 1 token = 32 token   （优先满足，不能拖 TPOT）
  └── prefill chunk：512 - 32 = 480 token          （用剩下的预算）
```

**vLLM 的调度策略是 decode 优先** —— 先把所有 running 的 decode 序列排进去，**剩余预算才给 prefill**。这保证了 decode 永远不会因为 prefill 而挨饿。**这个优先级细节能说出来，是"读过源码"级别的信号。**

### ⚠️ 常见陷阱

1. **说 Chunked Prefill 是为了让 prefill 更快** —— **反了**。它主要是为了**不让 prefill 拖累 decode**，prefill 自己的 TTFT 是略微变差的（虽然差得不多）。
2. **忽略"互补"这个精髓** —— 只说"削峰"是答了一半。答出"decode 是 memory-bound、prefill 是 compute-bound、混在一起 GPU 利用率反而更高"才是完整答案。
3. **以为它有损** —— 它是**完全无损**的调度优化。
4. **不知道它现在是默认开启的** —— vLLM V1 默认开，说"这是个可选优化"会显得没跟上版本。

### 🏢 大厂偏好

- **字节 / 阿里推理引擎岗**：会问，属于"区分度题"——答得出来说明真在做推理
- **应用岗**：不常直接问，但如果你在项目里提到"优化了流式输出的卡顿"，会被追问到这里
- **Agent 岗**：**高度相关**——Agent 的 prompt 特别长（工具定义 + 历史 + RAG 上下文），generation stall 是真实痛点

### 📚 延伸阅读
- [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369) - 原论文
- [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310) - OSDI'24，"stall-free batching" 的完整系统

---

## Q7：Prefix Caching / RadixAttention ⭐⭐⭐⭐

### 🎯 一句话标答

> 不同请求经常共享**相同的前缀**（system prompt、few-shot 示例、多轮对话历史、RAG 检索到的同一篇文档），Prefix Caching 把这些前缀的 KV Cache **跨请求复用**——命中的部分**完全跳过 Prefill 计算**，TTFT 可以降一个数量级；vLLM 用 **hash + block 引用计数**实现，SGLang 用 **Radix Tree（RadixAttention）** 做更细粒度的前缀树匹配和 LRU 驱逐。

### 🗣️ 30 秒口语版

"这题在 **Agent 和 RAG 场景下的价值最大**，属于'应用岗必须知道'的优化。

**动机**：看看真实业务里的 prompt 长什么样——

```
[System Prompt: 2000 token 的角色设定 + 安全规则]   ← 每个请求都一样
[Few-shot: 3 个示例，1500 token]                    ← 每个请求都一样
[Tool definitions: 20 个工具的 schema，3000 token]   ← 每个 Agent 请求都一样
[对话历史: 5000 token]                              ← 同一个会话内每一轮都一样
[用户的新问题: 20 token]                            ← 只有这里不一样
```

**6500 token 是完全相同的，只有最后 20 个 token 变了**。但每次请求都要把这 6500 个 token 重新 Prefill 一遍——**纯粹的浪费**。

**Prefix Caching**：把这些前缀的 KV Cache 留在显存里，下次命中直接复用。

**收益**：
- **TTFT 暴降**：只需 prefill 那 20 个新 token，而不是 6520 个。实测能降 **5-10 倍**
- **吞吐提升**：省下来的算力给别人用

**为什么能这么干？** 还是 causal mask——**前缀的 KV 只依赖前缀本身，跟后面接什么内容完全无关**。所以"[system][few-shot]" 这段的 KV，无论后面接什么问题，**永远是同一个值**。

**实现的关键前提是 PagedAttention**（Q4）——因为 KV 已经切成 block 了，共享一个 block 只需要**加引用计数**，天然支持。**如果 KV 是连续预分配的，跨请求共享根本无从谈起**。这就是我说 PagedAttention 的"共享"那半边是 Prefix Caching 基础的原因。

**两种实现**：
- **vLLM 的 Automatic Prefix Caching**：对每个 block **算 hash**（内容 + 它前面所有 block 的 hash，即前缀哈希），hash 相同就复用物理块
- **SGLang 的 RadixAttention**：维护一棵 **Radix Tree（压缩前缀树）**，节点存 KV block，能做**更细粒度的部分匹配**，配 LRU 驱逐

**Agent 场景的一个关键实践**：**prompt 的顺序决定了缓存命中率**。把不变的东西（system、tools、few-shot）放前面，变化的东西（用户输入、时间戳）放后面。**只要前面插入一个时间戳，后面全部前缀失效**——这是个真实的线上坑。"

### 📐 详细机制

#### 命中率的结构性来源

| 场景 | 共享前缀 | 典型命中率 |
|---|---|---|
| **多轮对话** | 前 n-1 轮的全部历史 | **极高**（对话越长越高） |
| **Agent（ReAct 循环）** | system + tool schema + 之前所有的 thought/action/observation | **极高**（每一轮循环只在末尾追加） |
| **Few-shot 批量任务** | system + 示例 | 高 |
| **RAG** | system；**若同一文档被多个问题命中**，则文档也共享 | 中-高 |
| **无状态单轮 QA** | 只有 system prompt | 低-中 |

> 🚨 **Agent 场景要特别注意**：ReAct 是个循环，每轮的 prompt = 上一轮的 prompt + 新的 observation。**这意味着第 k 轮和第 k+1 轮共享几乎全部前缀**。没开 prefix caching 的 Agent，等于每轮都把整个历史重新 prefill 一遍——**一个 10 轮的 Agent 任务会浪费掉 ~10 倍的 prefill 算力**。**这是 Agent 岗面试里能讲出的最实用的优化之一。**

#### vLLM 的 Hash-based APC

核心是**前缀哈希**：block 的 hash 必须包含它**前面所有内容**，否则会错误复用。

```python
# 概念示意
def compute_block_hash(prev_block_hash, token_ids_in_this_block):
    """
    关键：hash 要串起前缀！
    否则 "A的第2块" 和 "B的第2块" 内容相同就会被错误复用，
    但它们的前缀不同 → KV 值其实不同 → 结果错误
    """
    return hash((prev_block_hash, tuple(token_ids_in_this_block)))


class PrefixCache:
    def __init__(self):
        self.hash_to_block = {}   # block_hash -> physical_block_id

    def try_reuse(self, token_ids):
        """返回能复用的 block 数"""
        block_table, prev_hash = [], None
        for i in range(0, len(token_ids), BLOCK_SIZE):
            chunk = token_ids[i:i+BLOCK_SIZE]
            if len(chunk) < BLOCK_SIZE:
                break                          # ← 不满的块不缓存（内容还会变）
            h = compute_block_hash(prev_hash, chunk)
            if h in self.hash_to_block:
                blk = self.hash_to_block[h]
                self.blocks[blk].ref_count += 1
                block_table.append(blk)        # ✅ 命中：跳过这 16 个 token 的 prefill
                prev_hash = h
            else:
                break                          # ❌ 一旦不匹配，后面全部作废
        return block_table
```

> **两个关键设计**：
> 1. **`prev_block_hash` 必须参与 hash** —— 否则会出现"内容相同但前缀不同"的错误复用，**结果会错**。这是这套机制的正确性根基。
> 2. **一旦某块不匹配，后面全部作废（`break`）** —— 这是"前缀"的字面含义。所以**变化的内容必须放最后**。

**开启方式**：
```python
llm = LLM(model="...", enable_prefix_caching=True)   # vLLM V1 已默认开启
```

#### SGLang 的 RadixAttention

用 **Radix Tree（压缩前缀树）**替代 hash 表：

```
                    root
                     │
        ┌────────────┴────────────┐
   "You are a helpful       "You are a coding
    assistant..."            assistant..."
        │                          │
   ┌────┴────┐              ┌──────┴──────┐
"What is  "Explain      "Write a      "Debug
 the..."   the..."       function..."  this..."

每个节点持有一段 token 序列 + 对应的 KV block
匹配 = 从 root 往下走，走多远就复用多少
```

**相对 hash 方案的优势**：
- **细粒度匹配**：不受 block 边界对齐限制，能匹配到任意 token 位置
- **天然支持树形结构**：Agent 的分支探索、多候选采样、ToT（Tree-of-Thought）—— 这些本来就是树，用树来存最自然
- **LRU 驱逐更聪明**：可以从**叶子**开始驱逐（保留公共前缀，先淘汰特化的分支）——因为公共前缀被更多请求依赖

代价：树的维护逻辑更复杂。

### 🔥 高频追问 Top 3

**Q：Prefix Caching 有什么风险？为什么不是所有场景都开？**

A：三个真实的顾虑：

1. **安全 / 隐私（最重要）**：**跨用户共享 KV Cache 意味着跨用户共享数据**。虽然 KV 本身不能直接"读出"原文，但存在**时序侧信道攻击**——攻击者构造 prompt，通过观测 **TTFT 是否异常快**来判断"这个前缀是否被别人用过"，从而**探测其他用户的输入**。真实的公有云部署要考虑**按租户隔离 cache**。**这个点非常加分**——说明你有安全意识。
2. **显存占用**：cache 占着的显存就不能给 KV Cache 用了。命中率低的场景（每个用户 prompt 都不同），纯亏。所以需要 **LRU + 显存预算控制**。
3. **一致性**：如果同一个前缀在不同精度/不同 batch 下算出来的 KV 有微小浮点差异，复用会引入不可复现性。实践上影响可忽略，但**回归测试要留意**。

**Q：Prefix Caching 能优化 Decode 吗？**

A：**不能，只优化 Prefill**。这是个高频陷阱题。

Prefix Caching 省的是"**算 KV 的过程**"（prefill）。但 decode 时，**这些 KV 还是要被完整地读一遍**——attention 要看全部历史。所以：
- **TTFT**：显著改善 ✅
- **TPOT**：**完全没有改善** ❌（读的字节数一模一样）
- **显存**：改善（多请求共享一份）✅ → 间接允许更大 batch → 间接改善吞吐

**"Prefix Caching 救 TTFT 不救 TPOT"** —— 和 Q6 的 "Chunked Prefill 救 TPOT 略伤 TTFT" 正好互补。**把这两句放在一起说，会显得你的知识是成体系的，不是零散的名词。**

**Q：怎么设计 prompt 来最大化命中率？**

A：**这是应用岗最实用的一题**，核心原则是"**把稳定的放前面，把易变的放后面**"。

| ❌ 反模式 | ✅ 正确做法 |
|---|---|
| `[当前时间: 2026-07-16 14:23:07]` 放开头 | 时间戳放**最后**，或干脆去掉（真需要就用工具查） |
| `[用户ID: u_12345]` 放开头 | 用户信息放最后，或用统一占位 |
| 工具列表按**动态顺序**（比如按相关性排序）拼接 | 工具列表**固定顺序**——顺序一变，token 就变，前缀就废了 |
| 每次重新排列 few-shot 示例 | 示例顺序固定 |
| JSON 序列化时 key 顺序不固定 | **`sort_keys=True`** —— 这是个真实踩过的坑 |

**推荐的 prompt 结构**：
```
┌─────────────────────────────┐
│ System Prompt（永不变）        │  ← 全局共享
├─────────────────────────────┤
│ Tool Definitions（固定顺序）    │  ← 同一 Agent 的所有请求共享
├─────────────────────────────┤
│ Few-shot Examples（固定）      │  ← 同一任务共享
├─────────────────────────────┤
│ 对话历史（只追加，不修改）        │  ← 同一会话共享 ★ Agent 的最大收益点
├─────────────────────────────┤
│ 用户新输入 / 新 Observation     │  ← 只有这里 miss
└─────────────────────────────┘
```

> **一个特别隐蔽的坑**：**对话历史的"修剪"会毁掉缓存**。很多实现为了控制长度，会滑动窗口地丢弃最早几轮——但一丢，**前缀就变了，整个 cache 全废**。所以长历史的处理要么**只在末尾追加**，要么**在固定的检查点做压缩**（而不是每轮都滑一格）。这个 tradeoff 在第 18 章的 Context Engineering 会详谈。

### ⚠️ 常见陷阱

1. **忘了 hash 必须包含前缀** —— 只 hash block 内容会导致**错误复用、结果出错**。这是正确性问题，不是性能问题。
2. **以为能优化 TPOT** —— 只优化 TTFT。
3. **不提安全风险** —— 跨租户的 KV 共享是有侧信道的，能提到会显著加分。
4. **不知道"变化的东西放最后"这条铁律** —— 这是应用侧最实用的一条，不知道说明没在生产上用过。

### 🏢 大厂偏好

- **应用岗 / Agent 岗**：**高频**，且会问"你怎么优化的 Agent 延迟"——prompt 结构那题就是最好的答案
- **字节 / 阿里**：会问 vLLM hash 方案 vs SGLang radix tree 的取舍
- **有公有云业务的团队（阿里云、火山引擎）**：**会问跨租户的安全隔离**

### 📚 延伸阅读
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) - RadixAttention 原论文
- [vLLM Automatic Prefix Caching 设计文档](https://docs.vllm.ai/en/latest/design/v1/prefix_caching.html)
- [Prompt Cache](https://arxiv.org/abs/2311.04934) - 更激进的方案：连**非前缀**的模块化片段也想复用

---

## 📝 本章小结

| 关键点 | 你必须能脱口而出 |
|---|---|
| **两阶段** | Prefill = compute-bound = TTFT；Decode = memory-bound = TPOT |
| **KV Cache 公式** | $2 \times B \times S \times L \times n_{kv} \times d_h \times$ dtype，**注意是 $n_{kv}$ 不是 $n_q$** |
| **锚点数字** | LLaMA-2 7B = **512 KB/token**；70B(GQA-8) = **320 KB/token**（比 7B 还小！） |
| **算术强度** | Decode ≈ **batch size** FLOP/byte；A100 平衡点 ≈ **150** |
| **速度上界** | TPOT_min = 模型大小 ÷ HBM 带宽 → 7B fp16 单卡 **≈ 140 tok/s** |
| **三条优化路线** | 减小模型（量化）/ 提高 batch（调度）/ 一次多出几个 token（投机解码） |
| **PagedAttention** | 治**碎片**（内部/外部/无法共享），浪费 60-80% → <4%，kernel 慢 20% 但吞吐 2-4× |
| **Continuous Batching** | 调度粒度从"批"降到"**迭代**"，根因是**输出长度不可预测** |
| **两者的关系** | PagedAttention = 显存层，Continuous Batching = 调度层，**正交但互为前提** |
| **Chunked Prefill** | 救 TPOT、略伤 TTFT；**精髓是 prefill/decode 瓶颈互补，1+1>2**；**无损** |
| **Prefix Caching** | 救 TTFT、**不救 TPOT**；铁律：**变化的东西放最后** |
| **贯穿全章的因果链** | KV Cache 大 → 并发受限 → 吞吐受限 → 成本高。GQA/MLA 治"**大**"，PagedAttention 治"**浪费**"，Continuous Batching 治"**空转**" |

## ✅ 自测题

1. 面试官给你 A100-80G + LLaMA-3 70B（80 层，GQA-8，head_dim 128），问："fp16 部署，要支持 32K 上下文，最多几并发？要几张卡？" —— **现场算，30 秒内给出数字**。
2. 用 Roofline 解释为什么 batch=1 时 7B 模型在 A100 上跑不过 ~140 tok/s。如果换 H100 呢？如果量化到 int4 呢？
3. 为什么说"batching 能救权重的算术强度，救不了 attention 的"？这个结论对长上下文场景意味着什么？
4. PagedAttention 让 attention kernel 变慢了 20%，为什么整体吞吐反而涨 2-4 倍？用 Q3 的结论解释。
5. 你的 Agent 线上反馈"每次工具调用后要等好久才出下一个字"。给出**三个**可能的原因和对应的排查手段。
6. 设计一道题：给你一个 RAG + Agent 的线上服务，QPS 上不去、P99 延迟高。把本章 7 个技术按"**你会先上哪个**"排序，并说明理由。

---

> **下一章预告**：第 13 章｜量化专题——PTQ vs QAT、**GPTQ**（从 OBQ 到 Hessian 近似）、**AWQ**（为什么"激活感知"比"权重大小"更对）、SmoothQuant 的迁移技巧，以及 W4A16 / W8A8 / FP8 的选型。本章说了"减小模型大小是三条优化路线之一"，第 13 章就是那条路线的全部内容。**GPTQ 的推导是字节和阿里的高频手撕题**。
