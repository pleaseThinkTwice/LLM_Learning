# 第 11 章｜GRPO 与推理模型训练

> **本章定位**：GRPO 是 DeepSeek-R1 的核心算法，是 2024-2026 alignment 方向最重要的进展。本章覆盖：**从 PPO 到 GRPO 的演化**、**完整公式推导**、**过程奖励 PRM**、**DeepSeek-R1 训练全流程**、**推理能力涌现**。这是 DeepSeek 系面试的标志性深度题。

> **配套章节**：RLHF/PPO → 第 9 章 / DPO → 第 10 章 / DeepSeek 专题 → 第 6 部分前沿热点

---

## 📌 本章导航

| 序号 | 题目 | 难度 | 频率（字节/DeepSeek/Meta/OpenAI） | 类型 |
|---|---|---|---|---|
| Q1 | GRPO 是什么？为什么 R1 用它 | ⭐⭐⭐⭐⭐ | 6 / 9 / 4 / 4 | 概念 |
| Q2 | **GRPO 完整公式推导** | ⭐⭐⭐⭐⭐ | **6 / 10 / 5 / 4** | 杀手题 |
| Q3 | **为什么去掉 critic 也能 work** | ⭐⭐⭐⭐⭐ | 5 / 8 / 4 / 4 | 深度 |
| Q4 | PRM vs ORM：过程奖励 vs 结果奖励 | ⭐⭐⭐⭐ | 5 / 7 / 5 / 5 | 概念 |
| Q5 | **DeepSeek-R1 训练全流程** | ⭐⭐⭐⭐⭐ | **7 / 10 / 4 / 5** | 必问 |
| Q6 | R1-Zero 与 R1 的区别 | ⭐⭐⭐⭐ | 5 / 8 / 3 / 4 | 深度 |
| Q7 | 推理能力涌现：Aha Moment 与长 CoT | ⭐⭐⭐⭐ | 4 / 7 / 3 / 5 | 前沿 |
| Q8 | GRPO 的局限与替代方案 | ⭐⭐⭐ | 3 / 5 / 3 / 3 | 工程 |

---

## Q1：GRPO 是什么？为什么 R1 用它 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> GRPO（Group Relative Policy Optimization）= **去掉 critic 的 PPO 变体**——同一 prompt 采样一组响应，**用组内的相对 reward 作为 baseline**，省掉了 value 模型。**显存省、训练稳、推理任务上效果优秀**——是 DeepSeek-R1 的核心算法。

### 🗣️ 30 秒口语版

"GRPO 是 DeepSeek 在 DeepSeekMath 论文（2024）提出，在 DeepSeek-R1 上大放异彩的强化学习算法。

**核心思想**：传统 PPO 需要一个 **critic 模型** 估计 V(s)，用于计算 advantage A = Q - V。这增加了显存和工程复杂度。

GRPO 的 insight——**对同一个 prompt 采样 G 个响应**，用**组内 reward 的均值**作为 baseline，**标准差归一化**作为 advantage。完全省掉 critic。

具体公式（简化版）：
- 对 prompt q 采样 G 个响应 o_1, ..., o_G
- 算每个的 reward r_i（来自 RM 或规则）
- Advantage: A_i = (r_i - mean(r)) / std(r)
- 用 PPO 的 clipped objective 优化 policy

**优势**：
- **没有 critic 模型** → 省 30%+ 显存
- **训练更稳** → 没有 critic 偏差累积
- **推理任务友好** → 同一 prompt 的多采样适合数学/代码这种"答案对就好"的场景
- **代码简单** → 比 PPO 实现少一个模型一个 loss

**为什么 R1 用它**：DeepSeek-R1 训练目标是数学、代码、推理——这些任务有**可验证的 outcome reward**（对就 1，错就 0）。GRPO 的 group sampling 在这种场景下天然适配——多次尝试，对比效果，朝着对的方向优化。

**当前格局**：GRPO 已成为推理模型训练的事实标准。Kimi-K0、Qwen-QwQ、OpenAI o1 据传都用了类似方法。"

### 🔍 GRPO vs PPO 对比

| 维度 | PPO (RLHF) | GRPO |
|---|---|---|
| Critic 模型 | 需要 | **不需要** |
| Baseline | V(s) 网络估计 | **组内均值** |
| Advantage 计算 | A = Q - V | **A = (r - mean) / std** |
| 模型数量 | 4 | **3**（少一个 critic）|
| 显存压力 | 高 | 中 |
| 适合任务 | 通用对话 | **推理、数学、代码** |
| 训练稳定性 | 中 | **高** |
| 代表模型 | ChatGPT | **DeepSeek-R1** |

### 🔥 高频追问 Top 3

**Q：GRPO 适合通用对话吗？**

A：**理论可以，实际优势不明显**。

GRPO 的核心优势——**同一 prompt 多采样 + 组内对比**——在以下场景闪光：
- **数学/代码**：reward 是 0/1 可验证的
- **推理任务**：能区分对错
- **有客观标准**的任务

但通用对话——
- Reward 是连续值（人类偏好）
- 同一 prompt 的多次响应可能"都行但风格不同"
- 没有明显的对错信号

所以 GRPO 用在通用对话上没特别优势，反而 DPO 更简单。**GRPO 是为推理任务量身定制的**。

**Q：GRPO 一定要 outcome reward 吗？过程 reward 行不行？**

A：**两者都可以**。但实际上——

- **Outcome reward**（最终对错）：最常见，规则化简单
- **Process reward**（每步对错，来自 PRM）：更精细但更贵

DeepSeek-R1 主要用 outcome reward 配合 GRPO，**rule-based**：
- 数学题：检查最终答案是否匹配
- 代码：跑测试用例
- 格式：检查是否符合 `<think>...</think><answer>...</answer>` 模式

**简单的 rule-based outcome reward + GRPO** 居然就能训出强大的推理模型——这是 R1 论文的惊人发现。

**Q：GRPO 的"G"（组大小）怎么定？**

A：典型 **G = 16 到 64**。

- **G 太小**（< 8）：baseline 估计方差大，训练不稳
- **G 适中**（16-32）：平衡点
- **G 太大**（> 64）：每步计算量大，性价比下降

DeepSeek 在 R1 论文中用了 G = 64。这意味着每个 prompt 要生成 64 个完整响应才能算一次 GRPO 更新——**rollout 成本巨大**。

### ⚠️ 常见陷阱

1. 不要说 GRPO 是"PPO 的升级"——它是 PPO 的**简化变体**，去掉了 critic
2. GRPO 适合**有明确 reward 信号**的任务，不是万能
3. G 是关键超参——和 PPO 的 batch 不同，要单独算

### 🏢 大厂偏好

- **DeepSeek**：必问，是公司的"暗号题"
- **国内大模型团队**：推理模型方向跟进 GRPO
- **OpenAI / Anthropic**：会问"o1 怎么训"，可能涉及类似思想

### 📚 延伸阅读
- [DeepSeekMath 论文](https://arxiv.org/abs/2402.03300) - GRPO 原始提出
- [DeepSeek-R1 论文](https://arxiv.org/abs/2501.12948) - GRPO 大规模应用

---

## Q2：GRPO 完整公式推导 ⭐⭐⭐⭐⭐（杀手题）

### 🎯 一句话标答

> GRPO 目标函数 = **PPO 的 clipped objective + 组内相对 advantage + KL 正则**，关键创新是 `A_i = (r_i - mean(r)) / std(r)` 代替了 PPO 的 GAE。整个 loss 没有 value function。

### 🗣️ 30 秒口语版

"GRPO 的完整公式可以分四步建构——

**Step 1：从 PPO 出发**

PPO 的目标（per token）：
`L^PPO = E[min(r·A, clip(r, 1-ε, 1+ε)·A) - β·KL(π||π_ref)]`

GRPO 保留这个核心结构，但**改变 A 的计算方式**。

**Step 2：Group Sampling**

对每个 prompt q，用 π_old 采样 G 个响应：
`{o_1, o_2, ..., o_G} ~ π_old(·|q)`

每个 o_i 都是完整的序列。

**Step 3：Group Relative Advantage**

对每个响应 o_i 算 reward r_i（可以来自 RM、规则、或验证器）。

**Advantage 不再依赖 V(s)，而是组内归一化**：
`A_i = (r_i - mean(r_1, ..., r_G)) / std(r_1, ..., r_G)`

直觉：好于平均的响应 A > 0（被推），差于平均的 A < 0（被抑）。

**Step 4：GRPO Loss**

把组内 advantage 和 PPO 的 clipped objective 结合：

`L^GRPO = E_q E_{o_i~π_old} [1/G · Σ_i (1/|o_i|) · Σ_t L_clip(i,t)]`

其中：
- L_clip(i,t) 是位置 (i, t) 的 clipped surrogate
- 1/|o_i| 是按响应长度归一化
- 1/G 是组内平均
- KL 项加在 reward 里或直接加到 loss 里

**关键细节**：advantage A_i 对响应 o_i 的所有 token 是**共享的**——同一个响应内每个 token 用同一个 A 值。"

### 📐 完整数学公式

#### GRPO 目标函数

对每个 prompt q，从 $\pi_{\theta_{old}}$ 采样 G 个响应 $\{o_1, ..., o_G\}$，则 GRPO 目标：

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|q)} \left[\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{\min[r_{i,t}(\theta) \cdot A_i, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \cdot A_i] - \beta \cdot \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref})\right\}\right]$$

其中：

**Ratio**（per-token，per-response）：

$$r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})}$$

**Group Relative Advantage**（per-response，所有 token 共享）：

$$A_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}$$

其中 $r_i$ 是响应 $o_i$ 的 reward（标量）。

**KL 项**（per-token，approximated）：

$$\mathbb{D}_{KL}(\pi_\theta \| \pi_{ref}) = \frac{\pi_{ref}(o_{i,t} | q, o_{i,<t})}{\pi_\theta(o_{i,t} | q, o_{i,<t})} - \log \frac{\pi_{ref}(o_{i,t} | q, o_{i,<t})}{\pi_\theta(o_{i,t} | q, o_{i,<t})} - 1$$

注意 GRPO 用的是 **unbiased estimator of KL**（k3 estimator），不是简单的 log-ratio。

#### 与 PPO 对比

PPO 的目标（参考）：

$$\mathcal{J}_{PPO}(\theta) = \mathbb{E}_t\left[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)\right]$$

with $A_t = $ GAE estimate from value network $V_\phi$.

**核心差异**：
- PPO 的 $A_t$ 是 per-token，依赖 critic
- GRPO 的 $A_i$ 是 per-response，依赖 group baseline

### 💻 GRPO 训练伪代码

```python
def grpo_step(prompts, policy, ref_policy, reward_fn, G=64):
    losses = []
    
    for q in prompts:
        # 1. Group sampling
        responses = []
        old_log_probs_list = []
        for _ in range(G):
            with torch.no_grad():
                o = policy.generate(q)
                old_log_probs = policy.log_prob(q, o)  # per-token
            responses.append(o)
            old_log_probs_list.append(old_log_probs)
        
        # 2. Compute rewards (rule-based or RM)
        rewards = torch.tensor([reward_fn(q, o) for o in responses])
        
        # 3. Group relative advantage (per-response, shared by all tokens)
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r  # shape: (G,)
        
        # 4. PPO-style update for each response
        for i, o in enumerate(responses):
            A_i = advantages[i]
            
            # New log probs (require grad)
            new_log_probs = policy.log_prob(q, o)
            ratio = torch.exp(new_log_probs - old_log_probs_list[i])
            
            # Clipped surrogate
            surr1 = ratio * A_i
            surr2 = torch.clamp(ratio, 1-eps, 1+eps) * A_i
            policy_loss = -torch.min(surr1, surr2)
            
            # KL term (unbiased estimator)
            ref_log_probs = ref_policy.log_prob(q, o)
            log_ratio = ref_log_probs - new_log_probs
            kl = torch.exp(log_ratio) - log_ratio - 1
            
            # Per-token loss, normalize by length
            loss_i = (policy_loss + beta * kl).mean()
            losses.append(loss_i)
    
    # Average over group and prompts
    total_loss = torch.stack(losses).mean()
    total_loss.backward()
    optimizer.step()
```

### 💡 关键设计点解读

**1. Advantage 是 per-response，不是 per-token**

PPO 的 GAE 给每个 token 一个 advantage——因为 critic 估计每个状态的 V(s)。

GRPO 的 group baseline 是**整个响应的标量 reward**，所以 advantage 也是 per-response 的——**这个 response 的所有 token 共享同一个 A_i**。

**为什么这样可以？** 因为推理任务的 reward 本来就是 outcome-based（最终对错），各个中间 token 没有独立的 reward 信号。给整个响应一个统一的 advantage 是合理的。

**2. 长度归一化 1/|o_i|**

不同响应长度不同——长响应有更多 token，loss 累加会更大。除以 |o_i| 让 **每个响应在 loss 中权重相等**，不被长度主导。

**3. KL 在 loss 而非 reward**

PPO 中 KL 加在 per-token reward 里。GRPO **直接加在 loss 里**——理由是：
- GRPO 的 reward 是 outcome-based（最后一个 token 给出），中间 token 没有 reward 信号
- KL 加在 reward 里会和 outcome reward 混在一起，不清晰
- 加在 loss 里直接、清晰

### 🔥 高频追问 Top 3

**Q：为什么用 (r - mean) / std 而不是 (r - mean)？**

A：**标准化让 advantage 尺度无关**。

- 只用 (r - mean)：advantage 的尺度取决于 reward 本身。比如 reward 都在 [0, 1] vs [0, 100] 会让有效学习率差 100×
- (r - mean) / std：advantage 总是在合理范围（标准差 1），不依赖 reward 尺度

这种"白化"让 GRPO **对 reward 函数的尺度更鲁棒**——你可以随便定义 reward（0/1 还是 0/100），GRPO 都能稳定训练。

**Q：std 为什么不会让训练变差？比如 reward 都相同时 std=0。**

A：**实现中加了 epsilon**：`std + 1e-8`。

但更重要的是——**如果一个 group 的 reward 都相同，说明这个 prompt 上 policy 没分化，确实不需要更新**。advantages 全是 0 也合理。

实际训练中：
- 训练初期 group 内 reward 差异大（policy 还在学）→ advantage 显著
- 训练后期某些 prompt 的 group reward 趋同（policy 学会了）→ advantage 小

这个动态正是 GRPO 想要的——**已学好的 prompt 自动降低权重**。

**Q：GRPO 的 KL 用了 k3 estimator，为什么？**

A：k3 estimator：`KL ≈ ratio - log(ratio) - 1`，其中 ratio = π_ref/π。

它比简单的 `log(π/π_ref)` 好的原因——
- 是 KL 的**无偏估计**（unbiased）
- **始终非负**（log-ratio 形式有时负）
- 在小 KL 时**方差更小**

数学上：
- 标准 KL: `KL(π || π_ref) = E_{a~π}[log(π/π_ref)]`
- 但实际我们对一个采样 a 算 log(π/π_ref) 是个有偏估计（无方差控制）
- k3：`E_{a~π}[ratio_ref/π - log(ratio_ref/π) - 1]` 是 KL 的无偏估计

DeepSeek 用 k3 是为了训练稳定性。

### ⚠️ 常见陷阱

1. **要懂 advantage 是 per-response，不是 per-token**
2. **要懂标准化的重要性**——尺度无关
3. **KL 在 loss 不在 reward**——和 PPO 不同
4. **G（group size）是关键超参**——影响计算量

### 🏢 大厂偏好

- **DeepSeek**：必问，要求完整推导
- **字节大模型团队**：高频追问
- **国内推理模型团队**：经典题

### 📚 延伸阅读
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300) - §4.1 GRPO 详细
- [DeepSeek-R1 Paper](https://arxiv.org/abs/2501.12948) - GRPO 在 R1 的应用

---

## Q3：为什么去掉 critic 也能 work ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 因为 **同一 prompt 的 group sampling 提供了天然的 baseline**——组内均值就是 V(s) 的无偏估计，且**方差更小**（同一上下文消除了 prompt 间差异），所以不需要单独训 critic 也能算出可靠的 advantage。

### 🗣️ 30 秒口语版

"这是个深度题，能答好显示你真懂强化学习的 baseline 原理。

**回顾 actor-critic**：

PPO 用 critic V_φ 学习 V(s)，作为 baseline 计算 advantage：
`A_t = Q(s_t, a_t) - V_φ(s_t)`

**为什么要 baseline**？方差减少。Policy gradient 的方差大，减去一个不依赖动作的 baseline 不改变梯度期望但显著降方差。

**GRPO 的洞察**：

对于同一个 prompt q，采样 G 个响应。每个响应对应一个 reward r_i。**这 G 个 reward 共享同一个上下文 q**——它们的差异完全来自动作（不同响应），不来自状态（prompt 是同一个）。

所以：
`mean(r_1, ..., r_G) ≈ E[r | q] = V(q)`

**组内均值就是 V(q) 的蒙特卡洛估计**！而且：

**优势 1：无偏**

只要 G 足够大，mean 是 V 的无偏估计。

**优势 2：方差小**

V_φ 学习时受所有 (q, a) 影响——不同 prompt 之间互相干扰。组内均值**只基于这个 prompt** 的样本，**自然解耦**。

**优势 3：免训练**

V_φ 是个需要训练的模型，会犯错。组内均值是直接计算，没有训练偏差。

**优势 4：省显存**

V 模型和 policy 同规模——去掉省 30%+ 显存。

**为什么之前没人这么做**？因为传统 RL 任务里——
- 不同 trajectory 起始状态不同（如 Atari 游戏的随机初始）
- 难以保证同一状态多次采样

LLM RLHF 的特殊性——**同一 prompt 可以无限次重采样**，恰好满足 group sampling 的前提。这是 LLM-RL 独有的优势。

GRPO 把这个独特优势利用起来，绕过了 critic。"

### 📐 数学分析

#### Policy Gradient 中 baseline 的作用

策略梯度：

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau \left[\sum_t \nabla_\theta \log \pi(a_t|s_t) \cdot Q(s_t, a_t)\right]$$

**关键引理**：减去任意不依赖 $a_t$ 的 baseline $b(s_t)$ 不改变期望：

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau \left[\sum_t \nabla_\theta \log \pi(a_t|s_t) \cdot (Q(s_t, a_t) - b(s_t))\right]$$

证明思路：
$$\mathbb{E}_{a \sim \pi}[\nabla \log \pi(a|s) \cdot b(s)] = b(s) \cdot \mathbb{E}_{a \sim \pi}[\nabla \log \pi(a|s)] = b(s) \cdot 0 = 0$$

（因为 $\sum_a \pi(a|s) = 1$，导数为 0。）

#### 最优 baseline

最优 baseline 是**让 advantage 方差最小**的那个，理论上：

$$b^*(s) = \frac{\mathbb{E}_a[Q(s,a) \cdot (\nabla \log \pi(a|s))^2]}{\mathbb{E}_a[(\nabla \log \pi(a|s))^2]}$$

实践中近似为 $V(s) = \mathbb{E}_{a \sim \pi}[Q(s, a)]$，因为它简单且方差减少效果接近最优。

#### GRPO 的 baseline 选择

GRPO 选 baseline = $\text{mean}(\{r_i\}_{i=1}^G)$，即组内均值。

**它和 V(s) 的关系**：

$$\text{mean}(\{r_i\}) = \frac{1}{G} \sum_{i=1}^G r(q, o_i) \xrightarrow{G \to \infty} \mathbb{E}_{o \sim \pi_{old}}[r(q, o)] = V_{\pi_{old}}(q)$$

所以**组内均值是 $V_{\pi_{old}}(q)$ 的 Monte Carlo 估计**。当 G 足够大时近似 V(q)。

#### 为什么不学 critic 更好

学一个 critic V_φ 有 **两个误差来源**：
1. **训练误差**：V_φ 估计不准（尤其训练初期）
2. **分布偏移**：V_φ 在历史数据上训，可能 generalize 不到新状态

组内均值没有这些问题——它是**当前 policy 下的直接采样估计**。

**代价**：需要采样 G 次，rollout 成本是 PPO 的 G 倍（但 PPO 只需要 1 次）。

### 🔍 GRPO vs PPO 的 trade-off

| 维度 | PPO (with critic) | GRPO (no critic) |
|---|---|---|
| Baseline 来源 | V_φ 网络 | 组内均值 |
| Baseline 无偏性 | 训练时有偏 | **无偏**（MC 估计）|
| Rollout 成本 | 1× | G×（G=64 则贵 64 倍）|
| 显存 | 高（有 critic）| 中（无 critic）|
| 工程复杂度 | 高 | 中 |
| 适用任务 | 通用 | 推理（reward 简单）|

### 🔥 高频追问 Top 3

**Q：G=1 时 GRPO 变成什么？**

A：**没有意义**——只有 1 个样本时，mean(r) = r，std=0，advantage = 0/0 未定义。

实际上 GRPO 要求 G ≥ 2 才有意义。

**更深层**：G 太小时 baseline 估计方差极大，GRPO 退化成无 baseline 的 REINFORCE，训练不稳。

经验：**G ≥ 8 才有意义，G = 16-64 是 sweet spot**。

**Q：GRPO 的 rollout 成本是 PPO 的 G 倍，为什么还划算？**

A：表面看是 G 倍，但有几个抵消因素——

- **批处理**：G 个响应可以**并行生成**（batch inference），实际墙钟时间不是 G 倍
- **省显存**：不用 critic，可以用更大的 batch 或更大的 base model
- **训练步数少**：组内多样本提供更可靠 gradient，**每步学到的更多**，总训练步数减少
- **稳定性**：更稳定意味着不需要回滚、调参成本低

实际工业经验——**GRPO 的总训练成本比 PPO 略高，但效果好且工程友好**。对推理任务来说值得。

**Q：能不能 GRPO + critic 一起用？**

A：可以——这是个研究方向。

思路：
- Critic 提供 per-token V(s)
- 组内均值提供 per-response baseline
- 两者结合，advantage 更精细

但实践中**没有显著收益**——增加了 critic 的复杂度，但 GRPO 单独已经足够好。所以纯 GRPO 仍是主流。

### ⚠️ 常见陷阱

1. 不要说"组内均值是个 trick"——它有严格的理论基础
2. 要懂 baseline 不改变梯度期望但降方差的原理
3. G=1 时 GRPO 没意义

### 🏢 大厂偏好

- **DeepSeek**：必问
- **强化学习方向研究岗**：会从理论角度深入

---

## Q4：PRM vs ORM：过程奖励 vs 结果奖励 ⭐⭐⭐⭐

### 🎯 一句话标答

> **ORM**（Outcome Reward Model）只看最终答案对错——简单但稀疏；**PRM**（Process Reward Model）给每个推理步骤打分——精细但贵。**ORM 更工程友好**，**PRM 理论更优**。R1 主要用 ORM，DeepSeek-Math 探索过 PRM。

### 🗣️ 30 秒口语版

"两种 reward 模型反映了**对推理任务的两种思路**——

**ORM（结果奖励）**：
- 输入：(prompt, complete_response)
- 输出：标量分数，反映最终答案的好坏
- 训练数据：(prompt, response, correct/wrong)

**PRM（过程奖励）**：
- 输入：(prompt, response, step_k)
- 输出：第 k 步的分数（这一步推理是否正确）
- 训练数据：(prompt, response with steps, step-level annotations)

**ORM 的优劣**：

优势：
- 训练数据容易获取（只需要最终答案对错）
- 实现简单，工程友好
- DeepSeek-R1 主要用 ORM

劣势：
- Reward 稀疏（推理 100 步只在末尾给信号）
- 学不会"哪一步错了"
- 对长推理任务不利

**PRM 的优劣**：

优势：
- 信号密集，每步都有反馈
- 能指出错误的具体位置
- 理论上更适合长 CoT

劣势：
- 标注极贵（需要专家标每一步）
- 训练复杂
- Reward Hacking 可能（模型学会'看起来正确'的步骤）

**实际选择**：
- DeepSeek-Math：PRM 探索
- DeepSeek-R1：ORM + rule-based（最终答案正确性 + 格式正确性）
- OpenAI o1：据传也用 PRM
- 工业主流：以 ORM 为主，PRM 作为补充"

### 🔍 ORM 与 PRM 详细对比

| 维度 | ORM | PRM |
|---|---|---|
| 监督粒度 | 整个响应 | 每个推理步骤 |
| 标注成本 | 低（最终答案） | 极高（步级专家标注）|
| 训练数据规模 | 容易做大 | 难以做大 |
| Reward 稀疏性 | 稀疏（只末尾） | 密集（每步） |
| 适合任务 | 数学/代码（答案可验证） | 复杂多步推理 |
| Reward Hacking | 较少 | 较多（"装作对"） |
| 工程复杂度 | 低 | 高 |
| 代表用例 | DeepSeek-R1 | DeepSeek-Math, o1（传） |

### 💡 Rule-based ORM 的简洁

DeepSeek-R1 的 ORM 极其简单——

```python
def deepseek_r1_reward(prompt, response):
    # 1. 格式 reward
    format_correct = check_format(response)  # 检查 <think>...</think><answer>...</answer>
    
    # 2. 准确性 reward
    if is_math_problem(prompt):
        predicted_answer = extract_answer(response)
        correct_answer = ground_truth[prompt]
        accuracy = (predicted_answer == correct_answer)
    elif is_code_problem(prompt):
        accuracy = run_test_cases(response)
    
    # 组合
    return alpha * format_correct + beta * accuracy
```

完全 rule-based，**不需要训 RM**。

为什么 work？因为对**数学/代码这类有客观正确性的任务**，规则比 RM 更可靠。RM 可能犯错（数学题答对但 RM 给低分），规则不会。

这也是为什么 GRPO + Rule-based ORM 在 DeepSeek-R1 上效果惊人——**绕开了 RM 的所有问题**（标注、训练、reward hacking）。

### 🔥 高频追问 Top 3

**Q：PRM 的标注怎么做？**

A：典型方法——

**人工标注（金标准）**：
- 给标注者看数学题的解答步骤
- 每步标 "correct / incorrect / neutral"
- 极贵：一道题可能 5-10 美元

**自动标注（PRM800K 等数据集）**：
- 用 OpenAI 的 PRM800K 数据集（800K 步级标注）
- 数据来源是 GPT-4 + 人工筛选

**Monte Carlo 标注**：
- 从某步开始让 LLM 继续推理 N 次
- 看完成正确答案的比例 → 这步的 reward
- DeepSeek-Math 用过这种方法

**Q：PRM 的 reward hacking 怎么应对？**

A：PRM 的 reward hacking 比 ORM 严重——

典型 hack：
- 模型输出"看起来合理但实际无意义"的步骤
- 每步在 PRM 上得高分，但整体推理跑偏

应对：
- **组合 PRM + ORM**：PRM 提供密集信号，ORM 作为最终检验
- **PRM ensemble**：多个 PRM 投票，降低单一漏洞
- **定期重训 PRM**：跟上 policy 的演化

但 reward hacking 是 PRM 的根本困难，没有完美解决。

**Q：R1 完全没用 PRM 吗？**

A：**主要用 ORM**，但 R1 论文也提到 PRM 是个探索方向。

DeepSeek 的判断——
- ORM + rule-based 在数学/代码上已经够强
- PRM 的训练成本和 hacking 风险大于收益
- R1 选择了"简单可靠"的 ORM 路线

但对**逻辑推理、复杂规划**等任务，PRM 可能更有价值。这是开放问题。

### ⚠️ 常见陷阱

1. 不要把 PRM 神化——工业上 ORM 更主流
2. Rule-based reward 不丢人——R1 证明了它的强大
3. PRM 的 hacking 比 ORM 严重

### 🏢 大厂偏好

- **DeepSeek**：会问 ORM 设计细节
- **OpenAI 系**：会问 PRM 的工程实现

### 📚 延伸阅读
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) - OpenAI 的 PRM 工作
- [PRM800K Dataset](https://github.com/openai/prm800k)

---

## Q5：DeepSeek-R1 训练全流程 ⭐⭐⭐⭐⭐（必问）

### 🎯 一句话标答

> R1 的训练是**四阶段管道**：①**Cold Start**（少量长 CoT SFT）→ ②**Reasoning RL**（GRPO + 规则 reward）→ ③**Rejection Sampling SFT**（用 RL 结果生成 SFT 数据 + 通用数据）→ ④**Final RL**（覆盖推理 + 通用对齐）。

### 🗣️ 30 秒口语版

"R1 训练流程是 DeepSeek 论文的核心贡献，必须能完整讲清。

**前提：R1-Zero 与 R1 的关系**

- **R1-Zero**：从 base model 直接 RL，**没有 SFT**——证明 RL 单独能产生推理能力，但语言混乱
- **R1**：在 R1-Zero 基础上加 SFT 修复语言问题——是工业可用版本

**R1 的四阶段管道**：

**阶段 1：Cold Start（小量 SFT）**
- 准备**几千条**高质量长 CoT 数据
- 数据来源：人工 + R1-Zero 的输出 + few-shot prompt 生成
- 用这些数据 SFT base model
- 目的：让模型先有**基本的 CoT 格式和语言能力**

**阶段 2：Reasoning RL（GRPO + 规则 reward）**
- 在 Cold Start 模型上做 GRPO
- Reward = 数学答案正确性 + 格式正确性（rule-based）
- 训练数据：数学、代码、逻辑题
- 这一步让模型**学会复杂推理**，CoT 越来越长，出现 'aha moment'

**阶段 3：Rejection Sampling + 综合 SFT**
- 用阶段 2 的 RL 模型生成大量响应
- **Rejection sampling**：用规则筛选出正确的高质量响应
- 加上**通用对话数据**（写作、知识问答、安全等）
- 在 base model 上重新 SFT（约 80 万条数据）

**阶段 4：Final RL（全场景）**
- 再做一次 RL，覆盖**所有场景**：
  - 推理：rule-based reward（保持准确性）
  - 通用：reward model（保持对齐）
- 让模型在保持推理能力的同时具备通用对话能力

**最终产物**：DeepSeek-R1 既能解奥数题，又能正常对话写代码——**通才推理模型**。"

### 📊 完整流程图

```
Base Model (DeepSeek-V3-Base)
        │
        ▼
┌──────────────────────────────────┐
│  阶段 1: Cold Start SFT          │
│  · 数据: 几千条长 CoT            │
│  · 目的: 学基本 CoT 格式         │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  阶段 2: Reasoning RL (GRPO)     │
│  · 数据: 数学/代码/逻辑          │
│  · Reward: 答案正确性 + 格式     │
│  · 涌现: 长 CoT, aha moment      │
└──────────────────────────────────┘
        │
        ▼
   生成大量响应 → Rejection Sampling
        │
        ▼  60w 推理数据 + 20w 通用数据
┌──────────────────────────────────┐
│  阶段 3: 综合 SFT (from base)    │
│  · 数据: 80w 条混合              │
│  · 重新从 base model SFT         │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  阶段 4: Final RL (全场景)       │
│  · 推理: rule-based reward       │
│  · 通用: reward model            │
│  · 安全对齐                       │
└──────────────────────────────────┘
        │
        ▼
   DeepSeek-R1 (最终模型)
```

### 💡 流程设计的关键洞察

**为什么需要 Cold Start？**
- R1-Zero 证明纯 RL 能涌现推理能力，但输出**语言混乱**（中英混杂、可读性差）
- 一点点 SFT 数据就能修复——让 RL 有个"良好的语言起点"

**为什么阶段 3 要重新从 base SFT？**
- 阶段 2 的 RL 模型在推理上很强，但**通用能力退化**（语言、对话）
- 用其生成的高质量推理数据 + 通用数据，在 base 上重新 SFT——**重新平衡能力**

**为什么阶段 4 还要 RL？**
- SFT 只能"模仿"已知模式
- Final RL 让模型在所有场景上**继续超越人类标注水平**
- 同时巩固推理能力 + 对齐通用偏好

**精髓**：**RL + SFT 交替**——每个阶段解决一个问题，最终融合所有能力。

### 🔥 高频追问 Top 3

**Q：R1 一共训练了多少 token？多少卡？**

A：DeepSeek 没有完全公开，但根据论文披露——
- **Base model（V3-Base）**：14.8T token 预训练
- **R1 训练**：相对较短，但 RL 阶段 rollout 成本高
- **总训练成本**：约 600 万美元（包含 base 预训练）——比 GPT-4 估算的 1 亿美元便宜得多

显卡：约 2048 张 H800。

**Q：阶段 2 的 RL 训练能直接出强模型，为什么还需要阶段 3-4？**

A：阶段 2 后的模型在**纯推理 benchmark**（AIME、MATH 等）上已经很强，但有几个问题：
- **语言能力退化**：在数学题上突出，但日常对话变差
- **格式不通用**：只擅长 `<think>...</think>` 格式
- **安全对齐不足**：可能输出有害内容

阶段 3-4 是为了**把推理能力嫁接到通用基座上**，让 R1 成为"既能解奥数又能聊天写代码"的通才。

**Q：能不能复现 R1？**

A：可以——这是 2025 年开源社区的热门方向。

复现要点：
- **Base model**：用 LLaMA-3.1 70B 或 Qwen2.5 等强 base
- **SFT 数据**：可以用 DeepSeek-R1 蒸馏（公开 R1 输出）
- **GRPO 实现**：HuggingFace TRL 库已经支持
- **算力**：8 卡 A100/H100 + 几周训练

已经有不少社区项目：Open-R1、TinyZero、SimpleRL-Zoo 等。

**门槛**：数据 + 算力 + 调参经验。不是不可能，但需要工程能力。

### ⚠️ 常见陷阱

1. **R1 不是一阶段训出来的**——是四阶段管道
2. **Cold Start 数据量很小**（几千条）——很多人以为是大规模 SFT
3. **阶段 3 从 base 重新 SFT**——不是在 RL 模型上继续

### 🏢 大厂偏好

- **DeepSeek 系**：必问，要求详细到每个阶段的数据来源
- **国内大模型团队**：复现 R1 是热门方向

### 📚 延伸阅读
- [DeepSeek-R1 论文](https://arxiv.org/abs/2501.12948) - 必读
- [Open-R1](https://github.com/huggingface/open-r1) - 开源复现

---

## Q6：R1-Zero 与 R1 的区别 ⭐⭐⭐⭐

### 🎯 一句话标答

> **R1-Zero**：从 base 直接 RL，**没有任何 SFT**——证明 RL 单独能涌现推理能力，但语言混乱；**R1**：在 R1-Zero 基础上加 SFT 修复语言问题，是工业可用版本。**R1-Zero 是科学发现，R1 是工程产品**。

### 🗣️ 30 秒口语版

"R1-Zero 是 DeepSeek 论文的**真正惊喜**，比 R1 更有学术意义。

**R1-Zero 的训练**：
- 直接在 DeepSeek-V3-Base 上做 GRPO
- **没有 SFT！**
- Reward = 答案正确性 + 格式（rule-based）
- 训练数据：数学/代码题

**惊人发现**：
1. **推理能力涌现**：CoT 越来越长，从几百 token 涨到几千甚至上万
2. **Aha Moment**：模型学会**自我反思**——'wait, let me reconsider...'
3. **数学能力超群**：AIME 2024 上达到 71%（超过 o1-preview）

**但有缺陷**：
- 输出语言混乱（中英混杂）
- 可读性差（推理过程难懂）
- 不会通用任务（写作、对话等）

**所以做 R1**：
- 把 R1-Zero 的'推理能力'保留
- 加 SFT + 通用 RL 修复语言和通用任务
- 得到**工业可用**的最终版本

**学术意义**：
R1-Zero **打破了'必须先 SFT 再 RL'的共识**。证明——
- 只要 reward 信号准确，RL 单独可以从 base 训出推理能力
- SFT 不是必需的；它是工程上让训练稳的'拐杖'
- 这开启了'纯 RL 训 LLM'的研究路线

R1-Zero 的影响可能比 R1 更深远。"

### 🔍 R1-Zero vs R1 对比

| 维度 | R1-Zero | R1 |
|---|---|---|
| 训练流程 | Base → RL | Base → SFT → RL → SFT → RL |
| 训练阶段 | 1 | 4 |
| 推理能力 | 强（AIME 71%） | 强（AIME 79.8%） |
| 语言能力 | 差（中英混杂） | 强 |
| 通用任务 | 差（只会推理） | 强 |
| 可读性 | 差 | 好 |
| 工程可用 | 否 | **是** |
| 学术意义 | **极高** | 中 |

### 💡 R1-Zero 的训练曲线

DeepSeek 论文图：

```
推理长度（token）
   ↑
   │              ╱  ← Aha moment 发生
12K│             ╱       (~1K steps)
   │            ╱
 8K│         ╱╱
   │      ╱╱
 4K│   ╱╱
   │ ╱
   └─────────────────→ Training Steps
   0   500  1000  1500


AIME Accuracy
   ↑
80%│             ╱─── ← 接近 o1
   │           ╱
60%│         ╱
   │       ╱
40%│    ╱
   │  ╱
20%│╱
   └─────────────────→ Steps
```

**关键观察**：
- **响应长度自然增长**——模型自己学会"想更久"
- **准确率与长度同步上升**——长 CoT 真的有用
- **没有 SFT 引导**——纯 reward signal 推动

### 🔥 高频追问 Top 3

**Q：R1-Zero 为什么会出现 Aha Moment？是怎么"学会"自我反思的？**

A：这是**涌现现象**，没有 prompt 明确要求。机制可能是——

- Base model 已经见过包含 "wait", "let me reconsider" 等表达的文本（互联网上的解题）
- 早期推理时随机使用这些表达
- 当使用 self-reflection 后能更频繁地得到正确答案 → reward 高 → 频率被强化
- 慢慢"学到" reflection 是个有效的推理策略

类似自然选择——**有效的策略被 reward 信号筛选出来**。

注意：**Aha moment 不是 R1 特有**——其他强推理模型（o1 等）可能也有，只是 DeepSeek 论文公开命名了。

**Q：R1-Zero 的"语言混乱"具体是什么样？**

A：DeepSeek 论文报告——
- 中英文随机切换
- 重复某些 token
- 输出可读性差（人难理解推理过程）
- 但**最终答案对**

例子：
```
"To solve this 我们需要 first compute the value 这意味着 ..."
```

这是因为——base model 是多语言的，pure RL 只奖励**正确性**，不奖励语言一致性。模型会用所有可用的"工具"（包括随意切换语言）来得到正确答案。

R1 通过 SFT 引导，让模型说统一的语言。

**Q：R1-Zero 的发现对未来 LLM 研究有什么启示？**

A：几个深远启示——

**1. RL 比 SFT 更接近"学习"的本质**
- SFT 是模仿，能力上限是数据
- RL 通过 reward 探索，可以**超越数据**

**2. 推理能力可以"涌现"**
- 不需要人工设计推理模板
- 只要 reward 信号准，模型自己会找

**3. 数据质量 < reward 质量**
- 传统 LLM 拼数据，现在拼 reward 设计
- Reward Engineering 成为关键技能

**4. 多语言能力是双刃剑**
- Base model 的多语言能力让 R1-Zero 能用任何方式推理
- 但也导致语言混乱
- 未来需要更好的"语言一致性 reward"

这些启示影响了 OpenAI o3、Qwen-QwQ、Kimi-K0 等后续工作。

### ⚠️ 常见陷阱

1. 不要把 R1-Zero 和 R1 混为一谈
2. R1-Zero 的"语言混乱"是 reward 设计问题，不是 RL 本身的问题
3. Aha moment 是涌现现象，不是显式设计

### 🏢 大厂偏好

- **DeepSeek**：必问
- **国内推理模型团队**：复现 R1-Zero 是研究热点

### 📚 延伸阅读
- [DeepSeek-R1 Paper](https://arxiv.org/abs/2501.12948) §2.2 R1-Zero
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero) - R1-Zero 小规模复现

---

## Q7：推理能力涌现：Aha Moment 与长 CoT ⭐⭐⭐⭐

### 🎯 一句话标答

> 推理模型的核心涌现现象——**长 CoT 自发延长**（从几百到上万 token）+ **Self-reflection 出现**（'wait, let me reconsider'）+ **回溯能力**（发现错误后回退）。这些都不是显式训练的，是 RL 训练中自然涌现的。

### 🗣️ 30 秒口语版

"R1 出现的'推理涌现'现象可以分三类——

**1. 长 CoT 自发延长**

训练初期，模型的推理长度约几百 token。随着 RL 训练，长度自然增长到几千甚至上万——**没有任何 prompt 或 reward 直接奖励长度**，模型自己发现"想得更久" → "更可能算对" → "reward 更高"。

**2. Self-reflection 涌现**

模型学会**自我审视**：
- "Wait, let me check this again..."
- "Hmm, this doesn't seem right..."
- "Actually, I should reconsider..."

DeepSeek 论文把这命名为 **Aha Moment**——training 进行到某一刻，这种语言模式突然出现并被强化。

**3. 回溯（Backtracking）涌现**

模型在发现错误后**主动回退**：
- "Let me redo step 3..."
- "Wait, I made a mistake at..."

这是高级推理能力，传统 SFT 很难学到。

**这些现象为什么发生**？

不是因为人工设计，而是因为——
- Base model 已经见过这些表达（互联网上的解题、辩论、思考过程）
- RL 给出"答案对就好"的简单 reward
- 这些策略能提高准确率 → reward → 被强化 → 频率上升

**核心 insight**：**Reward 信号足够纯净时，复杂的认知策略可以涌现**——这是 RL 的魅力，也是它超越 SFT 的根本原因。

类似的现象——
- OpenAI o1：内部 reasoning 也有 self-reflection
- Qwen-QwQ：开源复现，类似 aha moment
- Kimi-K0：长 CoT 推理"

### 📊 R1 推理长度统计（论文数据）

| 训练阶段 | 平均响应长度 | AIME 准确率 |
|---|---|---|
| Step 0 (initial) | ~400 token | ~15% |
| Step 200 | ~1500 token | 30% |
| Step 500 | ~3000 token | 50% |
| Step 1000 | ~6000 token | 65% |
| Step 1500+ | ~10000+ token | 71% |

**观察**：长度和准确率**几乎线性相关**——更长的推理 = 更高的准确率。

### 💡 涌现的语言模式（R1-Zero 实例）

DeepSeek 论文的真实输出片段（典型 R1-Zero 响应）：

```
Let me solve this step by step.
Step 1: First, I'll compute ...
       After calculation, I get x = 4.

Hmm, wait. Let me double-check this calculation.
Going back to step 1, I had ... 
Actually, I think I made an arithmetic error.
Let me redo this more carefully...

(重新计算)
OK, so the correct value is x = 5.

Now continuing with step 2 using x = 5...
```

**关键元素**：
- **显式标记 step**（"Step 1", "Step 2"）
- **自我审视**（"wait", "let me double-check"）
- **回溯**（"going back to", "let me redo"）
- **承认错误**（"I made an error"）

这些都不是 prompt 要求的，是 model 自己学会的。

### 🔥 高频追问 Top 3

**Q：能不能加速涌现？比如更好的 base model 或 reward？**

A：可以——这是当前研究热点。

**加速涌现的方法**：
- **更强的 base model**：base 能力越强，涌现越快（R1 在 V3-Base 上比小模型快）
- **更精细的 reward**：rule-based + PRM 混合
- **Cold start data**：少量长 CoT SFT 数据能"启动"涌现
- **课程学习**：从简单题到复杂题循序渐进

**当前最快**：在 32B 模型上几千 step 能看到涌现迹象。

**Q：涌现能转移到其他任务吗？**

A：**部分能**。

R1 在数学/代码上训出来的推理能力，**部分**迁移到其他场景：
- 逻辑推理：迁移好
- 物理/化学题：迁移好
- 通用对话：差（这就是为什么需要阶段 3-4）
- 创意写作：几乎不迁移

经验：**"推理"是个相对窄的能力**——擅长可验证的逻辑步骤；不擅长开放性、主观性任务。

**Q：为什么小模型不容易涌现？**

A：根本原因——**base 能力不足**。

涌现需要 base model 有：
- 足够的语言能力（能写连贯的 CoT）
- 足够的知识（知道解题步骤）
- 足够的推理 priors（能识别错误）

小模型（< 10B）往往这些 priors 不够强——RL 训练时学不到有意义的策略。

但**蒸馏可以救小模型**——用大 R1 蒸馏 1.5B/7B 模型，小模型能继承部分推理能力。DeepSeek 也发布了蒸馏版（R1-Distill 系列）。

### ⚠️ 常见陷阱

1. 涌现不是"魔法"——它是 reward 信号驱动的自然选择
2. 长 CoT 不等于强推理——质量 > 长度
3. 涌现需要 base 能力，小模型难以从零涌现

### 🏢 大厂偏好

- **DeepSeek / 字节**：必问涌现机制
- **研究岗**：会问如何加速涌现

---

## Q8：GRPO 的局限与替代方案 ⭐⭐⭐

### 🎯 一句话标答

> GRPO 的主要局限——**rollout 成本高（G 倍）、依赖可验证 reward、对 reward 设计敏感**。替代/改进方案：**GRPO + PRM**、**DPO 用于推理**、**Self-Play**、**Iterative GRPO**。

### 🗣️ 30 秒口语版

"GRPO 不是银弹，有几个明显局限——

**局限 1：Rollout 成本高**

每个 prompt 要采样 G=64 个响应——计算量是 PPO 的 64 倍。对长 CoT（每个响应 10K+ token），这是巨大开销。

**应对**：
- 用专门的 inference engine（如 vLLM）批量生成
- 减少 G（牺牲 baseline 估计精度）
- 优化 KV Cache 重用

**局限 2：依赖可验证 Reward**

GRPO 在数学/代码上 work 因为有客观的对错。对**主观任务**（写作、对话），rule-based reward 难定义。

**应对**：
- 数学/代码用 GRPO
- 通用任务用 DPO 或 PPO + RM
- 混合训练（阶段 3-4 of R1）

**局限 3：Reward Hacking**

虽然 GRPO 的 hacking 比 PPO 轻，但仍存在：
- 模型可能学到"装作对"的输出
- Format reward 可能被 hack
- 在复杂 reward 下加剧

**应对**：
- 多个 reward signal 组合（不能让单一 reward 主导）
- 定期人工抽查
- 用 PRM 监督过程

**局限 4：长上下文压力**

R1 的 CoT 越来越长（10K+ token），意味着——
- 训练时需要长上下文支持
- KV Cache 显存压力大
- Attention 计算 O(n²) 主导

**应对**：
- Flash Attention v3
- 长上下文优化（YaRN 等）
- Sequence Parallel

**替代方案**：
- **DPO-iter**：DPO + 迭代生成偏好对，简单可用
- **Self-Play**：模型自己生成对手响应
- **GRPO + PRM**：用 PRM 补充 outcome reward
- **REINFORCE++**：进一步简化的 RL 变体"

### 🔍 GRPO 的工程困难

| 困难 | 原因 | 应对 |
|---|---|---|
| Rollout 慢 | G=64 倍计算 | 高效推理引擎 + 减小 G |
| 长 CoT 显存 | 10K+ token 序列 | Flash Attention + SP |
| Reward 设计难 | 主观任务无客观 reward | 混合 reward / 任务分类 |
| Hacking 风险 | 长 CoT 给更多 hack 空间 | Multi-reward + 人工监控 |
| 调参敏感 | β, G, lr 等多超参 | 经验配方 + 迭代 |

### 🔥 高频追问 Top 3

**Q：GRPO 之后还会有什么新算法？**

A：可能的方向——

**1. 简化 GRPO**：
- REINFORCE-style：进一步去掉 PPO 的 clipping
- 更小的 G + 更稳的 baseline

**2. 多阶段 reward**：
- 阶段 1 用 outcome
- 阶段 2 用 PRM 精细化
- 多 reward 协同

**3. Online + Self-improve**：
- 模型生成 → 自评 → 训练
- 持续 self-play 增强

**4. 跨任务推理迁移**：
- 在数学上训的推理能力，迁移到代码、物理等
- 跨任务 reward 设计

**5. 多步骤 RL**：
- 把推理拆成多步，每步独立优化
- 类似 Multi-step DPO

**Q：GRPO 能扩展到 Agent 任务吗？**

A：理论可以，实践挑战大——

Agent 任务的难点：
- Reward 极稀疏（任务完成才有 reward）
- 步骤极多（10+ 工具调用）
- Credit assignment 难

**适配 GRPO**：
- 把"任务成功"作为 outcome reward
- Group sampling 同一任务多次
- Advantage 按 group 算

但实际效果——尚未广泛验证。Agent 训练仍以传统 PPO 为主。

**Q：如果我要做推理模型，没有 R1 那么多资源，怎么入门？**

A：**蒸馏 + 小规模 RL**——

1. **蒸馏起步**：用 R1 / o1 等模型蒸馏 SFT 数据，先训出一个会 CoT 的小模型
2. **小规模 RL**：在数学数据集（如 GSM8K + MATH）上做 GRPO，G=8-16
3. **持续迭代**：rejection sampling + 重训
4. **资源**：8 × A100 + 几周

工具：
- Open-R1（HF 复现）
- TRL（GRPO 实现）
- vLLM（高效 rollout）

社区资源越来越多，复现门槛在下降。

### ⚠️ 常见陷阱

- GRPO 不是终点，未来还会有更优算法
- 推理任务有 GRPO，通用任务 DPO 仍是主流
- Rollout 成本是 GRPO 的核心 bottleneck

### 🏢 大厂偏好

- **DeepSeek / Mistral**：会问 GRPO 改进方向
- **AI Infra**：会问 rollout 优化

---

## 📝 本章小结

| 关键点 | 你必须能脱口而出 |
|---|---|
| GRPO 核心 | 去掉 critic 的 PPO，用 group 均值作 baseline |
| Advantage 公式 | A_i = (r_i - mean) / std，per-response |
| GRPO 优势 | 无 critic，省显存，推理任务友好 |
| Group size G | 16-64，DeepSeek 用 64 |
| ORM vs PRM | outcome 简单可用 / process 精细但贵 |
| Rule-based ORM | 数学验证 + 格式检查，R1 主用 |
| R1 流程 | Cold Start SFT → Reasoning RL → 综合 SFT → Final RL |
| R1-Zero | 纯 RL（无 SFT），证明涌现，但语言乱 |
| Aha Moment | RL 训练中涌现的 self-reflection |
| 长 CoT 增长 | 响应长度自然从几百涨到上万 |
| GRPO 局限 | rollout 成本高、依赖可验证 reward |

## ✅ 自测题

1. **完整推导 GRPO 目标函数**，并解释每个组成部分的作用（特别是为什么 advantage 是 per-response 而非 per-token）
2. **为什么去掉 critic 后 GRPO 仍能 work**？从 baseline 理论角度回答
3. ORM vs PRM 各自适合什么场景？DeepSeek-R1 为什么主要选 ORM？
4. **画出 DeepSeek-R1 训练四阶段流程图**，详述每个阶段的数据来源和目的
5. R1-Zero 的 Aha Moment 是什么？为什么会涌现？
6. 给定场景"用 32B 模型训数学推理"，给出完整的 GRPO 训练配方（数据、超参、监控）
7. GRPO 的 rollout 成本是 PPO 的 G 倍，工程上如何优化？

---

> **本部分完成**：训练与对齐分册（第 6-11 章）已经完成。从预训练到 SFT、PEFT、RLHF、DPO、GRPO——LLM 训练流水线的全链路已经覆盖。下一章进入**推理优化**主题——KV Cache、Flash Attention、量化等工程话题，是 AI Infra 岗的高频区。
