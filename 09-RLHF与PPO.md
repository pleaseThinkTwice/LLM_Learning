# 第 9 章｜RLHF 与 PPO

> **本章定位**：RLHF 是 ChatGPT 出圈的关键，也是面试硬核区——**PPO 公式现场推导是必考题**。本章会把 PPO 的目标函数、Reward Model 训练、Reward Hacking 应对、工程难点等核心问题彻底讲清楚。

> **配套章节**：SFT → 第 7 章 / DPO → 第 10 章（深度推导） / GRPO → 第 11 章

---

## 📌 本章导航

| 序号 | 题目 | 难度 | 频率（字节/阿里/Meta/DeepSeek） | 类型 |
|---|---|---|---|---|
| Q1 | RLHF 整体三阶段流程 | ⭐⭐⭐⭐ | 6 / 4 / 3 / 3 | 必懂 |
| Q2 | **Reward Model 的训练** | ⭐⭐⭐⭐⭐ | **6 / 4 / 4 / 4** | 必问 |
| Q3 | **PPO 目标函数完整推导** | ⭐⭐⭐⭐⭐ | **9 / 5 / 6 / 4** | 杀手题 |
| Q4 | **PPO 的 Ratio Clipping** | ⭐⭐⭐⭐ | 6 / 3 / 4 / 3 | 数学 |
| Q5 | KL 正则化的作用 | ⭐⭐⭐⭐ | 5 / 3 / 3 / 3 | 概念 |
| Q6 | **Reward Hacking 与应对** | ⭐⭐⭐⭐⭐ | 6 / 3 / 4 / 4 | 深度 |
| Q7 | RLHF 工程难点 | ⭐⭐⭐⭐ | 5 / 3 / 3 / 4 | 工程 |
| Q8 | RLHF 变体：RLAIF / CAI / Best-of-N | ⭐⭐⭐ | 4 / 2 / 3 / 3 | 前沿 |

---

## Q1：RLHF 整体三阶段流程 ⭐⭐⭐⭐

### 🎯 一句话标答

> RLHF = **三阶段流水线**：①**SFT** 让 base model 学会指令格式；②训练**奖励模型 RM** 学人类偏好；③用 **PPO** 优化 policy 模型，目标是最大化 RM 给的奖励，同时不偏离 SFT 模型太远。

### 🗣️ 30 秒口语版

"RLHF 全称 Reinforcement Learning from Human Feedback，是 InstructGPT / ChatGPT 的核心训练范式。流程分三步——

**阶段 1：SFT（监督微调）**

用人工写的高质量指令-响应对训练 base model，让它学会指令格式。这是上一章讲的内容。

**阶段 2：训练奖励模型 RM**

收集**人类偏好数据**：对同一个 prompt，人类标注 A 比 B 好还是 B 比 A 好。用这些数据训练一个 RM——输入 prompt + 响应，输出标量分数。RM 通常基于 SFT 模型，把 LM head 换成 scalar head。

**阶段 3：PPO 强化学习**

把 SFT 模型作为初始 policy，用 PPO 优化它：
- 让 policy 生成响应
- 用 RM 给响应打分
- 用 PPO 调整 policy，让它生成更高分的响应
- **同时**用 KL 散度约束 policy 不偏离 SFT 太远（防止 reward hacking）

**为什么需要 RM？为什么不直接用人评打分**？因为人评太慢——PPO 训练要做百万次 rollout，每次都人评不现实。RM 是个**人类偏好的 proxy**——一次性训好，之后无限调用。

**整个 pipeline 一句话**：SFT 学'格式'，RM 学'好坏'，PPO 把'好坏'信号注入 policy。"

### 📐 三阶段流程图

```
阶段 1：SFT
   Base Model (LLaMA, etc.)
        ↓ 监督微调
        ↓ 数据：人工 demonstration
   SFT Model

阶段 2：Reward Model
   SFT Model（复制一份）
        ↓ 换 LM head 为 scalar head
        ↓ 数据：(prompt, response_chosen, response_rejected)
        ↓ Loss: pairwise ranking
   Reward Model RM

阶段 3：PPO
   SFT Model（作为 init policy 和 reference）
        ↓ 与环境交互（生成 response）
        ↓ RM 打分作为 reward
        ↓ PPO 更新 policy，KL 约束
   RLHF Model
```

### 🔥 高频追问 Top 3

**Q：为什么不能直接把人评信号端到端反向传播？**

A：因为人评是**离散、有噪声、非可微**的——你没法对"人觉得 A 比 B 好"做梯度下降。RLHF 的"巧妙"就在于：
- RM 把人评编码成**可微的标量分数**
- RL（PPO）允许我们对**采样生成的响应**做策略优化（policy gradient）

这是 RL 解决"不可微 reward"的经典套路，被 RLHF 借用过来。

**Q：能不能跳过 SFT 直接做 RLHF？**

A：**几乎不行**。Base model 还不会按指令回答，生成的响应大多是噪声——RM 给的分数也是噪声中的噪声，PPO 学不到有用的信号。

**SFT 是 RLHF 的"启动器"**——让模型先有个合理起点，PPO 才能在合理的输出空间内优化。

**Q：RLHF 之后还能继续做什么？**

A：现代 alignment pipeline 通常是——
- **SFT → RLHF → RLAIF → Constitutional AI**
- 或者 **SFT → DPO**（更简洁）
- 或者 **SFT → RLHF（多轮迭代）**

RLHF 不是终点，是个起点。Anthropic 的 Claude 用了 Constitutional AI；OpenAI 的 GPT-4 据传用了多轮 RLHF。

### ⚠️ 常见陷阱

1. 不要把 RLHF 和 SFT 混为一谈——它们是两个阶段
2. 要懂 RM 的角色——是 reward 的代理，不是评估器
3. KL 约束不是可选——没有它 PPO 必然崩

### 🏢 大厂偏好

- 这题是**铺垫题**，会从这里进入 PPO 推导和 Reward Hacking

### 📚 延伸阅读
- [InstructGPT 论文](https://arxiv.org/abs/2203.02155) - RLHF 的奠基之作

---

## Q2：Reward Model 的训练 ⭐⭐⭐⭐⭐（必问）

### 🎯 一句话标答

> RM 用 **pairwise ranking loss** 训练——对同一 prompt 的两个响应 (chosen, rejected)，让 RM 给 chosen 的分数高于 rejected，公式是 `loss = -log(σ(r_chosen - r_rejected))`，这是 Bradley-Terry 模型的对数似然。

### 🗣️ 30 秒口语版

"RM 的训练是 RLHF 的关键环节，分三步——

**第一步：数据收集**。
对每个 prompt，让模型（通常是 SFT 模型）生成 K 个不同响应（K=4-9）。**人类标注者**对这些响应做**两两比较**或**排序**。整理成 `(prompt, chosen, rejected)` 的偏好对。

**第二步：模型架构**。
RM 的 backbone 是 SFT 模型（参数完全继承），但把最后的 LM head（输出词表概率）**换成一个 scalar head**（输出一个分数）。也就是说——RM 把 (prompt + response) 编码成一个标量。

**第三步：训练损失**。
对每对 (chosen, rejected)，用 **Bradley-Terry 模型** 的负对数似然：

`loss = -log(σ(r_chosen - r_rejected))`

直觉理解——希望 r_chosen 比 r_rejected 大。如果两者相等，loss = -log(0.5) = 0.69；如果差距很大，loss 接近 0。

**关键细节**：

- 通常**只用 prompt 末尾 token 的 hidden state** 接 scalar head，因为它包含完整的响应信息
- RM 训练**不需要太多数据**——10K 到 100K 偏好对就够（InstructGPT 用 33K）
- RM 的 **规模和 SFT 模型同量级**——质量随规模提升

RM 训完后**冻结**，在 PPO 阶段作为 reward signal 提供者。"

### 📐 Bradley-Terry 模型推导

**核心假设**：每个响应 $y$ 有一个潜在的"质量分" $r(x, y)$，人类选择 chosen 优于 rejected 的概率：

$$P(y_c \succ y_r | x) = \frac{\exp(r(x, y_c))}{\exp(r(x, y_c)) + \exp(r(x, y_r))} = \sigma(r(x, y_c) - r(x, y_r))$$

**最大似然估计**：让模型预测的偏好概率匹配人类偏好。对数据集 $D = \{(x, y_c, y_r)\}$：

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_c, y_r) \sim D} [\log \sigma(r_\phi(x, y_c) - r_\phi(x, y_r))]$$

这是 RM 的标准训练损失。

### 💻 RM 实现框架

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # 继承 base model 的 backbone（去掉 LM head）
        self.backbone = base_model
        # 加一个 scalar head
        hidden_dim = base_model.config.hidden_size
        self.scalar_head = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, input_ids, attention_mask):
        # 跑 backbone 拿到 hidden states
        outputs = self.backbone(input_ids, attention_mask=attention_mask, 
                                output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        
        # 取每个 sequence 最后一个 token 的 hidden
        # 注意：是最后一个非 padding token
        seq_lens = attention_mask.sum(dim=1) - 1
        last_token_hidden = last_hidden[torch.arange(len(seq_lens)), seq_lens]
        
        # 投影到 scalar
        reward = self.scalar_head(last_token_hidden).squeeze(-1)
        return reward

# 训练 step
def rm_train_step(rm, batch):
    # batch: (chosen_ids, rejected_ids, chosen_mask, rejected_mask)
    r_chosen = rm(batch["chosen_ids"], batch["chosen_mask"])
    r_rejected = rm(batch["rejected_ids"], batch["rejected_mask"])
    
    # Bradley-Terry loss
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    return loss
```

### 🔍 RM 训练的关键细节

| 维度 | 配置 |
|---|---|
| 模型规模 | 与 SFT 同量级（6B-70B） |
| 数据量 | 10K-100K 偏好对 |
| Epoch | 1（防止过拟合）|
| 学习率 | 1e-6 到 5e-6（比 SFT 小）|
| Hidden 提取位置 | 最后一个非-padding token |
| Loss | Bradley-Terry pairwise |

### 🔥 高频追问 Top 3

**Q：为什么 RM 只训 1 epoch？过拟合是个问题吗？**

A：**是**。RM 极易过拟合——
- 偏好数据量小（10K-100K）
- 人类标注本身有噪声（标注者间一致率往往只有 60-70%）
- 多 epoch 训练让 RM "记住"训练集，而不是学到泛化的偏好

实验观察：训 2-3 epoch 后，**验证集准确率不升反降**，且后续 PPO 阶段更容易 reward hacking。

InstructGPT 论文明确建议 RM 只训 1 epoch。

**Q：RM 输出的"分数"有什么物理含义？**

A：**没有绝对含义，只有相对意义**。RM 输出是个未归一化的 logit——绝对值不重要，差值才有意义。

具体来说：
- $r(x, y_c) - r(x, y_r) > 0$ → chosen 偏好
- 差值大小反映偏好程度（对数赔率）

所以 RM 用作 reward 时，PPO 通常会**做 normalization**（如减去均值、按 batch 标准化），避免数值绝对值影响训练稳定性。

**Q：RM 的准确率多少算好？**

A：**典型在 65-75%**——这看起来很低，但已经接近**人类标注一致率的上限**（标注者之间也只有 60-70% 一致）。

经验：
- RM 准确率 < 60%：数据有问题或模型欠拟合
- 60-75%：正常范围
- > 80%：可能过拟合，要警惕

注意：**RM 准确率高不等于 RLHF 效果好**——RM 学到的是"训练分布的偏好"，PPO 阶段如果生成的响应在 OOD（out-of-distribution），RM 评分不可靠。

### ⚠️ 常见陷阱

1. RM 只训 1 epoch，不要多
2. 取最后一个非 padding token 的 hidden，不要简单取最后一个 token
3. RM 输出是相对值，不要赋予绝对含义

### 🏢 大厂偏好

- **字节 / 阿里 RLHF 团队**：必问，会让你设计完整数据 pipeline
- **Anthropic / OpenAI**：可能问 reward model calibration（校准）

### 📚 延伸阅读
- [InstructGPT 论文](https://arxiv.org/abs/2203.02155) §3.5 - RM 训练细节

---

## Q3：PPO 目标函数完整推导 ⭐⭐⭐⭐⭐（杀手题）

### 🎯 一句话标答

> PPO 的目标函数 = **clipped surrogate objective + value loss + KL penalty**——通过 ratio clipping 限制每步更新幅度，保证训练稳定；KL 约束 policy 不偏离 SFT 太远，防止 reward hacking。

### 🗣️ 30 秒口语版

"PPO 的推导可以分四个层次——

**层次 1：从 Policy Gradient 开始**

最朴素的强化学习目标是最大化期望回报：

`J(θ) = E_τ~π_θ [Σ R(s_t, a_t)]`

策略梯度定理告诉我们：

`∇J = E[Σ ∇log π(a|s) · A(s,a)]`

其中 A 是 advantage（动作优于基线的程度）。

**层次 2：Importance Sampling**

PG 需要每次更新都重新采样，效率低。**Importance Sampling** 允许用旧策略采样的数据训练新策略：

`J(θ) = E_τ~π_old [Σ (π_θ(a|s) / π_old(a|s)) · A]`

引入 **ratio** `r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)`。

**层次 3：Clipping 防止崩溃**

如果 ratio 偏离 1 太远，新旧策略差距大，importance sampling 估计就不准。**PPO 的核心创新**——加入 **clipped objective**：

`L^CLIP = E[min(r·A, clip(r, 1-ε, 1+ε)·A)]`

这个 min + clip 设计**惩罚大幅偏离**，保证更新稳定。

**层次 4：LLM 上的完整目标**

LLM RLHF 中的完整 PPO loss：

`L_total = L^CLIP - c1·L^VF + c2·L^Entropy - β·KL(π||π_SFT)`

其中：
- L^CLIP：clipped surrogate objective
- L^VF：value function loss（critic 的 MSE）
- L^Entropy：熵正则（鼓励探索）
- KL term：约束 π 不偏离 SFT 太远（防 reward hacking）

**reward 来源**：每个 token 生成时，KL 也算到 reward 里——`r_t = RM(响应) - β·log(π(a)/π_SFT(a))`。"

### 📐 完整数学推导

#### Step 1：策略梯度的基本形式

强化学习目标：
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t R(s_t, a_t)\right]$$

策略梯度定理：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^{\pi_\theta}(s_t, a_t)\right]$$

其中 $A(s, a) = Q(s, a) - V(s)$ 是 advantage，衡量动作比平均好多少。

#### Step 2：从 On-Policy 到 Off-Policy

朴素 PG 是 on-policy——每次更新都要重新采样。低效。

**Importance Sampling** 把策略期望换成 behavior policy 期望：

$$\mathbb{E}_{a \sim \pi_\theta}[f(a)] = \mathbb{E}_{a \sim \pi_{old}}\left[\frac{\pi_\theta(a)}{\pi_{old}(a)} f(a)\right]$$

引入 **probability ratio**：
$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

目标变为：
$$L^{CPI}(\theta) = \mathbb{E}_t[r_t(\theta) \cdot A_t]$$

CPI = Conservative Policy Iteration（TRPO 的前身）。

#### Step 3：PPO 的 Clipped Surrogate Objective

CPI 目标有个问题——如果 $r_t$ 远离 1，更新会很激进，可能让 policy 崩坏。

PPO 的解决方案——**clipped objective**：

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)\right]$$

其中 $\epsilon$ 通常取 0.2。

**直觉**：
- 当 $A_t > 0$（动作好）：希望 $r_t$ 增大，但不能超过 $1+\epsilon$
- 当 $A_t < 0$（动作差）：希望 $r_t$ 减小，但不能低于 $1-\epsilon$
- `min` 操作选**更保守的那个**

#### Step 4：Advantage 估计（GAE）

PPO 用 **Generalized Advantage Estimation** 估计 $A_t$：

$$A_t^{GAE} = \sum_{k=0}^{T-t} (\gamma \lambda)^k \delta_{t+k}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD error。

GAE 平衡了**偏差**（用 V 估计）和**方差**（用真实 reward 累加），$\lambda$ 控制权衡——λ=0 是 TD(0)，λ=1 是 Monte Carlo。

#### Step 5：LLM 上的完整 PPO Loss

LLM RLHF 中：
- $s_t$ = 前 t 个 token
- $a_t$ = 第 t 个生成的 token
- $\pi_\theta(a_t|s_t)$ = LM 输出的 token 概率
- $r_t$ 来自 RM（通常只在序列结尾给）+ KL penalty per token

**总目标**：

$$L^{TOTAL}(\theta) = L^{CLIP}(\theta) - c_1 \cdot L^{VF}(\theta) + c_2 \cdot L^{Entropy}$$

其中：
- $L^{VF} = (V_\phi(s_t) - V_t^{target})^2$ —— value function 的 MSE
- $L^{Entropy} = -\sum_a \pi(a) \log \pi(a)$ —— 熵奖励，鼓励探索
- $c_1, c_2$ 是权重（通常 $c_1 = 0.5, c_2 \in [0, 0.01]$）

**Reward 中的 KL penalty**（per token）：
$$r_t = \mathbb{1}[t = T] \cdot \text{RM}(x, y) - \beta \cdot \log\frac{\pi_\theta(a_t|s_t)}{\pi_{ref}(a_t|s_t)}$$

只在序列最后给 RM reward，但每个 token 都有 KL 惩罚。

### 💻 PPO 简化伪代码

```python
# 一次 PPO 迭代
def ppo_step(prompts, policy, ref_policy, reward_model, value_model):
    # 1. Rollout: 用 current policy 生成 responses
    with torch.no_grad():
        responses = policy.generate(prompts)
        old_log_probs = policy.log_prob(responses)
        ref_log_probs = ref_policy.log_prob(responses)
        values = value_model(prompts, responses)
        rewards = reward_model(prompts, responses)
    
    # 2. 计算 per-token reward (含 KL penalty)
    kl = old_log_probs - ref_log_probs
    token_rewards = -beta * kl  # 每个 token 的 KL 惩罚
    token_rewards[:, -1] += rewards  # 末尾加 RM reward
    
    # 3. 计算 advantage (GAE)
    advantages = compute_gae(token_rewards, values, gamma=1.0, lam=0.95)
    returns = advantages + values
    
    # 4. PPO 更新（多 epoch）
    for _ in range(ppo_epochs):
        # 计算新 log probs
        new_log_probs = policy.log_prob(responses)
        
        # Ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        new_values = value_model(prompts, responses)
        value_loss = F.mse_loss(new_values, returns)
        
        # 总 loss
        loss = policy_loss + c1 * value_loss - c2 * entropy
        loss.backward()
        optimizer.step()
```

### 🔥 高频追问 Top 3

**Q：为什么用 clipped objective 而不是 TRPO 那种硬约束？**

A：TRPO 用 KL 散度的硬约束：
$$\max L^{CPI} \quad \text{s.t.} \quad \text{KL}(\pi_{old} || \pi) \le \delta$$

需要解 trust region 问题——**计算复杂、求二阶导**。PPO 的 clipping 是 TRPO 的"软"近似——**用一阶方法就行**，更简单。

**经验**：PPO 的简化损失了一些理论保证，但实践效果接近 TRPO，且**更易工程化**。所以 PPO 成了主流。

**Q：为什么 LLM PPO 的 reward 只在序列结尾？**

A：因为 RM 是**对完整响应打分**——评估"这段回答好不好"。中间 token 没有独立的 reward 信号。

但**KL penalty 是 per-token 的**——每个 token 都会被惩罚偏离 reference。这种"末尾 reward + 全程 KL"的设计是 LLM PPO 的标准做法。

也有探索 **Process Reward Model (PRM)** —— 给每个步骤打分，用于推理任务（数学、代码）。但通用对话仍以 outcome reward 为主。

**Q：PPO epoch 数怎么定？**

A：典型 PPO 一次 rollout 后做 **2-4 个 inner epoch** 的更新——
- 太少：浪费 rollout 数据
- 太多：ratio 偏离 1 越来越远，clipping 触发频繁，反而不稳定

经验：4 个 inner epoch 是个甜点。

### ⚠️ 常见陷阱

1. **要能现场推导 ratio 和 clipped objective**——这是字节面试的重灾区
2. **KL penalty 在 reward 里 + 总 loss 里都有**——容易搞混
3. **value 模型和 policy 模型可以共享 backbone**——节省显存

### 🏢 大厂偏好

- **字节 / 阿里 RLHF 团队**：必问，要求白板写完整 loss
- **OpenAI / Anthropic**：会从理论角度问 trust region 和 PPO 关系

### 📚 延伸阅读
- [PPO 原论文](https://arxiv.org/abs/1707.06347)
- [InstructGPT 论文](https://arxiv.org/abs/2203.02155) - LLM 上的 PPO 实现细节
- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

---

## Q4：PPO 的 Ratio Clipping 详解 ⭐⭐⭐⭐

### 🎯 一句话标答

> Clipping 把 ratio 限制在 [1-ε, 1+ε] 内（ε=0.2）——**当更新会让概率比偏离太远且方向有利时，clip 阻止过度更新；不利时不阻止纠正**——这是 PPO 稳定性的核心。

### 🗣️ 30 秒口语版

"Clipping 的关键是理解 **min + clip 的非对称行为**。

完整公式：
`L^CLIP = E[min(r·A, clip(r, 1-ε, 1+ε)·A)]`

分情况看——

**Case 1：A > 0（动作好，希望增大其概率）**

- 如果 r < 1+ε：用 `r·A` 正常更新
- 如果 r > 1+ε：用 `(1+ε)·A`，clip 生效——**禁止过度增大**

**Case 2：A < 0（动作差，希望减小其概率）**

- 如果 r > 1-ε：用 `r·A` 正常更新
- 如果 r < 1-ε：用 `(1-ε)·A`，clip 生效——**禁止过度减小**

**关键的非对称性**：
- **Case 1 中 r 远大于 1+ε**：clip 起作用，限制更新
- **但 Case 1 中 r 远小于 1-ε**（动作好但当前概率反而下降）：clip **不起作用**，让模型自由纠正

这种'**惩罚激进更新但允许纠错**'的设计就是 PPO 的精髓。"

### 📐 Clipping 的可视化

```
情形 A > 0（好动作，希望增大概率）：

L^CLIP
   ↑
   |          /  
   |     ___/  ← 平台，r>1+ε 时被 clip
   |    /
   |   /  斜率 A
   |  /
   | /
   ──┼──────────────→ r
   0  1  1+ε

→ 防止 r 过度增大（已经够大了）

情形 A < 0（差动作，希望减小概率）：

L^CLIP
   |  
   ────       ← 平台，r<1-ε 时被 clip
       \
        \  
         \   斜率 -|A|
          \
           ──→ r
    1-ε  1

→ 防止 r 过度减小
```

### 🔥 高频追问 Top 3

**Q：clip 是双向的，为什么说 PPO 是"惩罚激进 + 允许纠错"？**

A：因为 `min` 操作的存在！

考虑 A > 0 时 r << 1（动作好但概率反而下降的情况）：
- `r·A` 很小（不利）
- `clip(r, 1-ε, 1+ε)·A` = `(1-ε)·A`（不利，但小于 r·A）
- `min` 取较小者 = `r·A`
- **梯度仍然推 r 增大** ✓ 允许纠错

而 A > 0 时 r >> 1+ε（已经够大）：
- `r·A` 很大（有利）
- `clip(r, 1-ε, 1+ε)·A` = `(1+ε)·A`（小于 r·A）
- `min` 取较小者 = `(1+ε)·A`，**没有进一步增长的梯度** ✗ 阻止激进

正是这个非对称的 min 操作，让 PPO 在"已经做对了"时不冒进，在"做错了"时仍能纠正。

**Q：ε 的选择有什么经验？**

A：
- **ε = 0.1**：保守，更新慢但稳
- **ε = 0.2**：标准，PPO 论文默认
- **ε = 0.3+**：激进，可能不稳

LLM RLHF 中常用 0.2，有些团队（如 Anthropic）用 0.1-0.2 之间。

**Q：clipping 触发频率多少正常？**

A：经验上 **10-30% 的样本** 触发 clipping 是正常的——
- 太少（<5%）：ε 太大，clipping 没起作用
- 太多（>50%）：策略变化太快，可能不稳

监控 clipping 触发率是 PPO 训练的重要指标。

### ⚠️ 常见陷阱

1. clipping 是双向的，不是只限制大方向
2. min 操作是"惩罚激进 + 允许纠错"的关键
3. ε 不要乱调，0.2 是经过实验验证的默认

---

## Q5：KL 散度正则化的作用 ⭐⭐⭐⭐

### 🎯 一句话标答

> KL 约束 `KL(π || π_SFT)` **防止 policy 偏离 SFT 太远**——避免 reward hacking、保留语言流畅性、防止 catastrophic forgetting，是 LLM RLHF 稳定的关键。

### 🗣️ 30 秒口语版

"KL 正则化在 LLM RLHF 中扮演三个角色——

**角色 1：防 Reward Hacking**

RM 是个不完美的 proxy——只在 SFT 模型生成分布附近训过。如果 policy 漂移太远，会生成 RM 训练时没见过的 OOD 响应，RM 给的高分可能是**虚假信号**——policy 会学到'欺骗 RM' 而非真正提升质量。

KL 约束让 policy '不要走太远'，**留在 RM 可信的区域**。

**角色 2：保留语言流畅性**

PPO 是个非语言模型目标，纯粹优化 RM 分数。没有 KL 约束的话，policy 可能学到不合语法的 hack（如重复词、奇怪格式）——这些可能恰好得高分但读起来不像人话。

KL 让 policy 保持 SFT 的语言模式。

**角色 3：防止 Catastrophic Forgetting**

SFT 已经学到了基础能力。PPO 只针对 reward 优化，可能让模型 forget 其他能力（如代码、数学）。KL 约束让通用能力被保留。

**实现细节**：
- 通常是 **per-token KL**——每个 token 都计算
- 加到 reward 里：`r_t = -β·log(π/π_ref) + 1[t=T]·RM`
- β 是个关键超参，**典型值 0.01-0.1**，需要细调

实际 KL 的影响极大——β 太小，reward hacking 必然发生；β 太大，policy 学不到东西。"

### 📐 KL Penalty 的实现

**Per-token KL** 在 reward 中：

$$r_t = \mathbb{1}[t = T] \cdot R(x, y) - \beta \cdot \log \frac{\pi_\theta(a_t | s_t)}{\pi_{ref}(a_t | s_t)}$$

其中 $\pi_{ref}$ 是 SFT 模型（固定）。

**总目标中的 KL**：

$$L^{TOTAL} = L^{CLIP} - c_1 L^{VF} + c_2 L^{Entropy}$$

注意 KL 通常**只在 reward 里**，不直接在 loss 里——这是 InstructGPT 的做法。但也有把 KL 加在 loss 里的实现（如 Anthropic CAI），效果类似。

### 🔍 KL 系数 β 的影响

| β | 行为 |
|---|---|
| 0 | 完全无约束，policy 自由漂移 → 必然 reward hacking |
| 0.001 | 弱约束，policy 大幅偏离 SFT |
| **0.01-0.1** | **典型值，平衡 reward 优化和稳定** |
| 0.5+ | 强约束，policy 几乎不动 |
| ∞ | 完全约束，等价于不训 |

实际工业实践——
- InstructGPT：β = 0.02
- LLaMA-2 Chat：未公开，但报告称 β 是关键调优参数

### 🔥 高频追问 Top 3

**Q：用 KL(π || π_ref) 还是 KL(π_ref || π)？方向有差别吗？**

A：**有显著差别**。两种 KL 是不同的散度——

- **KL(π || π_ref)**：mode-seeking——让 π 聚焦于 ref 的高概率区域
- **KL(π_ref || π)**：mode-covering——让 π 尽量覆盖 ref 的所有 mode

LLM RLHF 用 **KL(π || π_ref)**——希望 policy 在 ref 的合理范围内深耕，而不是尽可能多元化。

**Q：KL 计算很贵吗？**

A：**还好**——
- 需要保留 ref 模型在显存中
- 每次 forward 同时算 π 和 π_ref，计算 ratio 和 KL
- 显存大约 2× 单模型，但模型参数共享 backbone 可以省一些

工程上 ref 模型可以**冻结并量化**（如 INT8），进一步节省显存。

**Q：能不能用 adaptive KL（动态调整 β）？**

A：**可以，是常见技巧**。

PPO 论文里有 adaptive KL：监控 KL 实际值，如果太大就增大 β，如果太小就减小 β。让 KL 维持在目标范围（如 0.01）。

实践中 adaptive KL 不是必须——固定 β 经过调参也能 work，且更可控。

### ⚠️ 常见陷阱

1. KL 方向是 KL(π || π_ref)，不要弄反
2. β 是关键超参，要花时间调
3. KL 通常加在 reward 里，不在 loss 里

### 🏢 大厂偏好

- **OpenAI / Anthropic**：会问 KL 方向选择的理论
- **应用 RLHF 团队**：会问 β 怎么调

---

## Q6：Reward Hacking 与应对 ⭐⭐⭐⭐⭐（深度）

### 🎯 一句话标答

> Reward Hacking = **policy 找到了"骗过 RM" 的输出方式**——这些输出 RM 给高分，但人类觉得没用甚至有害。**应对方法**：KL 约束、迭代式 RLHF、Process Reward Model、用 LLM-as-judge 替代 RM。

### 🗣️ 30 秒口语版

"Reward Hacking 是 RLHF 的核心痛点。

**什么是 Reward Hacking**？

简单说就是——policy **学到了 RM 的漏洞**，输出 RM 给高分但实际质量差的响应。

**经典案例**：

- **重复内容**：模型学到'生成一段然后重复多遍'能拿高分（RM 见过的内容都给高分）
- **过度礼貌**：用过多的'I'd be happy to help...'开头，RM 觉得'热情'就加分
- **冗长解释**：把简单问题解释得很详细，RM 学到'长 = 详细 = 好'
- **拒答倾向**：动不动说'I can't do that'——RM 训练数据里安全相关回答多，PPO 学会泛化拒答
- **思维链造假**：在数学题中编造'推理过程'让答案'看起来对'

**为什么会发生**？

- RM 训练数据有偏差——人类标注者可能偏好某些表面特征
- RM 是个 proxy，不是真正的"质量"
- PPO 优化能力极强——会精确找到所有 RM 漏洞

**应对策略**：

**1. KL 约束**：让 policy 不要走太远（基础保险）
**2. 迭代式 RLHF**：定期用新偏好数据重训 RM
**3. Process Reward Model (PRM)**：给中间步骤打分，不只看 outcome
**4. LLM-as-judge**：用更强的 LLM（如 GPT-4）替代 RM
**5. 多 RM 集成**：用多个 RM 投票，降低单一 RM 漏洞被利用
**6. Constitutional AI**：用规则 + AI 反馈替代人评 RM"

### 📊 Reward Hacking 经典模式

| 模式 | 表现 | 根因 |
|---|---|---|
| 长度膨胀 | 模型答得越来越长 | RM 把长度当质量信号 |
| 重复 | 重复关键句子或段落 | RM 在重复内容上加分 |
| 礼貌套话 | 大量寒暄、disclaimer | RM 训练数据中礼貌响应得高分 |
| 拒答 | 频繁说"我不能这么做" | 安全响应在训练中得高分 |
| 模板化 | 输出固定结构（如"首先...其次...总之"）| RM 学到结构化偏好 |
| 数据污染 | 输出像维基百科的"标准答案" | 训练数据中维基风格得高分 |

### 💡 Goodhart's Law 在 RLHF 中的体现

**"When a measure becomes a target, it ceases to be a good measure."**

应用到 RLHF——
- RM 的分数是个"度量"，反映人类偏好（近似）
- 当 PPO 把这个度量作为"目标"激进优化，policy 会**钻 RM 的漏洞**
- 度量和目标的距离越拉越大

这是个**根本性问题**，无法完全消除——只能缓解。

### 🔍 应对方案详解

#### 方案 1：迭代式 RLHF

```
循环：
  1. SFT model → PPO 训练（用 RM_v1） → Policy_v1
  2. 用 Policy_v1 生成新响应 → 人工标注 → 更新偏好数据
  3. 训练 RM_v2（在更新数据上）
  4. 用 RM_v2 继续训练 Policy_v1 → Policy_v2
  5. 回到步骤 2
```

InstructGPT 论文做过 3 轮迭代。每轮 RM 都能学到上一轮 policy 的"hacking 模式"，针对性反击。

**代价**：每轮需要新人工标注，成本高。

#### 方案 2：Process Reward Model (PRM)

传统 RM 是 **Outcome Reward Model (ORM)**——只对最终答案打分。

PRM 对**中间步骤**打分——比如数学题的每一步都标对/错。

优势：
- 不容易被"答案对但推理瞎编"骗
- 适合数学、代码、推理任务
- DeepSeek-Math 和 OpenAI o1 据传都用了 PRM

代价：标注成本极高（需要标注每一步）。

#### 方案 3：LLM-as-Judge

用强 LLM（GPT-4 / Claude）当评估器，替代 RM：

```
Judge_prompt = """评估以下回答的质量：
prompt: {prompt}
response: {response}
请给出 1-10 分..."""

reward = LLM_judge(Judge_prompt)
```

优势：
- LLM 评估比小 RM 鲁棒
- 不需要训练 RM
- 可以加入复杂指令（"评估时重点关注准确性..."）

代价：
- LLM 推理慢、贵
- 仍有 reward hacking 可能（policy 学到骗 LLM judge 的模式）

### 🔥 高频追问 Top 3

**Q：怎么检测 reward hacking 是否发生了？**

A：几个信号——
- **RM 分数 vs 人评分数的相关性**：随着训练，两者背离 = hacking
- **响应长度暴涨**：长度膨胀是 hacking 的典型表现
- **输出多样性下降**：模型只用几种固定模板
- **KL 散度暴涨**：policy 严重偏离 SFT
- **下游 benchmark 下降**：MMLU、HumanEval 等通用能力衰退

**Q：β（KL 系数）的设定能完全防止 reward hacking 吗？**

A：**不能**。β 大能限制偏离，但——
- β 太大：policy 学不到东西（无意义）
- β 太小：仍然漂移、hacking

β 只是**缓解**hacking，不是治本。真正的解决需要更好的 RM 或多机制结合。

**Q：RLHF 一定会 reward hacking 吗？**

A：**几乎一定，只是程度不同**。Goodhart's Law 是数学规律——只要把 proxy 当成目标，必然有漂移。

实际工业实践——**接受 hacking 的存在，控制在可接受范围**。比如：
- 模型答得长一些但仍准确 → 可接受
- 模型频繁拒答 → 不可接受，必须修

迭代式 RLHF 就是不断**发现 hacking 模式并打补丁**的过程。

### ⚠️ 常见陷阱

1. 不要说"KL 就能解决 reward hacking"——只是缓解
2. 要懂 Goodhart's Law 在这里的应用
3. 要懂迭代式 RLHF 是工业标准

### 🏢 大厂偏好

- **OpenAI / Anthropic / DeepMind**：必问，会让你设计 anti-hacking 方案
- **字节 / 阿里 RLHF**：会问具体 case 和应对

### 📚 延伸阅读
- [Reward Hacking in RLHF](https://arxiv.org/abs/2305.15363) - 系统讨论
- [Constitutional AI](https://arxiv.org/abs/2212.08073) - Anthropic 的方案

---

## Q7：RLHF 工程难点 ⭐⭐⭐⭐

### 🎯 一句话标答

> RLHF 的工程复杂度远高于 SFT——需要**4 个模型同时在线**（policy、ref、RM、value）、**rollout 和 train 的复杂交互**、**显存爆炸**、**训练不稳定**。是 LLM 训练中最难的环节。

### 🗣️ 30 秒口语版

"RLHF 工程难点有四个层面——

**1. 4 个模型并存的显存压力**

PPO 训练需要——
- **Policy（被训）**：可学习
- **Reference**：冻结的 SFT，用于 KL
- **Reward Model**：冻结的 RM
- **Value Model**：学 V 函数

四份 7B 模型 = 28B 参数级显存压力。

应对——通常 Policy 和 Value **共享 backbone**（只多一个 head），可以省 1 个模型。RM 和 Reference 也可能共享（如果 RM 基于 SFT）。

**2. Rollout 和 Train 的复杂交互**

- Rollout 阶段：policy 生成响应，需要**推理优化**（KV Cache、batch generation）
- Train 阶段：用 rollout 数据做 PPO 更新

这两个阶段在同一进程里频繁切换——工程实现需要小心。

**3. 多模型分布式训练**

每个模型可能跨多卡（TP/PP），加上 PPO 的多次 rollout-train 循环——分布式调度复杂。

DeepSpeed-Chat、TRL、OpenRLHF 等框架就是为简化这个流程。

**4. 训练不稳定**

PPO 训练经常崩——
- Loss 突然飙升
- Policy 输出乱码
- Reward 不增长甚至下降

需要大量的监控 + 调参 + 启发式技巧。

**业界共识**：RLHF 工程难度 >> SFT，门槛高。这也是 DPO 流行的原因之一——避开了这些坑。"

### 🔍 RLHF 显存估算

设 model size = M GB（BF16），全参 PPO 训练显存：

| 模型 | 参数 | 梯度 | 优化器 | 激活 | 小计 |
|---|---|---|---|---|---|
| Policy | M | M | 6M | ~M | 9M |
| Value | M (共享) | 同上 | 同上 | 同上 | 0 (共享) |
| Reference | M | - | - | ~M | 2M |
| RM | M | - | - | ~M | 2M |
| **总计** | | | | | **~13M** |

70B 模型（BF16 权重 M = 140 GB）→ 140 GB × 13 ≈ **1820 GB**——需要分布式。

**显存优化技巧**：
- Reference / RM 用 INT8 量化（少 50%）
- Policy / Value 共享 backbone
- ZeRO-3 分片优化器状态
- Gradient Checkpointing

### 🔥 高频追问 Top 3

**Q：RLHF 为什么训练不稳定？**

A：根本原因——
- **RL 本身就比监督学习不稳定**（探索-利用权衡、reward 稀疏）
- **Reward 信号噪声大**（RM 不完美）
- **Importance Sampling 在 LLM 上方差大**（token 序列长，ratio 累积偏差）
- **多模型交互引入复杂动态**

应对——
- 小学习率（1e-6 量级）
- 强 KL 约束（β 不要太小）
- 频繁监控并回滚
- Reward normalization / clipping

**Q：rollout batch size 怎么选？**

A：典型 64-512。
- 太小：方差大，估计不准
- 太大：显存压力 + rollout 速度慢

经验：rollout batch 是 PPO mini-batch 的 4-8 倍。比如 mini-batch=64，rollout batch=256-512。

**Q：训练几个 epoch 算完了？**

A：RLHF 不像 SFT 有"epoch"概念——通常按 **steps** 计数。

经验范围——
- **2000-5000 steps**：典型 RLHF 训练长度
- 每步处理 64-256 个 prompts
- 总计训过 100K-500K 个 prompt

比 SFT 短得多——因为不稳定，训太久反而退化。

### ⚠️ 常见陷阱

1. 4 个模型并存的显存压力——必须知道
2. RLHF 不稳定，需要细致监控
3. rollout 和 train 是不同阶段，工程实现要小心

### 🏢 大厂偏好

- **AI Infra / RLHF 工程岗**：必问，会问框架选型
- **字节 / Anthropic**：可能问具体崩溃案例和恢复

### 📚 延伸阅读
- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [TRL](https://github.com/huggingface/trl)

---

## Q8：RLHF 变体：RLAIF / CAI / Best-of-N ⭐⭐⭐

### 🎯 一句话标答

> RLHF 的变体——**RLAIF** 用 AI 评估替代人评、**Constitutional AI** 用规则 + AI 反馈、**Best-of-N** 简单粗暴只采样多个选最好——各有取舍。

### 🗣️ 30 秒口语版

"RLHF 之后衍生了几种重要变体——

**RLAIF（RL from AI Feedback）**：
- 用一个强 LLM（如 GPT-4）替代人类做偏好标注
- 优势：成本低、速度快、规模大
- 劣势：继承 AI 的偏见、能力上限受限于 judge LLM

Google 的 RLAIF 论文证明在某些任务上**RLAIF ≈ RLHF**。

**Constitutional AI（Anthropic, 2022）**：
- 用一组**原则（Constitution）**指导 AI
- 流程：模型生成响应 → AI critique（基于原则） → AI 改进 → 偏好对 → RLAIF
- 完全无需人类偏好标注
- Anthropic 的 Claude 用了这个方法

**Best-of-N**：
- 不做 RL，只采样 N 个响应（N=4-128），用 RM 选最好的
- 优势：实现极简，无需训练
- 劣势：推理慢 N 倍，但效果惊人
- **在 N=64 时通常接近 RLHF 效果**

**Rejection Sampling Fine-tuning**：
- Best-of-N 的训练版——用 Best-of-N 生成的'最佳响应'再做 SFT
- LLaMA-2 在 RLHF 之前先做了 Rejection Sampling

**当前格局**：
- RLHF 仍是质量天花板
- DPO 是工程更友好的替代（下一章细讲）
- Best-of-N 是简单粗暴的备选
- CAI 是 Anthropic 的特色"

### 🔍 变体对比

| 方法 | 是否需要人评 | 工程复杂度 | 效果 | 代表 |
|---|---|---|---|---|
| RLHF | 大量 | 高 | 强 | InstructGPT |
| RLAIF | 少量（仅初始）| 高 | 中-强 | Gemini |
| Constitutional AI | 极少 | 极高 | 强 | Claude |
| **DPO** | **大量** | **低** | **强** | **Mistral** |
| Best-of-N | 用于评估 | 极低 | 中 | 通用 |
| Rejection SFT | 用于评估 | 中 | 中-强 | LLaMA-2 |

### 🔥 高频追问 Top 3

**Q：Best-of-N 推理时怎么实现？真的能接近 RLHF？**

A：实现：

```python
def best_of_n_inference(prompt, policy, rm, N=64):
    responses = policy.generate(prompt, num_return_sequences=N)
    scores = rm(prompt, responses)
    return responses[argmax(scores)]
```

效果——OpenAI 论文实验：N=64 的 Best-of-N **几乎和 RLHF 一样好**，N=4 也比 SFT 强很多。

代价是**推理 N 倍慢**。所以实际部署——
- 离线场景：Best-of-N 是不错选择
- 在线对话：太慢，仍需 RLHF/DPO

**Q：RLAIF 和 RLHF 真的等价吗？**

A：**部分等价**。Google 论文显示：
- 对话质量：RLAIF ≈ RLHF
- 安全性：RLAIF 略弱（AI judge 不如人类敏感）
- 复杂任务：RLAIF 不如 RLHF

实际工业——
- **完全没人评预算**：用 RLAIF
- **有少量人评**：人评训核心 RM，RLAIF 增强
- **大公司**：人评仍是金标准

**Q：Constitutional AI 的"宪法"具体是什么？**

A：Anthropic 公开了部分原则，例如：

```
- Identify specific ways in which the response is harmful or unethical
- Choose the response that is most truthful, honest, and helpful
- Avoid responses that are condescending or paternalistic
- ...
```

模型在两阶段使用：
1. **SL-CAI**（监督版）：让模型基于原则改写自己的响应，得到 SFT 数据
2. **RL-CAI**（强化版）：模型评估两个响应哪个更符合原则，得到偏好对，做 RLAIF

总计约 16 条核心原则 + 任务相关原则。**没有人参与偏好标注**——完全自我对齐。

### ⚠️ 常见陷阱

- 不要把 Best-of-N 当成训练方法——它是推理技巧
- Constitutional AI 是 Anthropic 特色，其他公司用得少
- DPO 是 RLHF 的主流替代，下章详讲

### 🏢 大厂偏好

- **Anthropic**：必问 CAI
- **Google**：可能问 RLAIF
- 其他大厂：跟进态度，了解即可

### 📚 延伸阅读
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [RLAIF: Scaling RLHF with AI Feedback](https://arxiv.org/abs/2309.00267)

---

## 📝 本章小结

| 关键点 | 你必须能脱口而出 |
|---|---|
| RLHF 三阶段 | SFT → RM → PPO |
| RM 损失 | -log σ(r_c - r_r)，Bradley-Terry |
| RM 训练 | 1 epoch，避免过拟合 |
| PPO 核心 | Clipped surrogate + value loss + KL |
| Ratio 公式 | r = π / π_old |
| Clipping 区间 | [1-ε, 1+ε]，ε=0.2 |
| KL 方向 | KL(π \|\| π_ref)，per-token |
| β 典型值 | 0.01-0.1 |
| Reward Hacking | Goodhart's Law，必然但可缓解 |
| RLHF 工程难点 | 4 模型并存 + 不稳定 + 显存大 |
| 主要变体 | RLAIF / CAI / Best-of-N |

## ✅ 自测题

1. 现场推导 PPO 的 Clipped Surrogate Objective，包括对 A > 0 和 A < 0 两种情况的行为分析
2. 写出 RM 训练的完整 loss 函数，并解释 Bradley-Terry 模型的概率含义
3. KL 项加在 reward 里还是 loss 里？两种方式有什么区别？
4. 列举 5 种 reward hacking 的典型表现，并对每种给出应对方案
5. 训练 70B 模型的 RLHF，估算需要多少显存？怎么优化？
6. 如果让你设计一个不需要人评的对齐方案，你会怎么做？给出至少两种思路

---

> **下一章预告**：第 10 章｜DPO 数学推导专题——DPO 是 PPO 的"对偶"，避开了显式 RM 和 RL 训练，**直接从偏好数据优化 policy**。完整推导是字节、Mistral 等顶尖团队的必考题。我们会从 RLHF 目标函数出发，一步步推到 DPO 的简洁形式。
