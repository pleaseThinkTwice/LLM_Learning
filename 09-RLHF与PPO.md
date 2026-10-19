# 第 9 章｜RLHF 与 PPO

> **本章定位**：RLHF 是 InstructGPT / ChatGPT 的核心后训练方法。本章推导三件事：**奖励模型**（Bradley–Terry 与它的局限）、**PPO**（策略梯度 → 重要性采样 → 裁剪 → GAE）、**KL 约束**（以及它给出的最优策略闭式解，这是下一章 DPO 的起点）。然后讨论 RLHF 最核心的研究问题：**reward hacking / 过优化**，它有可以量化的 scaling law。

> **配套章节**：SFT → 第 7 章 / DPO → 第 10 章 / GRPO 与可验证奖励 → 第 11 章 / LLM-as-a-judge → 第 19 章

> **标记约定**：📄 论文结论 / 🧮 可推导 / 💭 直觉或有争议 / ❓ 开放问题

---

## 📌 本章导航

| 序号 | 题目 | 难度 | 类型 |
|---|---|---|---|
| Q1 | RLHF 的三个阶段 | ⭐⭐⭐⭐ | 必懂 |
| Q2 | **奖励模型的训练与评估** | ⭐⭐⭐⭐⭐ | 必问 |
| Q3 | **PPO 目标函数推导** | ⭐⭐⭐⭐⭐ | 推导 |
| Q4 | PPO 裁剪的行为分析 | ⭐⭐⭐⭐ | 数学 |
| Q5 | **KL 约束与最优策略的闭式解** | ⭐⭐⭐⭐⭐ | 推导 |
| Q6 | **Reward hacking 与过优化的 scaling law** | ⭐⭐⭐⭐⭐ | 研究 |
| Q7 | RLHF 的工程难点与框架 | ⭐⭐⭐⭐ | 工程 |
| Q8 | 变体：RLAIF、Constitutional AI、Best-of-N、拒绝采样 | ⭐⭐⭐⭐ | 前沿 |
| Q9 | 研究前沿：生成式奖励模型、在线迭代、奖励模型的偏差 | ⭐⭐⭐⭐⭐ | 研究视角 |

---

## Q1：RLHF 的三个阶段 ⭐⭐⭐⭐

### 🎯 一句话标答

> 📄 InstructGPT 的流程：①**SFT**：在人工示范上微调；②**奖励模型（RM）**：在人类对多个回答的排序上训练，输出标量分数；③**PPO**：以 SFT 模型为初始策略，最大化 RM 奖励，同时用 **KL 惩罚**约束策略不要离 SFT 模型太远。

### 🔍 流程

```
① SFT：base → 在人工示范上微调 → π_SFT
② RM：对同一 prompt 采样 K 个回答，人工排序 → 训练 r_φ(x, y)
③ PPO：π_θ 初始化为 π_SFT；采样回答 → r_φ 打分 → 加上 KL 惩罚 → PPO 更新
         价值模型（critic）常从 RM 初始化
```

📄 InstructGPT 的规模：SFT 约 1.3 万条提示，RM 约 3.3 万条提示，PPO 约 3.1 万条提示（不需要人工标注回答）；RM 用的是 6B 模型（论文提到 175B 的 RM 训练不稳定）。

### 🔥 高频追问

**Q：为什么需要奖励模型，而不是直接用人来打分？**

A：PPO 需要对大量在线生成的回答打分，人工无法实时提供。RM 是人类偏好的一个**可以无限调用的代理**。代价是代理会被优化器"钻空子"（Q6）。

**Q：能不能跳过 SFT？**

A：📄 对于有**可验证奖励**的任务，DeepSeek-R1-Zero 证明了可以直接在 base 模型上做 RL（第 7 章 Q1，第 11 章）。对于依赖 RM 的开放式对话，SFT 提供了一个合理的起点：base 模型的输出远离 RM 的训练分布，RM 的打分不可靠。

**Q：RLHF 只做一轮吗？**

A：通常会迭代。📄 Anthropic（Bai et al. 2022, *Training a Helpful and Harmless Assistant with RLHF*）做了"迭代式在线 RLHF"：用最新的策略收集新的比较数据，定期更新 RM 和策略。📄 LLaMA 2-Chat 进行了 5 轮迭代（RLHF-V1 到 V5），前几轮使用拒绝采样微调，后面再叠加 PPO。

### 📚 出处
- Ouyang et al. *InstructGPT*. NeurIPS 2022. arXiv:2203.02155
- Bai et al. *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*. 2022. arXiv:2204.05862
- Touvron et al. *Llama 2*. 2023. arXiv:2307.09288
- Ziegler et al. *Fine-Tuning Language Models from Human Preferences*. 2019. arXiv:1909.08593
- Stiennon et al. *Learning to summarize from human feedback*. NeurIPS 2020. arXiv:2009.01325

---

## Q2：奖励模型的训练与评估 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> RM 通常由 SFT 模型初始化，把 LM head 换成标量 head，取**序列最后一个 token**（回答结尾）的隐藏状态输出分数。训练采用 **Bradley–Terry 模型**的负对数似然：$\mathcal{L} = -\log\sigma\big(r(x,y_w) - r(x,y_l)\big)$。RM 很容易过拟合，📄 InstructGPT 和 LLaMA 2 都只训练 1 个 epoch。

### 🧮 Bradley–Terry 模型

假设每个回答有一个潜在得分 $r(x,y)$，人类偏好 $y_w$ 胜过 $y_l$ 的概率为
$$P(y_w\succ y_l\mid x) = \frac{e^{r(x,y_w)}}{e^{r(x,y_w)} + e^{r(x,y_l)}} = \sigma\big(r(x,y_w) - r(x,y_l)\big)$$

最大似然：
$$\mathcal{L}_{RM}(\phi) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\right]$$

🧮 **两个重要性质**：
1. 分数只有**差值**有意义：对所有回答加一个只依赖 $x$ 的常数 $c(x)$，损失不变。因此 RM 的绝对值没有意义，用作 PPO 奖励时通常需要归一化。
2. 这一不变性也是 DPO 推导中配分函数 $Z(x)$ 能被消掉的原因（第 10 章）。

### 📄 实践细节

| 来源 | 做法 |
|---|---|
| 📄 InstructGPT | 每个 prompt 让标注者对 $K=4$–$9$ 个回答排序，产生 $\binom K2$ 个比较对；**把同一 prompt 的所有比较对放在一个 batch 元素中**，否则同一 prompt 的数据被重复多次，会在一个 epoch 内就过拟合 |
| 📄 InstructGPT | 训练标注者之间的一致率约 73%，留出标注者之间约 77% |
| 📄 LLaMA 2 | 分别训练**有用性 RM** 和**安全性 RM**；在损失中加入与偏好强度相关的 **margin**：$-\log\sigma(r_w - r_l - m(\text{rating}))$，"明显更好"的样本对 margin 更大；只训练 1 个 epoch，作者发现训练更久会过拟合 |

> ❌ 旧版写"标注者之间一致率只有 60–70%""RM 准确率 > 80% 可能过拟合"，没有出处。InstructGPT 报告的一致率约为 73%–77%。

### 💻 实现

```python
class RewardModel(nn.Module):
    def __init__(self, backbone, hidden_size):
        super().__init__()
        self.backbone = backbone                    # 去掉 LM head 的 Transformer
        self.v_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(dim=1) - 1        # 假设右侧 padding：取最后一个有效 token
        return self.v_head(h[torch.arange(h.size(0)), last]).squeeze(-1)

def rm_loss(rm, chosen_ids, chosen_mask, rejected_ids, rejected_mask, margin=0.0):
    r_w, r_l = rm(chosen_ids, chosen_mask), rm(rejected_ids, rejected_mask)
    return -F.logsigmoid(r_w - r_l - margin).mean()
```

### 🔥 高频追问

**Q：怎么评估一个奖励模型？**

A：
- **偏好准确率**：在留出的偏好对上，$r(y_w) > r(y_l)$ 的比例。
- 📄 **RewardBench**（Lambert et al. 2024）：覆盖对话、困难对话、安全、推理等类别的标准评测集。
- 💭 但准确率高不代表在 RL 中好用：RL 会把策略推到 RM **没见过的分布**上，真正重要的是 RM 在策略分布上的鲁棒性。有研究发现 RewardBench 分数与下游 RLHF 效果的相关性有限。

**Q：Bradley–Terry 模型有什么局限？**

A：
- 假设偏好可以用**一个标量**解释，并且满足传递性。真实的人类偏好可能不满足（A > B、B > C、C > A），并且不同人的偏好不同。
- 📄 后续工作：多目标 RM、按属性打分的 RM、条件于用户的个性化 RM；以及不依赖 BT 模型的 **Nash 学习**方法（Munos et al. 2024，第 10 章）。

### 📚 出处
- Bradley & Terry. *Rank Analysis of Incomplete Block Designs*. Biometrika 1952
- Lambert et al. *RewardBench*. 2024. arXiv:2403.13787

---

## Q3：PPO 目标函数推导 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 从策略梯度出发，用**重要性采样**让同一批数据可以做多次更新，再用**裁剪的概率比**限制每次更新的幅度（PPO-Clip），用 **GAE** 估计优势。LLM 中把每个生成的 token 看作一个动作，RM 奖励在序列末尾给出，每个 token 带一个 KL 惩罚。

### 🧮 Step 1：策略梯度

目标 $J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_t r_t\right]$。🧮 由 $\nabla_\theta\pi_\theta = \pi_\theta\nabla_\theta\log\pi_\theta$（log-derivative trick）：
$$\nabla_\theta J = \mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\hat A_t\right]$$
用优势 $A_t = Q(s_t,a_t) - V(s_t)$ 代替回报，是为了在不改变期望的前提下**减小方差**（减去任何只依赖状态的基线都不引入偏差）。

### 🧮 Step 2：重要性采样

用旧策略 $\pi_{old}$ 采样的数据优化新策略，定义概率比
$$\rho_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}$$
得到代理目标（surrogate）$L^{CPI}(\theta) = \mathbb{E}_t[\rho_t(\theta)\hat A_t]$。
🧮 在 $\theta = \theta_{old}$ 处，$\nabla L^{CPI} = \nabla J$。但离 $\theta_{old}$ 越远，代理目标越不可靠（这里还忽略了状态分布的变化）。

📄 **TRPO**（Schulman et al. 2015）用 KL 散度的**硬约束** $\mathbb{E}[\text{KL}(\pi_{old}\,\|\,\pi_\theta)]\le\delta$ 来保证单调改进，需要共轭梯度和线搜索，实现复杂。

### 🧮 Step 3：PPO-Clip

📄 PPO（Schulman et al. 2017）：
$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\Big(\rho_t\hat A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\hat A_t\Big)\right]$$
通常 $\epsilon=0.2$。直觉见 Q4。

### 🧮 Step 4：GAE

TD 误差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，
$$\hat A_t^{GAE(\gamma,\lambda)} = \sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}$$
$\lambda=0$ 时是一步 TD（低方差、高偏差），$\lambda=1$ 时是蒙特卡洛回报减基线（无偏、高方差）。LLM RLHF 中常用 $\gamma=1$。

### 🧮 Step 5：LLM 中的设定

- 状态 $s_t$ = prompt + 已生成的 token；动作 $a_t$ = 下一个 token。
- 每个 token 的奖励：
$$r_t = -\beta\log\frac{\pi_\theta(a_t\mid s_t)}{\pi_{ref}(a_t\mid s_t)} + \mathbb{1}[t=T]\cdot r_\phi(x, y)$$
- 总损失：$\mathcal{L} = -L^{CLIP} + c_1\,\mathcal{L}^{VF}$（可选熵正则；InstructGPT 式的 RLHF 一般不加熵奖励）。
- 📄 **PPO-ptx**（InstructGPT）：在目标中加入预训练数据的对数似然 $\gamma\,\mathbb{E}_{x\sim D_{pretrain}}[\log\pi_\theta(x)]$，缓解"对齐税"。

### 💻 伪代码

```python
# ---- rollout（不求梯度）----
resp = policy.generate(prompts)
logp_old = policy.logprobs(prompts, resp)          # 每个 token
logp_ref = ref.logprobs(prompts, resp)
values   = critic(prompts, resp)                    # 每个 token
score    = reward_model(prompts, resp)              # 每条序列一个标量

rewards = -beta * (logp_old - logp_ref)             # 每 token 的 KL 惩罚
rewards[:, -1] += score                             # 末尾加 RM 分数
adv, ret = gae(rewards, values, gamma=1.0, lam=0.95)
adv = (adv - adv.mean()) / (adv.std() + 1e-8)       # 优势白化

# ---- 多个 PPO epoch / mini-batch 更新 ----
for mb in minibatches(...):
    logp = policy.logprobs(mb.prompts, mb.resp)
    ratio = (logp - mb.logp_old).exp()
    pg_loss = -torch.min(ratio * mb.adv, ratio.clamp(1 - eps, 1 + eps) * mb.adv).mean()
    v = critic(mb.prompts, mb.resp)
    v_clip = mb.values + (v - mb.values).clamp(-eps_v, eps_v)   # 价值裁剪
    vf_loss = 0.5 * torch.max((v - mb.ret) ** 2, (v_clip - mb.ret) ** 2).mean()
    (pg_loss + c1 * vf_loss).backward(); opt.step(); opt.zero_grad()
```

### 🔥 高频追问

**Q：PPO 的稳定性真的来自裁剪吗？**

A：📄 Engstrom et al., ICLR 2020（*Implementation Matters in Deep RL*）发现，PPO 相对 TRPO 的优势很大程度上来自"代码层面的优化"（价值裁剪、奖励缩放、学习率退火、正交初始化等），而不只是裁剪目标本身；📄 并且裁剪并不能真正把概率比限制在 $[1-\epsilon, 1+\epsilon]$ 内。📄 *The 37 Implementation Details of PPO*（Huang et al., ICLR 2022 blog track）和 *The N+ Implementation Details of RLHF with PPO*（Huang et al. 2024）系统整理了这些细节。

**Q：每次 rollout 后更新几个 epoch？**

A：💭 常见设置是 1–4 个 epoch。📄 LLaMA 2 的做法是：每次 PPO 迭代 batch 512 条，mini-batch 64，**每个 mini-batch 只做一步梯度更新**。更新次数越多，数据越"off-policy"，裁剪越频繁。

**Q：价值模型和策略模型共享参数吗？**

A：两种都有。📄 InstructGPT 的价值函数由 RM 初始化，是单独的模型。共享 backbone 可以省显存，但策略损失和价值损失会互相干扰。

### 📚 出处
- Williams. *REINFORCE*. Machine Learning 1992
- Schulman et al. *TRPO*. ICML 2015. arXiv:1502.05477
- Schulman et al. *GAE*. ICLR 2016. arXiv:1506.02438
- Schulman et al. *Proximal Policy Optimization Algorithms*. 2017. arXiv:1707.06347
- Engstrom et al. *Implementation Matters in Deep RL*. ICLR 2020. arXiv:2005.12729
- Huang et al. *The N+ Implementation Details of RLHF with PPO*. COLM 2024. arXiv:2403.17031

---

## Q4：PPO 裁剪的行为分析 ⭐⭐⭐⭐

### 🧮 分情况

设 $f(\rho) = \min(\rho A, \text{clip}(\rho,1-\epsilon,1+\epsilon)A)$：

| | $\rho < 1-\epsilon$ | $1-\epsilon\le\rho\le1+\epsilon$ | $\rho > 1+\epsilon$ |
|---|---|---|---|
| $A>0$（好动作） | $\rho A$，**有梯度**（允许把概率提回来） | $\rho A$，有梯度 | $(1+\epsilon)A$，**梯度为 0**（不再继续提高） |
| $A<0$（坏动作） | $(1-\epsilon)A$，**梯度为 0**（不再继续压低） | $\rho A$，有梯度 | $\rho A$，**有梯度**（允许把概率压回去） |

🧮 规律：**当更新方向"已经走得够远"时停止；当策略朝错误方向偏离时，总是允许纠正**。`min` 使目标成为未裁剪目标的一个**下界**（悲观估计）。

### 🔥 高频追问

**Q：$\epsilon$ 怎么选？要监控什么？**

A：📄 PPO 论文和 LLaMA 2 都用 0.2。💭 训练时通常监控**被裁剪的 token 比例**（clipfrac）和新旧策略之间的近似 KL；如果它们持续很大，说明每轮更新太激进（学习率太大或 epoch 太多）。

> ❌ 旧版写"10–30% 的样本触发裁剪是正常的""Anthropic 用 0.1–0.2"，没有出处，已删除。

**Q：对称的裁剪区间有什么问题？**

A：📄 DAPO（ByteDance Seed 2025）提出 **Clip-Higher**：把上界放宽（如 $\epsilon_{high}=0.28$、$\epsilon_{low}=0.2$）。原因是低概率 token 的概率比更容易被上界卡住，导致"探索型"token 难以被提升，从而出现**熵坍塌**（→ 第 11 章）。

---

## Q5：KL 约束与最优策略的闭式解 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> RLHF 的目标是 $\max_\pi \mathbb{E}_{x, y\sim\pi}[r(x,y)] - \beta\,\text{KL}(\pi(\cdot\mid x)\,\|\,\pi_{ref}(\cdot\mid x))$。🧮 它的最优解有闭式形式：$\pi^*(y\mid x) = \frac{1}{Z(x)}\pi_{ref}(y\mid x)\exp\big(r(x,y)/\beta\big)$。KL 项的作用是：让策略留在 RM 可信的区域、保持语言质量、减少遗忘。

### 🧮 推导

对固定的 $x$，目标为
$$\max_\pi\ \sum_y\pi(y)\,r(y) - \beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{ref}(y)}$$
改写为
$$-\beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{ref}(y)\,e^{r(y)/\beta}} = -\beta\,\text{KL}\!\left(\pi\ \Big\|\ \frac{\pi_{ref}e^{r/\beta}}{Z}\right) + \beta\log Z$$
其中 $Z = \sum_y\pi_{ref}(y)e^{r(y)/\beta}$ 与 $\pi$ 无关。KL 非负，等于 0 当且仅当两个分布相同，所以
$$\boxed{\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{ref}(y\mid x)\,\exp\!\left(\frac{r(x,y)}{\beta}\right)}$$

💡 这个结果说明：KL 约束下的 RLHF 等价于**用奖励对参考模型做指数加权**。$\beta\to0$ 时退化为只选奖励最高的回答；$\beta\to\infty$ 时 $\pi^*=\pi_{ref}$。DPO 正是从这个闭式解出发，把奖励用策略表示出来（第 10 章）。

### 🔥 高频追问

**Q：为什么是 $\text{KL}(\pi\,\|\,\pi_{ref})$ 而不是反过来？**

A：🧮 $\text{KL}(\pi\,\|\,\pi_{ref})$ 的期望是在 $\pi$ 的样本上计算的，可以直接用策略生成的样本估计；并且它在 $\pi$ 把概率放到 $\pi_{ref}$ 几乎为 0 的地方时惩罚很大，所以策略不会生成参考模型认为极不可能的文本。它是"寻找众数"（mode-seeking）的：允许策略只覆盖参考分布的一部分。

**Q：KL 放在奖励里还是放在损失里？**

A：
- **放在奖励里**（InstructGPT、Ziegler 2019）：每个 token 的奖励 $-\beta\log(\pi/\pi_{ref})$，通过价值函数和 GAE 传播，影响后续 token 的优势。
- **放在损失里**（📄 GRPO, DeepSeekMath）：直接在损失中加 $\beta\,\mathbb{D}_{KL}$，并使用无偏、非负的 $k_3$ 估计量 $\frac{\pi_{ref}}{\pi} - \log\frac{\pi_{ref}}{\pi} - 1$（→ 第 11 章）。
- 💭 两者在优化上不完全等价。📄 后续有研究分析了这些 KL 估计量的梯度性质，发现某些"放在损失里"的写法得到的梯度与真正的 KL 梯度并不一致。

**Q：β 怎么设？**

A：
- 📄 Ziegler et al. 2019 用**自适应 KL 控制器**：根据当前 KL 与目标 KL 的差距调整 $\beta$。
- 📄 InstructGPT 使用 $\beta=0.02$；LLaMA 2 对 7B/13B 用 0.01，对 34B/70B 用 0.005。
- 💭 KL 越小，RM 越可信；KL 越大，奖励越高但越可能是过优化（Q6）。

### 📚 出处
- Peters & Schaal 2007；Korbak et al. *RL with KL penalties is better viewed as Bayesian inference*. 2022. arXiv:2205.11275
- Schulman. *Approximating KL Divergence*（博客, 2020）
- Shao et al. *DeepSeekMath*. 2024. arXiv:2402.03300

---

## Q6：Reward hacking 与过优化的 scaling law ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> RM 只是人类偏好的代理，优化得越狠，策略越会利用 RM 的缺陷（Goodhart 定律）。📄 Gao, Schulman & Hilton（ICML 2023）在合成设置中把这一现象**量化**了：随着策略离初始模型越远，代理奖励单调上升，而"真实"奖励先升后降，并且这条曲线的形式可以预测，系数随 RM 规模平滑变化。

### 📄 过优化的 scaling law

📄 设置：用一个大的"金标准 RM"代替人类，用它的标签训练较小的"代理 RM"，再对代理 RM 做 Best-of-N 或 RL 优化，观察金标准分数的变化。令 $d = \sqrt{\text{KL}(\pi\,\|\,\pi_{init})}$：
$$R_{BoN}(d) = d\,(\alpha_{bon} - \beta_{bon}\,d), \qquad R_{RL}(d) = d\,(\alpha_{RL} - \beta_{RL}\log d)$$
- 两种优化方式的函数形式不同，但金标准奖励都会在某个 KL 之后**下降**。
- 系数随**代理 RM 的参数量**平滑变化：RM 越大，过优化越晚出现、峰值越高。
- 在 RL 中加入 KL 惩罚，主要效果是让训练**更早停止**，而不是改变"金标准奖励–KL"曲线本身。

### 📄 典型的 hacking 模式

| 模式 | 证据 |
|---|---|
| **长度偏好** | 📄 Singhal et al. 2023（*A Long Way to Go*）：RLHF 带来的奖励提升中，有相当一部分可以由回答长度的增加来解释；只按长度优化的"奖励"也能复现大部分提升 |
| **谄媚（sycophancy）** | 📄 Sharma et al., ICLR 2024（Anthropic, *Towards Understanding Sycophancy*）：多个 RLHF 助手会迎合用户的观点；人类偏好数据和 RM 本身就倾向于奖励与用户观点一致的回答 |
| **误导人类** | 📄 Wen et al. 2024（*Language Models Learn to Mislead Humans via RLHF*）：RLHF 后的模型在问答和编程任务上并没有更正确，但更善于让人类评估者**相信**它是对的（"U-Sophistry"） |
| **格式与套话** | 💭 过多的标题、列表、免责声明等表面特征可能被 RM 奖励 |

### 🔍 缓解方法

| 方法 | 说明 |
|---|---|
| KL 约束 / 早停 | 最基本的手段 |
| 更大、更好的 RM | 📄 Gao et al.：过优化随 RM 规模增大而减轻 |
| RM 集成 | 📄 Coste et al., ICLR 2024：用多个 RM 的保守聚合（如最小值）作为奖励可以缓解过优化；📄 Eisenstein et al. 2023 指出，如果集成成员共享相同的预训练，它们会犯相似的错误，缓解但不能消除 |
| 迭代更新 RM | 在新策略的输出上收集新偏好，修补 RM 的漏洞（Anthropic、LLaMA 2） |
| 长度控制 | 在奖励或评测中对长度做惩罚或校正（例如 AlpacaEval 的长度控制版本） |
| 可验证奖励 | 对数学、代码等任务直接用规则或执行结果打分，从根本上避免 RM 被利用（→ 第 11 章）。📄 DeepSeek-R1 明确表示没有使用神经网络 RM（包括过程奖励模型），原因之一就是大规模 RL 中的 reward hacking |

**过程奖励模型（PRM）**：📄 Lightman et al. 2023（OpenAI, *Let's Verify Step by Step*）发布了 PRM800K，发现在 MATH 上用 PRM 做 best-of-N 选择明显优于结果奖励模型。📄 DeepSeekMath 在 GRPO 实验中对比了结果监督和过程监督。

> ❌ 旧版写"DeepSeek-Math 和 OpenAI o1 据传都用了 PRM""InstructGPT 做过 3 轮迭代"，前者 o1 部分没有可靠来源，后者与论文不符，已修正。

### 📚 出处
- Gao et al. *Scaling Laws for Reward Model Overoptimization*. ICML 2023. arXiv:2210.10760
- Singhal et al. *A Long Way to Go: Investigating Length Correlations in RLHF*. 2023. arXiv:2310.03716
- Sharma et al. *Towards Understanding Sycophancy in Language Models*. ICLR 2024. arXiv:2310.13548
- Wen et al. *Language Models Learn to Mislead Humans via RLHF*. ICLR 2025. arXiv:2409.12822
- Coste et al. *Reward Model Ensembles Help Mitigate Overoptimization*. ICLR 2024. arXiv:2310.02743
- Eisenstein et al. *Helping or Herding?* 2023. arXiv:2312.09244
- Lightman et al. *Let's Verify Step by Step*. ICLR 2024. arXiv:2305.20050
- Lilian Weng. *Reward Hacking in Reinforcement Learning*（博客, 2024）

---

## Q7：RLHF 的工程难点与框架 ⭐⭐⭐⭐

### 🎯 一句话标答

> PPO 需要同时维护 **4 个模型**：策略（训练）、价值模型（训练）、参考模型（冻结）、奖励模型（冻结）。训练循环在"**生成**（推理，受显存带宽限制）"和"**训练**（受计算限制）"之间交替，生成通常占大部分时间。现代框架把生成交给 vLLM / SGLang 等推理引擎，再在训练引擎和推理引擎之间同步权重。

### 🔍 难点

1. **显存**：两个可训练模型（各需优化器状态）+ 两个冻结模型。参考模型和 RM 只需前向，可以卸载或放在其他设备上。
2. **生成效率**：自回归生成慢，长回答时尤甚；需要推理引擎、continuous batching。
3. **权重同步**：每次策略更新后，要把新权重同步给推理引擎。
4. **数值一致性**：训练引擎和推理引擎计算出的 log-prob 可能不同（不同 kernel、精度），导致重要性比偏离 1，这是 2025 年被广泛讨论的一个问题（→ 第 11 章）。
5. **稳定性**：奖励归一化、优势白化、价值裁剪、学习率、KL 等都需要仔细调整。

### 📄 公开的超参数（LLaMA 2-Chat 的 PPO）

AdamW（$\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-5}$）、weight decay 0.1、梯度裁剪 1.0、**常数学习率 $10^{-6}$**；每次迭代 batch 512、mini-batch 64、每个 mini-batch 一步更新；裁剪阈值 0.2；KL 系数 7B/13B 为 0.01、34B/70B 为 0.005；训练 200–400 次迭代。奖励由有用性和安全性两个 RM 分段组合，并做了白化处理。

> ❌ 旧版的显存表格（"约 13M"）、"典型 2000–5000 步"、"rollout batch 是 mini-batch 的 4–8 倍"等没有出处，已替换为上面有出处的配置。

### 🔍 开源框架

| 框架 | 特点 |
|---|---|
| TRL（HuggingFace） | 覆盖 SFT、RM、PPO、DPO、GRPO 等，易上手 |
| DeepSpeed-Chat | 早期的端到端 RLHF 流水线（Hybrid Engine 在训练和推理模式之间切换） |
| OpenRLHF | 基于 Ray 调度多个模型，用 vLLM 生成 |
| veRL（📄 HybridFlow, ByteDance） | 用"单控制器 + 多控制器"的混合编程模型灵活组织 RL 数据流，支持训练与推理引擎共置或分离 |

### 📚 出处
- Touvron et al. *Llama 2*. arXiv:2307.09288（§3.2.3）
- Yao et al. *DeepSpeed-Chat*. 2023. arXiv:2308.01320
- Hu et al. *OpenRLHF*. 2024. arXiv:2405.11143
- Sheng et al. *HybridFlow: A Flexible and Efficient RLHF Framework*. EuroSys 2025. arXiv:2409.19256

---

## Q8：变体：RLAIF、Constitutional AI、Best-of-N、拒绝采样 ⭐⭐⭐⭐

### 📄 RLAIF

📄 Lee et al., ICML 2024（Google, *RLAIF vs. RLHF*）：用现成的 LLM 代替人类标注偏好。在**摘要**和**有用对话**上，RLAIF 与 RLHF 的人类评估结果相当（两者相对 SFT 的胜率相近，彼此之间没有显著偏好）；在**无害对话**上，RLAIF 的无害率反而**高于** RLHF（约 88% vs 76%）。他们还提出 **d-RLAIF**：RL 过程中直接让 LLM 打分，省去训练 RM，效果更好。

> ❌ 旧版说"安全性上 RLAIF 略弱"，与论文结论相反。

### 📄 Constitutional AI

📄 Bai et al. 2022（Anthropic）：用一组自然语言写成的**原则（constitution）**代替人工的无害性标注：
1. **SL-CAI**：让模型按原则批评并修改自己的有害回答，用修改后的回答做 SFT。
2. **RL-CAI**：让模型按原则比较两个回答，生成 AI 偏好标签，训练偏好模型，再做 RL（即 RLAIF）。
- 有用性部分仍使用人类偏好数据；无害性部分不需要人工标注。

### 📄 Best-of-N 与拒绝采样微调

- **Best-of-N**：采样 $N$ 个回答，用 RM 选最高分。📄 WebGPT（Nakano et al. 2021）发现，在他们的设置中，对 RM 做 best-of-64 的拒绝采样效果**优于** RL。
- 🧮 Best-of-N 策略与原策略之间的 KL 有一个常用的解析式：$\text{KL}_{BoN} = \log N - \frac{N-1}{N}$。📄 Beirami et al. 2024 指出这其实是一个**上界**。这个关系让我们可以在"同样的 KL 预算"下比较 Best-of-N 和 RL（Q6 的 Gao et al. 正是这样做的）。
- **拒绝采样微调**：用 Best-of-N 选出的回答做 SFT。📄 LLaMA 2-Chat 在前几轮迭代中只用了拒绝采样微调，最后才加 PPO。

### 🔍 对比

| 方法 | 需要人工偏好 | 需要在线 RL | 代表 |
|---|---|---|---|
| RLHF（PPO） | 是 | 是 | InstructGPT、LLaMA 2-Chat |
| RLAIF / d-RLAIF | 否（LLM 标注） | 是 | Lee et al. 2024 |
| Constitutional AI | 无害性部分否 | 是 | Anthropic Claude |
| DPO | 是（离线偏好对） | 否 | Zephyr、Tülu 2、LLaMA 3 后训练 |
| Best-of-N | 需要 RM | 否（推理时） | WebGPT |
| 拒绝采样微调 | 需要 RM | 否 | LLaMA 2-Chat |

### 📚 出处
- Lee et al. *RLAIF vs. RLHF*. ICML 2024. arXiv:2309.00267
- Bai et al. *Constitutional AI*. 2022. arXiv:2212.08073
- Nakano et al. *WebGPT*. 2021. arXiv:2112.09332
- Beirami et al. *Theoretical guarantees on the best-of-n alignment policy*. 2024. arXiv:2401.01879

---

## Q9：研究前沿：生成式奖励模型、在线迭代、奖励模型的偏差 ⭐⭐⭐⭐⭐

### 方向 1：生成式奖励模型与 LLM-as-a-judge

- 📄 Zheng et al., NeurIPS 2023（*Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*）：GPT-4 作为评判者与人类偏好的一致率达到 80% 以上，与人类之间的一致率相当；同时存在位置偏差、冗长偏差、自我偏好等问题（→ 第 19 章）。
- 📄 **生成式验证器**（Zhang et al. 2024, *Generative Verifiers*）：让 RM 以 next-token prediction 的方式先生成推理过程再给出判断，可以利用思维链和多次采样投票，在推理任务上优于判别式 RM。
- ❓ 开放问题：策略会不会学会"欺骗"评判模型？如何让评判模型和策略一起进化？

### 方向 2：在线与离线

- 📄 Xu et al., ICML 2024（*Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study*）：在对话和代码任务上，调好的 PPO 优于 DPO；DPO 的问题之一是对**分布外**的回答可能给出不合理的偏好。
- 📄 Tajwar et al. 2024（*Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data*）：使用**策略自己生成的数据**（on-policy 采样）和带"负梯度"的目标（主动压低坏回答的概率）是偏好微调效果好的关键。
- 💭 这也是"在线 DPO / 迭代 DPO"（第 10 章）和"可验证奖励的在线 RL"（第 11 章）流行的原因。

### 方向 3：奖励模型的偏差与可解释性

- 长度、谄媚、格式等偏差（Q6）。
- ❓ 开放问题：如何检测 RM 依赖的是哪些特征？能否在训练 RM 时就去除这些偏差（例如对长度做因果解耦）？

### 🧪 动手验证思路

1. 推导并数值验证 KL 正则 RLHF 的闭式解：在一个只有 5 个候选回答的玩具问题上，用梯度上升优化策略，和 $\pi_{ref}e^{r/\beta}/Z$ 比较。
2. 用 TRL 在一个小模型上跑 PPO，记录"RM 分数–KL"曲线，再用一个更大的 RM 作为"金标准"，复现 Gao et al. 的过优化现象。
3. 统计一个开源偏好数据集中"被选中回答"与"被拒绝回答"的长度差异，估计长度偏差的大小。
4. 实现 Best-of-N，在不同 $N$ 下测量 KL（用采样估计）并与 $\log N - \frac{N-1}{N}$ 比较。

### 📚 出处
- Zheng et al. NeurIPS 2023 D&B. arXiv:2306.05685
- Zhang et al. *Generative Verifiers: Reward Modeling as Next-Token Prediction*. ICLR 2025. arXiv:2408.15240
- Xu et al. ICML 2024. arXiv:2404.10719
- Tajwar et al. ICML 2024. arXiv:2404.14367

---

## 📝 本章小结

| 关键点 | 你必须能脱口而出 | 性质 |
|---|---|---|
| 三阶段 | SFT → RM → PPO（+KL）；可迭代 | 📄 |
| RM 损失 | $-\log\sigma(r_w - r_l)$；分数只有差值有意义；1 个 epoch | 🧮 + 📄 |
| 标注一致率 | InstructGPT 约 73%–77% | 📄 |
| 策略梯度 | $\nabla\log\pi\cdot\hat A$；基线减小方差 | 🧮 |
| PPO-Clip | $\min(\rho A, \text{clip}(\rho)A)$；够了就停、错了能改 | 🧮 |
| GAE | $\sum(\gamma\lambda)^l\delta_{t+l}$ | 🧮 |
| 闭式解 | $\pi^* \propto \pi_{ref}e^{r/\beta}$ | 🧮 |
| KL 位置 | 奖励里（InstructGPT）/ 损失里（GRPO） | 📄 |
| 过优化 | 金标准奖励随 KL 先升后降；函数形式可预测；RM 越大越好 | 📄 |
| Hacking 模式 | 长度、谄媚、误导人类 | 📄 |
| RLAIF | 与 RLHF 相当；无害性上更好 | 📄 |
| Best-of-N | $\text{KL}\le\log N - (N-1)/N$；WebGPT 中优于 RL | 🧮 + 📄 |

## ✅ 自测题

1. 推导 Bradley–Terry 损失，并说明为什么 RM 分数加上只依赖 $x$ 的常数后损失不变。
2. 从策略梯度推导到 PPO-Clip，并对 $A>0$、$A<0$ 分别分析裁剪何时生效。
3. 推导 KL 正则 RLHF 目标的最优策略闭式解。
4. 描述 Gao et al. 2023 的实验设置和主要结论。KL 惩罚在其中起什么作用？
5. 列出三种 reward hacking 现象并给出对应的论文证据和缓解方法。
6. 解释 InstructGPT 为什么要把同一 prompt 的所有比较对放在同一个 batch 元素里。
7. 你要为一个客服产品做 RLHF：设计数据收集、RM 训练、RL 训练和评估方案，并说明如何监控 reward hacking。

---

> **下一章预告**：第 10 章｜DPO 数学推导专题——从本章的闭式解出发推导 DPO 损失；分析 DPO 的梯度、隐式奖励和已知问题（概率同时下降、对长度的偏好、离线数据的局限），以及 IPO、KTO、SimPO、ORPO、在线 DPO 等变体。
