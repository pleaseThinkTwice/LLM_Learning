# 第 11 章｜GRPO 与推理模型训练

> **本章定位**：2024–2025 年最重要的后训练进展是**用可验证奖励做大规模强化学习（RLVR）**来训练推理模型。本章推导 GRPO（并指出它的几个偏差）、梳理 DeepSeek-R1 / R1-Zero 的训练流程，然后进入当前最活跃的研究争论：**"顿悟时刻"是 RL 涌现的还是 base 模型本来就有的？RL 到底是扩展了推理能力，还是只是提高了采样效率？** 这些问题都还没有定论，正适合作为研究的切入点。

> **配套章节**：PPO / KL / reward hacking → 第 9 章 / DPO → 第 10 章 / SFT 与蒸馏 → 第 7 章 / 推理时扩展与解码 → 第 14 章 / Agent 的多轮 RL → 第 18 章

> **标记约定**：📄 论文结论 / 🧮 可推导 / 💭 直觉或有争议 / ❓ 开放问题

---

## 📌 本章导航

| 序号 | 题目 | 难度 | 类型 |
|---|---|---|---|
| Q1 | GRPO 是什么？为什么推理模型用它 | ⭐⭐⭐⭐ | 概念 |
| Q2 | **GRPO 目标函数推导** | ⭐⭐⭐⭐⭐ | 推导 |
| Q3 | **为什么不用 critic 也能工作？组基线的理论** | ⭐⭐⭐⭐⭐ | 理论 |
| Q4 | **GRPO 的偏差与修正：Dr. GRPO、DAPO、GSPO** | ⭐⭐⭐⭐⭐ | 前沿 |
| Q5 | 结果奖励 vs 过程奖励 | ⭐⭐⭐⭐ | 概念 |
| Q6 | **DeepSeek-R1 的训练流程** | ⭐⭐⭐⭐⭐ | 必问 |
| Q7 | R1-Zero 与"顿悟时刻" | ⭐⭐⭐⭐ | 研究 |
| Q8 | **RL 真的教会了推理吗？** | ⭐⭐⭐⭐⭐ | 研究争论 |
| Q9 | 工程问题与其他方向 | ⭐⭐⭐⭐ | 工程 + 前沿 |

---

## Q1：GRPO 是什么？为什么推理模型用它 ⭐⭐⭐⭐

### 🎯 一句话标答

> 📄 GRPO（Group Relative Policy Optimization，DeepSeekMath, Shao et al. 2024）是去掉价值模型（critic）的 PPO 变体：对同一个问题采样**一组** $G$ 个回答，用组内奖励的**均值和标准差**归一化得到每个回答的优势，这个优势被该回答的所有 token 共享；KL 惩罚直接放在损失中。它特别适合**结果奖励**（例如答案对错）的场景，被 DeepSeek-R1 采用后成为推理模型 RL 的常用基线。

### 🗣️ 30 秒口语版

"PPO 需要一个和策略差不多大的价值模型来估计每个 token 的基线，显存和工程代价都大，而且在只有最终结果奖励、回答又很长的情况下，价值模型本身很难训好。

GRPO 的做法是：对每个问题采样一组回答，比如 16 或 64 个，用奖励模型或规则给每个回答打分，然后用'减去组内均值、除以组内标准差'作为这个回答的优势，这个回答里所有 token 用同一个优势值，再套用 PPO 的裁剪目标，加一个 KL 惩罚。

对数学、代码这类有可验证答案的任务，奖励就是答案对不对，GRPO 非常自然：同一道题的回答之间互相比较，对的被提高，错的被压低。

不过 2025 年的研究也发现原始 GRPO 有几个偏差，比如按回答长度归一化会让错误的长回答受到的惩罚更小，按标准差归一化会让简单题和难题的权重失衡，所以后来有 Dr. GRPO、DAPO、GSPO 等修正版本。"

### 🔍 PPO vs GRPO

| | PPO（RLHF） | GRPO |
|---|---|---|
| 基线 | 价值模型 $V_\phi(s_t)$，每个 token 一个 | 同一问题的组内平均奖励，每个回答一个 |
| 优势 | GAE，按 token | $(r_i - \bar r)/\sigma_r$，按回答共享 |
| 模型 | 策略、价值、参考、RM | 策略、参考（+ RM 或规则） |
| KL | 通常放在每 token 奖励里 | 放在损失里， $k_3$ 估计量 |
| 采样 | 每个 prompt 通常 1 个回答 | 每个 prompt $G$ 个回答 |

> ❌ 旧版写"Kimi-K0、Qwen-QwQ、OpenAI o1 据传都用了类似方法"。"Kimi-K0"并不存在（Kimi 的推理模型是 k1.5，使用的是另一种策略优化方法）；o1 的训练方法没有公开。已删除。

### 📚 出处
- Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. 2024. arXiv:2402.03300
- DeepSeek-AI. *DeepSeek-R1*. 2025. arXiv:2501.12948；Nature 版本：Guo et al., *DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning*, Nature 645, 633–638 (2025)

---

## Q2：GRPO 目标函数推导 ⭐⭐⭐⭐⭐

### 🧮 目标函数（DeepSeekMath 原文形式）

对问题 $q$，从 $\pi_{\theta_{old}}$ 采样 $\lbrace o_1,\ldots,o_G\rbrace$：

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac1G\sum_{i=1}^G\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big\{\min\big(\rho_{i,t}\hat A_{i,t},\ \text{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_{i,t}\big) - \beta\,\mathbb{D}_{KL}[\pi_\theta\,\|\,\pi_{ref}]\Big\}\right]$$

- 概率比： $\rho_{i,t} = \dfrac{\pi_\theta(o_{i,t}\mid q, o_{i,\lt t})}{\pi_{\theta_{old}}(o_{i,t}\mid q, o_{i,\lt t})}$
- **结果监督**下的优势（同一回答的所有 token 共享）：

$$\hat A_{i,t} = \tilde r_i = \frac{r_i - \text{mean}(\mathbf r)}{\text{std}(\mathbf r)}$$

- **过程监督**下：每个推理步骤有奖励，先对所有步骤奖励做组内归一化，然后 $\hat A_{i,t}$ 是 token $t$ **之后**所有步骤的归一化奖励之和。
- KL 的 $k_3$ 估计量（按 token）：

$$\mathbb{D}_{KL} = \frac{\pi_{ref}(o_{i,t}\mid\cdot)}{\pi_\theta(o_{i,t}\mid\cdot)} - \log\frac{\pi_{ref}(o_{i,t}\mid\cdot)}{\pi_\theta(o_{i,t}\mid\cdot)} - 1$$

### 🧮 为什么用 $k_3$？

令 $u = \pi_{ref}(a)/\pi_\theta(a)$， $a\sim\pi_\theta$：
- $k_1 = -\log u$： $\mathbb{E}[k_1] = \text{KL}(\pi_\theta\thinspace \Vert \thinspace \pi_{ref})$，无偏，但单个样本可能为负，方差大。
- $k_3 = (u - 1) - \log u$：因为 $\mathbb{E}_ {a\sim\pi_\theta}[u] = 1$，所以 $\mathbb{E}[k_3] = \mathbb{E}[k_1]$，**同样无偏**；并且由 $\log u \le u - 1$，**每个样本都非负**，通常方差更小。
- 📄 出处：Schulman 的博客 *Approximating KL Divergence*（2020）。
- 💭 注意：把 $k_3$ 作为**损失项**求梯度，得到的梯度并不等于真正 KL 的梯度；2025 年有多篇工作分析了不同 KL 估计量在"作为奖励"和"作为损失"时的梯度性质。

### 💻 伪代码

```python
# 对一个 batch 的问题，每个问题采样 G 个回答
resp = policy.generate(prompts, n=G)                       # (B*G) 个回答
r = reward_fn(prompts, resp).view(B, G)                    # 规则或 RM
adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
adv = adv.view(-1, 1)                                      # 每个回答一个，广播到所有 token

logp_old = policy.logprobs(prompts, resp).detach()        # (B*G, T)
logp_ref = ref.logprobs(prompts, resp).detach()
mask = response_mask(resp)                                  # (B*G, T)

for _ in range(num_updates):
    logp = policy.logprobs(prompts, resp)
    ratio = (logp - logp_old).exp()
    pg = -torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
    kl = (logp_ref - logp).exp() - (logp_ref - logp) - 1
    per_tok = pg + beta * kl
    # 原始 GRPO：先对每个回答的 token 求平均，再对回答求平均（Q4 讨论其偏差）
    loss = ((per_tok * mask).sum(1) / mask.sum(1)).mean()
    loss.backward(); opt.step(); opt.zero_grad()
```

### 🔥 高频追问

**Q：组内奖励全相同（全对或全错）会怎样？**

A：🧮 优势全为 0（分母加了 $\epsilon$），这一组**不产生任何策略梯度**，只剩 KL 项。训练后期大量题目要么全对要么全错，有效 batch 会显著缩小。📄 DAPO 的"动态采样"就是针对这一点：过采样并丢弃这类组，保持每个 batch 中有梯度的样本数量（Q4）。

**Q： $G$ 取多少？**

A：📄 DeepSeekMath 中 GRPO 每个问题采样 64 个回答。DeepSeek-R1 的具体超参数见其 Nature 版本的补充材料。💭 $G$ 越大，基线越准，但相同算力下能覆盖的问题数越少。

> ❌ 旧版把 DeepSeekMath 的 $G=64$ 说成了 DeepSeek-R1 的配置。

---

## Q3：为什么不用 critic 也能工作？组基线的理论 ⭐⭐⭐⭐⭐

### 🧮 基线不改变梯度期望

对任何不依赖动作的 $b(s)$：

$$\mathbb{E}_{a\sim\pi}\left[\nabla_\theta\log\pi_\theta(a\mid s)\,b(s)\right] = b(s)\sum_a\nabla_\theta\pi_\theta(a\mid s) = b(s)\,\nabla_\theta\sum_a\pi_\theta(a\mid s) = 0$$

所以减去基线不改变期望，只改变方差。

### 🧮 组均值是 $V(q)$ 的蒙特卡洛估计

在结果奖励的设定下，整段回答可以看作一个"动作"， $V_{\pi_{old}}(q) = \mathbb{E}_ {o\sim\pi_{old}}[r(q,o)]$，组均值 $\bar r = \frac1G\sum_j r_j$ 是它的蒙特卡洛估计。LLM 的特殊之处在于**同一个问题可以低成本地重复采样**，所以可以直接用样本均值代替学习出来的价值函数。

### 🧮 但组均值作为基线是有偏的

$\bar r$ 包含了 $r_i$ 本身，因此**基线依赖于动作 $o_i$**，严格来说违反了上面"基线不依赖动作"的条件。
- 🧮 **留一法（Leave-One-Out）**基线 $b_i = \frac{1}{G-1}\sum_{j\ne i}r_j$ 不依赖 $o_i$，是无偏的。并且 $r_i - b_i = \frac{G}{G-1}(r_i - \bar r)$，与"减组均值"只差一个常数因子。
- 📄 这个想法并不新：Kool et al. 2019（*Buy 4 REINFORCE Samples, Get a Baseline for Free!*）在组合优化中提出了这种多样本基线；📄 Ahmadian et al., ACL 2024（Cohere, *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs*）提出 **RLOO**，在 RLHF 中用留一法 REINFORCE 取得了比 PPO 更好的效果，并指出 LLM 的 RLHF 中 PPO 的很多复杂组件并非必要。
- 🧮 **除以标准差**则会改变不同问题之间的相对权重（Q4）。

> ❌ 旧版写"之前没人这么做，因为传统 RL 难以对同一状态多次采样""组内均值是无偏的"，两者都不准确。

### 🔍 critic 真的没用吗？

📄 **VAPO**（ByteDance Seed 2025）是基于价值模型的方法，通过价值预训练、解耦的 GAE（策略和价值使用不同的 $\lambda$）、长度自适应的 GAE 等技术，在 AIME 2024 上超过了 DAPO 等无 critic 方法。💭 所以"去掉 critic"是一个工程上的权衡（简单、省显存），而不是理论上更优；在长思维链中做精细的 token 级信用分配，仍然是价值方法的潜在优势。

> ❌ 旧版写"GRPO + critic 实践中没有显著收益"，与 VAPO 等工作的结论不符。

### 📚 出处
- Kool et al. ICLR 2019 Workshop（DeepRLStructPred）
- Ahmadian et al. ACL 2024. arXiv:2402.14740
- Yue et al. *VAPO*. 2025. arXiv:2504.05118

---

## Q4：GRPO 的偏差与修正：Dr. GRPO、DAPO、GSPO ⭐⭐⭐⭐⭐

### 偏差 1：按回答长度归一化 → 偏好更长的错误回答

🧮 原始 GRPO 对每个回答先按 token 数 $|o_i|$ 取平均。对于优势为负的（错误）回答，每个 token 分到的惩罚是 $\hat A_i/|o_i|$，**回答越长，每个 token 受到的惩罚越小**；对于正确回答，越短的回答每个 token 得到的奖励越大。结果是：正确回答倾向于变短，**错误回答倾向于变长**。

📄 **Dr. GRPO**（Liu et al. 2025, *Understanding R1-Zero-Like Training: A Critical Perspective*）指出了这个偏差，以及下一个偏差，并提出去掉这两项归一化。他们报告：修正后在保持推理性能的同时提高了 token 效率；用一个 7B base 模型的最小化配方在 AIME 2024 上达到 43.3%。

### 偏差 2：按标准差归一化 → 问题难度偏差

🧮 对于几乎全对或几乎全错的问题，组内标准差很小，除以它会把优势放大；奖励差异大的问题反而被缩小。这使得**过于简单和过于困难的问题获得了更大的权重**。Dr. GRPO 去掉了标准差归一化（只减均值）。

### DAPO：四项改进

📄 **DAPO**（Yu et al., ByteDance Seed & 清华 AIR, NeurIPS 2025）：作者发现朴素 GRPO 在 Qwen2.5-32B 上只能达到 AIME 约 30 分，原因包括熵坍塌、奖励噪声和训练不稳定。改进如下：

| 技术 | 做法 | 解决的问题 |
|---|---|---|
| **Clip-Higher** | 上下裁剪边界解耦（如 $\epsilon_{low}=0.2,\ \epsilon_{high}=0.28$） | 低概率 token 难以被提高 → 熵坍塌、探索不足 |
| **动态采样** | 过采样并过滤掉准确率为 0 或 1 的问题组 | 有效梯度样本减少 |
| **Token 级损失** | 对 batch 中所有 token 求平均，而不是先在回答内平均 | 长回答中每个 token 的权重过低（偏差 1） |
| **超长奖励塑形** | 对被截断的回答使用软惩罚，或直接过滤 | 截断导致的奖励噪声 |

另外 DAPO **去掉了 KL 惩罚**，理由是长思维链 RL 中策略本来就应该大幅偏离初始模型。📄 结果：基于 Qwen2.5-32B base 在 AIME 2024 上达到 50 分，超过 DeepSeek-R1-Zero-Qwen-32B 的 47 分，训练步数只用了一半；代码基于 veRL 开源。

### GSPO：序列级的重要性比

📄 **GSPO**（Zheng et al. 2025, Qwen 团队, *Group Sequence Policy Optimization*）认为 GRPO 的不稳定来自**重要性采样权重的误用**：奖励是对整个序列给的，却在每个 token 上计算概率比，这引入的高方差噪声会随回答长度累积，并被裁剪放大，可能导致模型崩溃。GSPO 用**序列似然**定义重要性比（并按长度做几何平均）：

$$s_i(\theta) = \left(\frac{\pi_\theta(o_i\mid q)}{\pi_{\theta_{old}}(o_i\mid q)}\right)^{1/|o_i|}$$

并在**序列级**做裁剪。📄 论文报告 GSPO 显著稳定了 **MoE 模型**的 RL 训练（不再需要"路由回放"等额外技巧），并用于 Qwen3 的训练。

### 🔍 小结

| 方法 | 修改点 |
|---|---|
| GRPO | 组内均值 + 标准差归一化；回答内平均；token 级比；KL 放在损失里 |
| Dr. GRPO | 去掉长度归一化和标准差归一化 |
| DAPO | Clip-Higher、动态采样、token 级损失、超长塑形、去掉 KL |
| GSPO | 序列级重要性比与裁剪 |
| RLOO | 留一法基线的 REINFORCE |
| REINFORCE++ | 用全局（batch 级）优势归一化代替组内归一化，并保留 PPO 的部分技巧 |

### 📚 出处
- Liu et al. *Understanding R1-Zero-Like Training: A Critical Perspective*. 2025. arXiv:2503.20783
- Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*. NeurIPS 2025. arXiv:2503.14476
- Zheng et al. *Group Sequence Policy Optimization*. 2025. arXiv:2507.18071
- Hu. *REINFORCE++*. 2025. arXiv:2501.03262

---

## Q5：结果奖励 vs 过程奖励 ⭐⭐⭐⭐

### 🎯 一句话标答

> 结果奖励（ORM / 规则）只评判最终答案，过程奖励（PRM）评判每个推理步骤。📄 DeepSeek-R1 **只使用基于规则的结果奖励**（准确性 + 格式，后来加了语言一致性），并在"不成功的尝试"一节中解释了为什么没有使用 PRM。

### 📄 R1 对 PRM 的判断

DeepSeek-R1 论文列出了 PRM 在大规模 RL 中的三个问题：
1. 在一般推理中，**很难明确定义细粒度的"步骤"**。
2. **很难判断中间步骤是否正确**：自动标注效果不好，人工标注无法扩展。
3. 一旦引入基于模型的 PRM，就不可避免地出现 **reward hacking**，并且重新训练 RM 需要额外资源，使训练流程复杂化。

他们的结论是：PRM 在对模型生成的 top-N 回答做重排序或引导搜索时有用，但在大规模 RL 中，其收益相比额外的计算开销有限。论文中同样列为"不成功尝试"的还有 **MCTS**。

### 📄 PRM 的数据来源

| 方法 | 说明 |
|---|---|
| 人工步骤标注 | 📄 PRM800K（Lightman et al. 2023）：约 80 万条针对 GPT-4 生成的 MATH 解答的步骤级人工标注 |
| 蒙特卡洛估计 | 📄 Math-Shepherd（Wang et al., ACL 2024）：从某一步出发多次补全，用最终答对的比例作为这一步的标签，无需人工；DeepSeekMath 用它做了过程监督的 GRPO 实验 |
| LLM 评判 | 用强模型判断每一步是否正确 |

### 💡 规则奖励示例

```python
def r1_style_reward(question, response, gt_answer):
    fmt = 1.0 if re.fullmatch(r"(?s)<think>.*</think>.*", response.strip()) else 0.0
    pred = extract_final_answer(response)          # 例如取 \boxed{} 中的内容
    acc = 1.0 if math_equal(pred, gt_answer) else 0.0
    return acc + 0.1 * fmt                          # 权重仅为示意
```

💭 规则奖励也会被利用：例如答案抽取规则不严格时，模型可能输出多个候选答案"碰运气"；数学等价判断（如 $\frac12$ 与 $0.5$）的实现质量会直接影响训练。

> ❌ 旧版写"OpenAI o1 据传用了 PRM""一道题标注 5–10 美元"，没有可靠来源，已删除。

### 📚 出处
- Lightman et al. *Let's Verify Step by Step*. ICLR 2024. arXiv:2305.20050
- Wang et al. *Math-Shepherd*. ACL 2024. arXiv:2312.08935
- DeepSeek-AI. *DeepSeek-R1*（§4.2 Unsuccessful Attempts）

---

## Q6：DeepSeek-R1 的训练流程 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 📄 R1 的流程：①**冷启动 SFT**（数千条可读的长思维链数据）→ ②**面向推理的 RL**（GRPO + 准确性 / 格式奖励 + **语言一致性奖励**）→ ③**拒绝采样 + SFT**（约 60 万条推理数据 + 约 20 万条非推理数据，**从 DeepSeek-V3-Base 重新微调**）→ ④**全场景 RL**（推理任务用规则奖励，通用任务用有用性 / 无害性 RM）。

### 📐 流程

```
DeepSeek-V3-Base
  │ ① 冷启动 SFT：数千条长 CoT（few-shot 长 CoT 提示、让模型生成带反思验证的详细答案、
  │                整理 R1-Zero 的可读输出、人工后处理）
  ▼
  │ ② 推理 RL：GRPO；奖励 = 准确性 + 格式 + 语言一致性（思维链中目标语言词的比例）
  ▼
  │ ③ 拒绝采样：从 ② 的模型采样，只保留正确、可读的回答（约 60 万条）
  │    + 非推理数据（写作、事实问答、自我认知、翻译等，复用 DeepSeek-V3 的 SFT 数据，约 20 万条）
  │    → 在 DeepSeek-V3-Base 上重新做 SFT（约 80 万条，2 个 epoch）
  ▼
  │ ④ 全场景 RL：推理用规则奖励；通用用 RM
  │    有用性 RM 只评估最终总结部分；无害性 RM 评估完整回答（含思考过程）
  ▼
DeepSeek-R1
```

### 💡 关键设计

- **为什么要冷启动？** R1-Zero 的输出存在可读性差、语言混杂等问题；少量高质量长思维链让 RL 有一个更好的起点，并且论文认为这样训练得更好。
- **语言一致性奖励的代价**：📄 论文的消融显示，这个奖励会让推理性能**略有下降**，但输出更符合人类偏好，所以仍然保留。
- **为什么阶段 ③ 从 base 重新训练？** 用阶段 ② 的强推理模型生成数据，再和通用数据混合，在 base 上做一次干净的 SFT，同时获得推理和通用能力。

### 📄 成本

- 📄 DeepSeek-V3 报告：最终一次预训练（加上上下文扩展和后训练）约 278.8 万 H800 GPU 小时，按每小时 2 美元计约 **557.6 万美元**。报告明确说明这**不包括**此前的研究和消融实验成本。
- 📄 DeepSeek-R1 的 Nature 版本（2025 年 9 月）在补充材料中披露：在 V3-Base 基础上训练 R1 的成本约 **29.4 万美元**。

> ❌ 旧版写"R1 总训练成本约 600 万美元（包含 base 预训练）"，把 V3 的数字当成了 R1 的，并且忽略了"不含研发成本"的说明。

### 🔥 高频追问

**Q：小模型怎么获得推理能力？**

A：📄 R1 论文用 R1 生成的约 80 万条数据**蒸馏**了 Qwen2.5 和 Llama 系列（1.5B–70B）。他们还对比了在 Qwen-32B-Base 上直接做大规模 RL 的结果，发现其效果约等于 QwQ-32B-Preview，而蒸馏得到的模型明显更强。结论是：**对小模型，蒸馏比直接 RL 更经济有效**；但要突破现有能力的边界，可能仍需要更强的 base 模型和大规模 RL。

**Q：社区复现有哪些？**

A：Open-R1（HuggingFace）、TinyZero、SimpleRL-Zoo（Zeng et al. 2025，系统研究了不同 base 模型上的 zero-RL）、Open-Reasoner-Zero、DAPO 等。

### 📚 出处
- DeepSeek-AI. *DeepSeek-R1*. arXiv:2501.12948；Guo et al. Nature 645, 633–638 (2025)
- DeepSeek-AI. *DeepSeek-V3 Technical Report*. arXiv:2412.19437
- Zeng et al. *SimpleRL-Zoo*. 2025. arXiv:2503.18892

---

## Q7：R1-Zero 与"顿悟时刻" ⭐⭐⭐⭐

### 🎯 一句话标答

> 📄 R1-Zero 直接在 DeepSeek-V3-Base 上做 GRPO，不经过 SFT，只用规则奖励。AIME 2024 的 pass@1 从 15.6% 提升到 71.0%（多数投票 86.7%），与 OpenAI-o1-0912 相当；回答长度在训练中自然增长，并出现了被作者称为"顿悟时刻"（aha moment）的自我反思行为。📄 但后续研究发现，**这类自我反思在 base 模型中本来就存在**，"顿悟"是否真的是 RL 涌现的，存在争议。

### 📄 R1-Zero 的主要观察

- 使用一个简单的模板，要求模型先在 `<think>` 中推理再给出答案，不对内容做任何约束。
- 随着训练进行，平均回答长度持续增长（从数百 token 增长到上万 token），模型学会了"花更多时间思考"。
- 出现了反思、重新评估、尝试其他解法等行为。论文展示了一个中间 checkpoint 的输出，其中模型写道"Wait, wait. Wait. That's an aha moment I can flag here"。
- 问题：可读性差、语言混杂。

> ❌ 旧版中的"训练曲线"（"~1K steps 发生顿悟"）和"Step 0/200/500/1000 的长度与准确率表"是编造的；旧版的中英混杂示例也是虚构的，已删除。真实曲线请看原论文图 2、图 3。

### 📄 反方证据

- 📄 **Liu et al. 2025（Dr. GRPO 论文）**：在 RL 之前，DeepSeek-V3-Base 就已经会产生"顿悟"式的自我反思；Qwen2.5 base 模型即使不用提示模板也表现出很强的推理能力，暗示其预训练数据中可能包含大量此类内容。他们还发现，自我反思行为的出现与准确率之间没有明显的相关性。
- 💭 另一个视角：回答变长可能部分来自 GRPO 的长度偏差（Q4 偏差 1），而不完全是"学会了思考更久"。

❓ **开放问题**：RL 中出现的反思行为，在多大程度上是"新学到的策略"，又在多大程度上是"放大了 base 模型中已有的模式"？

### 📚 出处
- DeepSeek-AI. *DeepSeek-R1*（§2.2）
- Liu et al. 2025. arXiv:2503.20783

---

## Q8：RL 真的教会了推理吗？ ⭐⭐⭐⭐⭐

> 这是 2025 年最热的研究争论之一，也是理解"RL 到底在做什么"的关键。

### 📄 观点一：RLVR 主要提高采样效率，没有扩展能力边界

📄 **Yue et al., NeurIPS 2025（oral）**（*Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?*）：用 **pass@k**（ $k$ 次采样中至少有一次正确的比例）衡量"能力边界"：
- 在 $k$ 较小时（如 pass@1），RL 模型明显优于 base 模型；
- 但在 $k$ 很大时（如 pass@256），**base 模型反而追上甚至超过** RL 模型；
- RL 模型的正确推理路径**已经存在于 base 模型的采样分布中**；随着 RLVR 训练，能力边界往往**变窄**。
- 相比之下，**蒸馏**可以从教师引入新的推理模式，真正扩展能力。
- 作者也指出了局限：结论来自有限的模型族、算法和基准；在数学题上，大 $k$ 时的 pass@k 可能被"猜中"夸大。

### 📄 观点二：奖励信号可能没有我们以为的那么重要

📄 **Shao et al., ICML 2026**（*Spurious Rewards: Rethinking Training Signals in RLVR*）：在 Qwen2.5-Math-7B 上，用**随机奖励**做 GRPO 就能让 MATH-500 提升约 21 个点，接近使用真实奖励的约 29 个点；格式奖励、甚至错误标签的奖励也能带来明显提升。
- 作者把原因归于 GRPO 裁剪项带来的偏差：它会放大模型在预训练中已经很强的行为（对 Qwen2.5-Math 来说是"用代码推理"，频率从约 65% 上升到 90% 以上）。
- **这一效应依赖模型**：同样的虚假奖励在 Llama 3、OLMo 2 上几乎无效甚至有害。
- 💭 启示：只在 Qwen 系列上验证的 RLVR 结论需要谨慎对待。

### 📄 观点三：RL 可以扩展能力，但需要条件

- 💭 有后续工作认为，在更长的训练、更难的题目、更大的探索空间（如保持熵、允许更长回答）下，RL 可以解决 base 模型在大 $k$ 下也解决不了的问题。这一方向的证据仍在积累。
- 📄 Chu et al., ICML 2025（第 7 章 Q9）：在规则变体和视觉变体上，RL 的泛化明显好于 SFT。

### 🔍 小结

| 问题 | 目前的证据 |
|---|---|
| RL 能否提升 pass@1？ | 能，这一点没有争议 |
| RL 能否扩展 pass@k 意义上的能力边界？ | 有争议；多项研究发现边界收窄 |
| 奖励的正确性有多重要？ | 依赖模型；在 Qwen2.5-Math 上随机奖励也有效，在其他模型上无效 |
| 蒸馏 vs RL | 小模型上蒸馏更划算（R1 论文）；蒸馏可能引入新模式（Yue et al.） |

❓ **开放问题**：
- 如何设计实验，可靠地区分"学到新能力"与"重新加权已有能力"？
- 熵坍塌与能力边界收窄之间是什么关系？📄 Cui et al. 2025（*The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models*）分析了策略熵在 RL 中的变化规律，发现性能提升往往以熵的快速下降为代价，并提出了控制熵的方法。

### 📚 出处
- Yue et al. NeurIPS 2025. arXiv:2504.13837
- Shao et al. *Spurious Rewards*. ICML 2026. arXiv:2506.10947
- Cui et al. 2025. arXiv:2505.22617

---

## Q9：工程问题与其他方向 ⭐⭐⭐⭐

### 🔍 工程难点

| 问题 | 说明 |
|---|---|
| 生成占主要时间 | 长思维链（上万 token）的采样远比训练慢；需要 vLLM / SGLang 等推理引擎、异步 rollout |
| 长度不均衡 | 同一 batch 中少数超长回答拖慢整体（长尾）；DAPO 的超长过滤、部分 rollout 等方法缓解 |
| 训练与推理的数值不一致 | 训练引擎和推理引擎算出的 log-prob 不同（kernel、精度、MoE 路由不同），重要性比偏离 1，等价于在做 off-policy 训练；有工作用**截断重要性采样**进行修正。GSPO 的序列级比率对这种误差更鲁棒 |
| 熵坍塌 | 策略熵快速下降，探索不足；Clip-Higher、熵正则、控制高协方差 token 的更新等方法 |
| 规则奖励的实现 | 答案抽取、数学等价判断、代码沙箱的安全与稳定性 |

### 🔍 其他相关方向

- **Kimi k1.5**（📄 Kimi Team 2025）：长上下文 RL（128K）、基于在线镜像下降的策略优化、"长到短"的蒸馏以减少过度思考。
- **过度思考**：📄 Chen et al. 2024（*Do NOT Think That Much for 2+3=?*）发现 o1 类模型在简单问题上消耗了过多 token；如何根据难度分配思考预算是一个活跃方向（→ 第 14 章）。
- **Agent 的多轮 RL**：把工具调用、检索纳入 rollout（如 Search-R1），奖励稀疏、轨迹长，信用分配更难（→ 第 18 章）。
- **更通用的奖励**：在没有可验证答案的任务上，用生成式 RM / 评分标准（rubric）作为奖励，把 RLVR 扩展到写作、开放问答等领域。

### 🧪 动手验证思路

1. 用 TRL 或 veRL 在 Qwen2.5-0.5B/1.5B 上，以 GSM8K 为数据、规则奖励训练 GRPO，记录准确率、回答长度、策略熵的变化。
2. 对比"回答内平均"和"token 级平均"（DAPO / Dr. GRPO）两种损失聚合方式，观察错误回答的长度变化。
3. 在训练前后分别测量 pass@1 和 pass@k（ $k$ 从 1 到 256），复现 Yue et al. 的曲线交叉现象。
4. 换成随机奖励，分别在 Qwen 和 Llama 系列的小模型上训练，检验 Spurious Rewards 的模型依赖性。

### 📚 出处
- Kimi Team. *Kimi k1.5: Scaling Reinforcement Learning with LLMs*. 2025. arXiv:2501.12599
- Chen et al. 2024. arXiv:2412.21187
- Jin et al. *Search-R1*. 2025. arXiv:2503.09516

---

## 📝 本章小结

| 关键点 | 你必须能脱口而出 | 性质 |
|---|---|---|
| GRPO | 组内归一化优势、按回答共享；无 critic；KL 在损失中（ $k_3$） | 🧮 + 📄 |
| $k_3$ | $(u-1)-\log u$，无偏且非负 | 🧮 |
| 组基线 | 是 $V(q)$ 的 MC 估计，但含自身样本有偏；留一法无偏（RLOO） | 🧮 + 📄 |
| GRPO 的偏差 | 长度归一化 → 错误回答变长；标准差归一化 → 难度偏差 | 🧮 + 📄 |
| 修正 | Dr. GRPO、DAPO（Clip-Higher、动态采样、token 级损失、超长塑形）、GSPO（序列级比率） | 📄 |
| 价值方法 | VAPO 表明 critic 仍有价值 | 📄 |
| R1 流程 | 冷启动 SFT → 推理 RL（含语言一致性奖励）→ 拒绝采样 80 万条从 base 重训 → 全场景 RL | 📄 |
| PRM | R1 未用：步骤难定义、难判断、reward hacking | 📄 |
| 成本 | V3 最终训练约 558 万美元（不含研发）；R1 在其上约 29.4 万美元 | 📄 |
| 顿悟时刻 | base 模型中已存在；是否"涌现"有争议 | 📄 + ❓ |
| RL 是否扩展能力 | pass@1 提升；pass@k 边界可能收窄；随机奖励在 Qwen 上也有效 | 📄 + ❓ |

## ✅ 自测题

1. 写出 GRPO 的目标函数，并说明结果监督和过程监督下优势的计算方式有何不同。
2. 证明 $k_3$ 是 KL 的无偏估计并且非负。
3. 为什么说"组均值基线"是有偏的？留一法基线与它是什么关系？
4. 用一个具体例子说明 GRPO 的长度归一化为什么会让错误回答变长。
5. 列出 DAPO 的四项改进分别解决了什么问题。GSPO 与 GRPO 的根本区别是什么？
6. 描述 DeepSeek-R1 的四个阶段，以及为什么第三阶段要从 base 重新训练。
7. 总结 Yue et al. 2025 和 Spurious Rewards 的结论。如果你要证明"RL 让模型学到了新的推理能力"，你会设计什么实验？

---

> **下一章预告**：第 12 章｜推理基础与 KV Cache 工程——prefill 与 decode 的计算特性、KV Cache 的显存账本、PagedAttention、前缀缓存，以及 KV Cache 压缩。
