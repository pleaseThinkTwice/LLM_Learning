# 第 10 章｜DPO 数学推导专题

> **本章定位**：DPO 把"KL 约束下的 RLHF"变成一个对偏好对的分类问题，不需要显式的奖励模型，也不需要在线采样。本章完整推导 DPO，分析它的**梯度**和**隐式奖励**，然后重点讨论它的**已知问题**：被选中回答的概率也会下降、偏向更长的回答、在确定性偏好下忽略 KL 约束、离线数据的分布偏移。最后整理 IPO、KTO、SimPO、ORPO、在线/迭代 DPO 等变体分别针对哪个问题。

> **配套章节**：RLHF、Bradley–Terry、KL 闭式解 → 第 9 章 / GRPO 与可验证奖励 → 第 11 章

> **标记约定**：📄 论文结论 / 🧮 可推导 / 💭 直觉或有争议 / ❓ 开放问题

---

## 📌 本章导航

| 序号 | 题目 | 难度 | 类型 |
|---|---|---|---|
| Q1 | DPO 是什么？和 PPO 的本质区别 | ⭐⭐⭐⭐ | 概念 |
| Q2 | **DPO 完整推导** | ⭐⭐⭐⭐⭐ | 推导 |
| Q3 | **DPO 的梯度与 β 的作用** | ⭐⭐⭐⭐⭐ | 推导 |
| Q4 | 隐式奖励与奖励等价类 | ⭐⭐⭐⭐ | 理论 |
| Q5 | **DPO 的已知问题** | ⭐⭐⭐⭐⭐ | 研究 |
| Q6 | 参考模型的选择 | ⭐⭐⭐ | 工程 |
| Q7 | DPO 实践：数据、超参、监控 | ⭐⭐⭐⭐ | 工程 |
| Q8 | **变体：IPO、KTO、SimPO、ORPO、在线 / 迭代 DPO** | ⭐⭐⭐⭐⭐ | 前沿 |
| Q9 | DPO 还是 PPO？证据怎么说 | ⭐⭐⭐⭐ | 决策 |

---

## Q1：DPO 是什么？和 PPO 的本质区别 ⭐⭐⭐⭐

### 🎯 一句话标答

> 📄 DPO（Rafailov et al., NeurIPS 2023）从 KL 正则 RLHF 的闭式最优解出发，把奖励函数用"策略与参考模型的对数概率比"重新参数化，代入 Bradley–Terry 偏好模型后，得到一个**只依赖策略本身**的二分类损失。训练只需要策略模型、冻结的参考模型和**离线的偏好对**，不需要奖励模型、价值模型和在线采样。

### 🔍 对比

| 维度 | PPO（RLHF） | DPO |
|---|---|---|
| 需要的模型 | 策略、价值、参考、RM | 策略、参考 |
| 数据 | 在线采样的回答 + RM 打分 | 固定的偏好对 $(x, y_w, y_l)$ |
| 优化 | 强化学习 | 监督式的分类损失 |
| 能否利用 RM 之外的奖励（如代码执行结果） | 能 | 需要先把奖励转换成偏好对 |
| 数据是否在策略分布上 | 是（on-policy） | 通常不是（off-policy） |

📄 DPO 原论文在情感控制（IMDb）、摘要（TL;DR）和单轮对话（Anthropic HH）上报告，DPO 达到或超过了 PPO 的效果。📄 但后续更细致的比较发现，调好的 PPO 在很多任务上优于 DPO（Q9）。

### 🔥 高频追问

**Q：DPO 真的没有奖励模型吗？**

A：没有**显式**的 RM，但有隐式奖励 $\hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}$。论文的副标题就是 *Your Language Model is Secretly a Reward Model*。

**Q：哪些模型用了 DPO？**

A：📄 Zephyr-7B（HuggingFace, 2023）、Tülu 2（AI2, 2023）、Mixtral-Instruct、LLaMA 3 的后训练（拒绝采样 + SFT + DPO 的多轮迭代）等。

### 📚 出处
- Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023. arXiv:2305.18290
- Tunstall et al. *Zephyr*. 2023. arXiv:2310.16944
- Ivison et al. *Camels in a Changing Climate*（Tülu 2）. 2023. arXiv:2311.10702

---

## Q2：DPO 完整推导 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 四步：①KL 正则 RLHF 的最优策略 $\pi^*\propto\pi_{ref}e^{r/\beta}$；②反解出 $r = \beta\log\frac{\pi^*}{\pi_{ref}} + \beta\log Z(x)$；③代入 Bradley–Terry，$\beta\log Z(x)$ 在差值中抵消；④把 $\pi^*$ 换成 $\pi_\theta$，对偏好数据做最大似然。

### 🧮 Step 0：目标

对每个 prompt $x$：

$$\max_\pi\ \mathbb{E}_{y\sim\pi(\cdot\mid x)}\left[r(x,y)\right] - \beta\,\text{KL}\big(\pi(\cdot\mid x)\,\|\,\pi_{ref}(\cdot\mid x)\big)$$

### 🧮 Step 1：最优策略（拉格朗日乘子法）

在约束 $\sum_y\pi(y\mid x)=1$ 下：

$$\mathcal{L}(\pi,\lambda) = \sum_y\pi(y\mid x)\left[r(x,y) - \beta\log\frac{\pi(y\mid x)}{\pi_{ref}(y\mid x)}\right] - \lambda\left(\sum_y\pi(y\mid x) - 1\right)$$

对每个 $y$ 求偏导（注意 $\frac{\partial}{\partial\pi}\pi\log\pi = \log\pi + 1$）：

$$r(x,y) - \beta\log\frac{\pi(y\mid x)}{\pi_{ref}(y\mid x)} - \beta - \lambda = 0$$

解得

$$\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{ref}(y\mid x)\exp\!\left(\frac{r(x,y)}{\beta}\right),\qquad Z(x) = \sum_y\pi_{ref}(y\mid x)\exp\!\left(\frac{r(x,y)}{\beta}\right)$$

🧮 目标关于 $\pi$ 是严格凹的（期望奖励是线性项，负 KL 是严格凹项），所以驻点就是全局最优。另一种不需要求导的推导见第 9 章 Q5。

### 🧮 Step 2：反解奖励

$$\boxed{r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{ref}(y\mid x)} + \beta\log Z(x)}$$

### 🧮 Step 3：代入 Bradley–Terry

$$P(y_w\succ y_l\mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big) = \sigma\!\left(\beta\log\frac{\pi^*(y_w\mid x)}{\pi_{ref}(y_w\mid x)} - \beta\log\frac{\pi^*(y_l\mid x)}{\pi_{ref}(y_l\mid x)}\right)$$

$\beta\log Z(x)$ 只依赖 $x$，在差值中**抵消**。这一点至关重要：$Z(x)$ 需要对所有可能的回答求和，无法计算。

📄 论文还推广到了 Plackett–Luce 模型（$K$ 个回答的完整排序），$Z(x)$ 同样会抵消。

### 🧮 Step 4：DPO 损失

$$\boxed{\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}\right)\right]}$$

### 💻 实现

```python
import torch.nn.functional as F

def sequence_logps(logits, labels, ignore_index=-100):
    # logits: (B, T, V)，labels: (B, T)，prompt 部分为 ignore_index
    logits, labels = logits[:, :-1], labels[:, 1:]
    mask = labels != ignore_index
    logp = logits.log_softmax(-1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(-1)               # 对回答部分的 token 求和（不是平均）

def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    loss = -F.logsigmoid(logits).mean()
    reward_w, reward_l = beta * (pi_w - ref_w).detach(), beta * (pi_l - ref_l).detach()
    return loss, reward_w, reward_l
```

**要点**：$\log\pi(y\mid x)$ 是回答部分所有 token 对数概率的**和**，prompt 部分不计入。参考模型的对数概率可以预先计算并缓存。

### ⚠️ 常见陷阱

1. 求导时 $\pi\log\pi$ 的导数有一个 $+1$。
2. $Z(x)$ 依赖 $x$，所以只能在**同一个 prompt** 的两个回答之间抵消。
3. 序列对数概率是求和；如果改成按长度平均，就得到了 SimPO 一类的变体（Q8）。

---

## Q3：DPO 的梯度与 β 的作用 ⭐⭐⭐⭐⭐

### 🧮 梯度

令 $\hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}$。📄 DPO 损失的梯度为

$$\nabla_\theta\mathcal{L}_{DPO} = -\beta\,\mathbb{E}\Big[\underbrace{\sigma\big(\hat r_\theta(x,y_l) - \hat r_\theta(x,y_w)\big)}_{\text{隐式奖励"排错"的程度}}\Big(\underbrace{\nabla_\theta\log\pi_\theta(y_w\mid x)}_{\text{提高 }y_w} - \underbrace{\nabla_\theta\log\pi_\theta(y_l\mid x)}_{\text{降低 }y_l}\Big)\Big]$$

🧮 解读：
- 每个样本的权重是 $\sigma(\hat r_l - \hat r_w)$：隐式奖励把这一对**排得越错**，权重越大；已经排对、而且差距很大的样本，权重趋近于 0。
- 📄 论文的消融表明，去掉这个权重（即直接最大化 $\log\pi(y_w) - \log\pi(y_l)$，类似"非似然训练"）会导致模型退化。

### 🧮 β 的两层作用

1. **KL 强度**：来自 RLHF 目标的 KL 系数。$\beta$ 越大，最优策略越接近 $\pi_{ref}$。
2. **温度**：$\beta$ 缩放对数概率比，决定 sigmoid 在多大的差距时饱和。

🧮 由于 $\nabla\mathcal{L}$ 中有一个 $\beta$ 因子和 sigmoid 内的 $\beta$，$\beta$ 与学习率之间存在耦合：改变 $\beta$ 时，通常也需要重新调学习率。

### 🔥 高频追问

**Q：β 一般取多少？**

A：📄 DPO 原论文在大部分实验中使用 $\beta=0.1$（TL;DR 摘要用了 0.5）；Zephyr、Tülu 2、LLaMA 3 的 DPO 都使用 $\beta=0.1$。📄 β-DPO（Wu et al., NeurIPS 2024）提出根据每个 batch 的数据质量（偏好对之间的奖励差距）动态调整 $\beta$。

> ❌ 旧版写"DPO 的 β 约为 PPO KL 系数的 5–10 倍，因为偏好信号需要足够强的梯度"，这个解释没有出处。两者在理论上是同一个系数；数值上的差异与数据是否 on-policy、序列级 vs token 级的 KL 估计等因素有关，没有简单的换算关系。

### 📚 出处
- Rafailov et al. 2023（§4 梯度分析）
- Wu et al. *β-DPO: Direct Preference Optimization with Dynamic β*. NeurIPS 2024. arXiv:2407.08639

---

## Q4：隐式奖励与奖励等价类 ⭐⭐⭐⭐

### 🎯 一句话标答

> 📄 DPO 论文证明：对 Bradley–Terry / Plackett–Luce 模型，**两个奖励函数如果只相差一个只依赖 $x$ 的函数，它们诱导出相同的偏好分布和相同的最优策略**（属于同一个等价类）；并且每个等价类都可以用 $r(x,y) = \beta\log\frac{\pi(y\mid x)}{\pi_{ref}(y\mid x)}$ 的形式表示。所以用策略来参数化奖励**不损失表达能力**。

### 🧮 推论

1. 隐式奖励只在**同一个 prompt 内**可比；不同 prompt 之间的绝对值没有意义。
2. 训练好的 DPO 模型可以当作 RM 使用：📄 RewardBench 就把很多 DPO 模型作为"隐式 RM"进行评测。
3. 推理时不需要计算隐式奖励，它只是一个解释和分析工具。

### 🔥 高频追问

**Q：同一份数据训练两次，隐式奖励一样吗？**

A：💭 理论上，最优解在等价类意义下唯一；实际优化不会收敛到精确的最优解，且离线数据只覆盖了很小一部分回答空间，在数据之外的回答上，两次训练得到的隐式奖励可能差别很大（这与 Q5 的分布外问题相关）。

**Q：DPO 能避免 reward hacking 吗？**

A：DPO 没有可被利用的显式 RM，但它仍然会利用**偏好数据中的虚假相关**（例如更长的回答更常被选中）。📄 Rafailov et al. 2024（*Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms*）发现，DPO、IPO、SLiC 等直接对齐算法同样表现出与第 9 章类似的**过优化**模式：随着 KL 增大，真实质量先升后降，并且甚至在训练完一个 epoch 之前就可能出现。

### 📚 出处
- Rafailov et al. 2023（§5 理论分析）
- Rafailov et al. *Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms*. NeurIPS 2024. arXiv:2406.02900

---

## Q5：DPO 的已知问题 ⭐⭐⭐⭐⭐

### 问题 1：被选中回答的概率也在下降

📄 训练过程中常见的现象是：$\log\pi(y_w)$ 和 $\log\pi(y_l)$ **同时下降**，只是 $y_l$ 下降得更多，因此损失仍在下降。概率质量被转移到了**既不是 $y_w$ 也不是 $y_l$** 的其他回答上。

- 📄 Pal et al. 2024（*Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive*）：当 $y_w$ 和 $y_l$ 只有少数 token 不同（编辑距离小）时，DPO 会降低 $y_w$ 的概率；提出 **DPOP**，加入惩罚项防止 $y_w$ 的概率低于参考模型。
- 📄 Razin et al., ICLR 2025（*Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization*）：把这种现象称为**似然位移**，发现它可能把概率转移到语义相反的回答上（例如本来应该拒绝的有害请求被接受）；用隐藏表示的相似度（CHES 分数）可以识别容易导致位移的偏好对。
- 📄 LLaMA 3 的做法：在 DPO 损失中**加入对 $y_w$ 的 NLL 项**（系数 0.2），并且把格式 token（如角色头、结束 token）从 DPO 损失中**屏蔽**，以避免这些在 $y_w$ 和 $y_l$ 中都出现的 token 被同时压低，导致模型重复或忘记停止。

> ❌ 旧版写"chosen 和 rejected 的 reward 都升 → 数据问题"。实际情况相反：训练中更常见、也更值得警惕的是**两者都下降**。

### 问题 2：确定性偏好下 KL 约束失效

📄 Azar et al., AISTATS 2024（IPO 论文）指出：如果某个偏好对的经验偏好概率是 1（数据中总是 $y_w$ 胜），Bradley–Terry 模型要求 $r(y_w) - r(y_l)\to\infty$。🧮 由于 $\hat r$ 是由策略参数化的，这会把 $\pi(y_l)$ 推向 0，并且**无论 $\beta$ 多大**都会发生，即 KL 正则在这种情况下不起作用。在有限数据上，几乎每个偏好对都只被观察到一次，因此这种"确定性偏好"非常普遍。IPO 用平方损失把差值拉向一个有限的目标值来解决（Q8）。

### 问题 3：长度偏好

📄 Park et al., Findings of ACL 2024（*Disentangling Length from Quality in Direct Preference Optimization*）：DPO 训练后回答会明显变长，因为偏好数据中更长的回答更常被选中；提出在损失中加入长度正则项。SimPO 用长度归一化也部分解决了这个问题。

### 问题 4：离线数据与分布偏移

📄 Xu et al., ICML 2024：DPO 的偏好数据通常来自其他模型，与当前策略的分布不同；DPO 可能给**数据中没有出现的回答**分配很高的概率。他们在对话和代码任务上发现调好的 PPO 明显优于 DPO。缓解方法包括：用当前策略**重新生成**偏好对（on-policy）、在线 / 迭代 DPO（Q8）。

### 📚 出处
- Pal et al. *Smaug*. 2024. arXiv:2402.13228
- Razin et al. ICLR 2025. arXiv:2410.08847
- Azar et al. *A General Theoretical Paradigm to Understand Learning from Human Preferences*. AISTATS 2024. arXiv:2310.12036
- Park et al. Findings of ACL 2024. arXiv:2403.19159
- Xu et al. *Is DPO Superior to PPO for LLM Alignment?* ICML 2024. arXiv:2404.10719
- Grattafiori et al. *The Llama 3 Herd of Models*. arXiv:2407.21783（§4.1.4）

---

## Q6：参考模型的选择 ⭐⭐⭐

### 🎯 一句话标答

> 默认用 SFT 模型作为参考模型和初始策略。📄 如果偏好数据不是由 SFT 模型生成的，DPO 原论文建议先在 $y_w$ 上做一次 SFT（"Preferred-FT"），以减小参考模型与数据之间的分布差异。📄 也可以在训练中**更新参考模型**（TR-DPO）。📄 SimPO、ORPO 等方法不需要参考模型。

### 🔍 选项

| 选择 | 说明 |
|---|---|
| SFT 模型（默认） | 起点合理，KL 从 0 开始 |
| 先在 $y_w$ 上 SFT 再作为参考 | 📄 DPO 论文的建议，当偏好数据来自其他模型时使用 |
| 训练中更新参考模型 | 📄 TR-DPO（Gorbatovski et al. 2024）：用软更新（EMA）或硬更新（定期复制）替换参考模型，在多个基准上优于固定参考模型的 DPO |
| 无参考模型 | 📄 SimPO、ORPO（Q8） |

### 🔥 高频追问

**Q：用 LoRA 做 DPO 时，参考模型需要额外的显存吗？**

A：不需要。参考模型就是**关闭 adapter 后的 base 模型**。TRL 的 `DPOTrainer` 在使用 PEFT 时，会通过临时禁用 adapter 来计算参考对数概率，不需要再加载一份模型。另外也可以预先计算参考对数概率并缓存。

### 📚 出处
- Gorbatovski et al. *Learn Your Reference Model for Real Good Alignment*（TR-DPO）. ICLR 2025. arXiv:2404.09656

---

## Q7：DPO 实践：数据、超参、监控 ⭐⭐⭐⭐

### 📄 公开配方

| 来源 | 配置 |
|---|---|
| 📄 DPO 原论文 | $\beta=0.1$，学习率 $10^{-6}$，batch 64，RMSprop，150 步 warmup |
| 📄 Zephyr-7B | 在 UltraFeedback（约 6.4 万条 prompt，每条取最高分和一个随机的低分回答组成偏好对）上训练，$\beta=0.1$，学习率 $5\times10^{-7}$ |
| 📄 Tülu 2 | UltraFeedback，$\beta=0.1$，学习率 $5\times10^{-7}$，3 个 epoch |
| 📄 LLaMA 3 | 学习率 $10^{-5}$、$\beta=0.1$；屏蔽格式 token；加入系数 0.2 的 NLL 项；多轮迭代，每轮使用上一轮最好的模型生成新数据 |

> ❌ 旧版的"学习率不要超过 1e-6""3 epoch 是上限""数据甜区 20K–100K""KL 不超过 5–10"等都没有出处，已替换为上表。

### 🔍 监控什么

| 指标 | 健康的趋势 | 警惕 |
|---|---|---|
| DPO loss | 下降 | 很快降到接近 0（可能过拟合 / 确定性偏好问题） |
| 偏好准确率（$\hat r_w > \hat r_l$ 的比例） | 上升 | — |
| margin $\hat r_w - \hat r_l$ | 上升 | — |
| $\hat r_w$ 和 $\hat r_l$ 各自的值 | 理想情况是 $\hat r_w$ 上升、$\hat r_l$ 下降 | **两者同时大幅下降**：似然位移（Q5） |
| 回答长度 | 变化不大 | 持续增长：长度偏好 |
| 下游评测（AlpacaEval LC、MT-Bench、IFEval 等） | 提升 | 训练指标变好但下游变差：过优化 |

### 🔥 高频追问

**Q：偏好对从哪里来？**

A：
- 人工标注（如 Anthropic HH、Chatbot Arena 的对话）。
- AI 反馈：📄 UltraFeedback（Cui et al. 2023）用多个模型生成回答，再由 GPT-4 按多个维度打分。
- On-policy 构造：用当前模型对每个 prompt 采样多个回答，用 RM、LLM 评判或可验证奖励选出最好和最差的组成偏好对。📄 Tajwar et al. 2024 和多项工作表明，on-policy 的偏好数据通常更有效。

---

## Q8：变体：IPO、KTO、SimPO、ORPO、在线 / 迭代 DPO ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 每个变体都针对 DPO 的某个问题：**IPO** 解决确定性偏好下 KL 失效；**KTO** 只需要"好/坏"的单条标签；**SimPO** 去掉参考模型并用长度归一化；**ORPO** 把 SFT 和偏好优化合成一个阶段、不需要参考模型；**在线 / 迭代 DPO** 解决离线数据的分布偏移。

### 🧮 IPO

📄 Azar et al. 2024 提出一般框架 ΨPO，IPO 是取 Ψ 为恒等函数的特例。令

$$h_\theta(x,y_w,y_l) = \log\frac{\pi_\theta(y_w\mid x)\,\pi_{ref}(y_l\mid x)}{\pi_\theta(y_l\mid x)\,\pi_{ref}(y_w\mid x)}$$

$$\mathcal{L}_{IPO} = \mathbb{E}\left[\left(h_\theta(x,y_w,y_l) - \frac{1}{2\tau}\right)^2\right]$$

🧮 平方损失把对数概率比之差拉向一个**有限**的目标 $\frac1{2\tau}$，而不是像 sigmoid 那样推向无穷，从而在确定性偏好下仍然保持正则化效果。$\tau$ 扮演 KL 系数的角色。

### 🧮 KTO

📄 Ethayarajh et al., ICML 2024：基于 Kahneman–Tversky 的前景理论，只需要每条回答"可取 / 不可取"的二值标签。令 $r_\theta(x,y) = \log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}$，参考点 $z_0 = \text{KL}(\pi_\theta(\cdot\mid x)\,\|\,\pi_{ref}(\cdot\mid x))$（实践中用 batch 内不匹配的 $(x, y')$ 估计，并截断为非负，不回传梯度）：

$$v(x,y) = \begin{cases}\lambda_D\,\sigma\big(\beta(r_\theta(x,y) - z_0)\big) & y\text{ 可取}\\[2pt] \lambda_U\,\sigma\big(\beta(z_0 - r_\theta(x,y))\big) & y\text{ 不可取}\end{cases},\qquad \mathcal{L}_{KTO} = \mathbb{E}\left[\lambda_y - v(x,y)\right]$$

📄 论文报告在 1B–30B 规模上，KTO 只用二值反馈就能匹配或超过 DPO；$\lambda_D, \lambda_U$ 用于处理好坏样本数量不平衡。

> ⚠️ 旧版把参考点 $z_0$ 写成"数据集平均的 reward"，不准确：它是对当前策略与参考模型之间 KL 的估计。

### 🧮 SimPO

📄 Meng, Xia, Chen, NeurIPS 2024：

$$\mathcal{L}_{SimPO} = -\mathbb{E}\left[\log\sigma\!\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w\mid x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l\mid x) - \gamma\right)\right]$$

- 用**长度归一化的平均对数概率**作为隐式奖励，使训练时的奖励和**生成时的打分方式**（平均对数似然）一致。
- **不需要参考模型**，节省显存和计算。
- 目标间隔 $\gamma$ 要求好回答的奖励至少高出一个 margin。
- 📄 由于没有 KL 锚点，SimPO 使用的 $\beta$ 远大于 DPO（论文中通常取 2.0–2.5 左右），二者的 $\beta$ 不可直接比较。论文报告在 AlpacaEval 2 和 Arena-Hard 上明显优于 DPO。

### 🧮 ORPO

📄 Hong et al., EMNLP 2024：在 SFT 损失上加一个**几率比**项，一个阶段完成指令学习和偏好对齐，**不需要参考模型**：

$$\mathcal{L}_{ORPO} = \mathcal{L}_{SFT}(x,y_w) + \lambda\cdot\left(-\log\sigma\!\left(\log\frac{\text{odds}_\theta(y_w\mid x)}{\text{odds}_\theta(y_l\mid x)}\right)\right),\qquad \text{odds}_\theta(y\mid x) = \frac{P_\theta(y\mid x)}{1 - P_\theta(y\mid x)}$$

其中 $P_\theta(y\mid x)$ 是**按长度归一化**的似然 $\exp\left(\frac1{|y|}\sum_t\log\pi_\theta(y_t\mid x,y_{<t})\right)$。

> ❌ 旧版的对比表把 ORPO 标为"需要参考模型"，错误。

### 📄 在线 / 迭代 DPO 与自博弈

| 方法 | 做法 |
|---|---|
| 📄 OAIF（Guo et al. 2024, *Direct Language Model Alignment from Online AI Feedback*） | 每一步用当前策略采样两个回答，由 LLM 评判给出偏好，再做 DPO 更新；在线版本明显优于离线 DPO |
| 📄 迭代 DPO（Xiong et al., ICML 2024；LLaMA 3 后训练） | 多轮：用当前模型生成回答 → 用 RM 打分构造偏好对 → DPO → 下一轮 |
| 📄 Self-Rewarding LM（Yuan et al., ICML 2024） | 模型自己作为评判者（LLM-as-a-judge）为自己的回答打分，构造偏好对做迭代 DPO，评判能力和生成能力一起提升 |
| 📄 SPPO（Wu et al. 2024）、Nash-MD（Munos et al., ICML 2024） | 把对齐看作两个策略之间的博弈，寻找**纳什均衡**，不依赖 Bradley–Terry 的传递性假设 |

### 🔍 汇总

| 方法 | 需要参考模型 | 数据形式 | 针对的问题 |
|---|---|---|---|
| DPO | 是 | 偏好对 | — |
| IPO | 是 | 偏好对 | 确定性偏好下 KL 失效 |
| cDPO / 标签平滑 | 是 | 偏好对 | 标签噪声 |
| DPOP | 是 | 偏好对 | $y_w$ 概率下降 |
| KTO | 是 | 单条 + 好/坏标签 | 偏好对难以获得 |
| SimPO | **否** | 偏好对 | 训练与生成的奖励不一致；长度；显存 |
| ORPO | **否** | 偏好对 | 需要单独 SFT 阶段 |
| 在线 / 迭代 DPO | 是 | 在线生成的偏好对 | 分布偏移 |

### 📚 出处
- Ethayarajh et al. *KTO*. ICML 2024. arXiv:2402.01306
- Meng et al. *SimPO*. NeurIPS 2024. arXiv:2405.14734
- Hong et al. *ORPO*. EMNLP 2024. arXiv:2403.07691
- Guo et al. *OAIF*. 2024. arXiv:2402.04792
- Xiong et al. *Iterative Preference Learning from Human Feedback*. ICML 2024. arXiv:2312.11456
- Yuan et al. *Self-Rewarding Language Models*. ICML 2024. arXiv:2401.10020
- Wu et al. *Self-Play Preference Optimization*. 2024. arXiv:2405.00675
- Munos et al. *Nash Learning from Human Feedback*. ICML 2024. arXiv:2312.00886

---

## Q9：DPO 还是 PPO？证据怎么说 ⭐⭐⭐⭐

### 📄 关键比较研究

- 📄 **Xu et al., ICML 2024**：在对话、代码生成等任务上，经过仔细调参（优势归一化、大 batch、参考模型的指数滑动平均更新等）的 PPO 一致优于 DPO，在竞赛编程上取得了当时的最好结果。DPO 的主要问题是对分布外回答的偏好不可控、对分布偏移敏感。
- 📄 **Ivison et al., NeurIPS 2024**（*Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback*）：在 Tülu 2 上系统比较了偏好数据、学习算法、奖励模型、策略训练 prompt 四个因素。结论：**偏好数据的质量影响最大**，其次是学习算法（PPO 优于 DPO，但差距不大），再次是奖励模型和 prompt 的选择。
- 💭 实践中的折中：很多团队使用**在线 / 迭代 DPO**，兼顾 DPO 的简单和 on-policy 数据的好处。

> ❌ 旧版写"OpenAI / Anthropic 仍用 PPO（可能因为沉没成本）""PPO 上限比 DPO 高 1–2% benchmark"，前者没有可靠来源，后者没有出处，已替换为上面的研究结论。

### 🔍 选择建议

| 场景 | 建议 |
|---|---|
| 资源有限、有现成高质量偏好数据 | DPO（或 SimPO / ORPO） |
| 只有点赞 / 点踩数据 | KTO |
| 追求更好效果、能负担在线采样 | 迭代 / 在线 DPO，或 PPO |
| 有可验证奖励（数学、代码） | 在线 RL（PPO / GRPO，→ 第 11 章） |
| 多轮交互 / Agent 任务 | 在线 RL（→ 第 18 章） |

### 🧪 动手验证思路

1. 在一个 5 个候选回答的玩具问题上，分别用 DPO 和 IPO 训练，让某个偏好对是"确定性"的，观察 DPO 是否把 $\pi(y_l)$ 推向 0 而忽略 $\beta$。
2. 在一个真实偏好数据集上训练 DPO，记录 $\log\pi(y_w)$ 和 $\log\pi(y_l)$ 的变化，观察是否同时下降；再加入 NLL 项对比。
3. 用相同数据比较离线 DPO 和"每轮用当前模型重新采样"的迭代 DPO。
4. 统计训练前后回答长度的变化，并测试 SimPO 的长度归一化是否缓解。

---

## 📝 本章小结

| 关键点 | 你必须能脱口而出 | 性质 |
|---|---|---|
| 闭式解 | $\pi^*\propto\pi_{ref}e^{r/\beta}$ | 🧮 |
| 反解奖励 | $r = \beta\log\frac{\pi^*}{\pi_{ref}} + \beta\log Z(x)$ | 🧮 |
| $Z(x)$ 抵消 | 只在同一 prompt 的两个回答之间 | 🧮 |
| DPO 损失 | $-\log\sigma(\beta[\log\text{ratio}_w - \log\text{ratio}_l])$；序列对数概率求和 | 🧮 |
| 梯度 | 按"隐式奖励排错程度"加权 | 🧮 + 📄 |
| 等价类 | 只差 $f(x)$ 的奖励等价；策略参数化不损失表达能力 | 📄 |
| 已知问题 | 似然位移、确定性偏好下 KL 失效、长度偏好、分布偏移、过优化 | 📄 |
| LLaMA 3 的修补 | 屏蔽格式 token + 0.2 × NLL | 📄 |
| 变体 | IPO（平方损失）、KTO（二值标签）、SimPO（无参考、长度归一）、ORPO（SFT + 几率比、无参考） | 📄 |
| DPO vs PPO | 调好的 PPO 更好；但偏好数据质量影响最大 | 📄 |

## ✅ 自测题

1. 从 KL 正则 RLHF 目标出发，完整推导 DPO 损失。
2. 推导 DPO 损失的梯度，解释每个样本的权重含义。
3. 为什么说在确定性偏好下，DPO 的 KL 正则不起作用？IPO 如何解决？
4. 什么是似然位移？它会带来什么风险？列出两种缓解方法。
5. 写出 SimPO 和 ORPO 的损失，说明它们为什么不需要参考模型。
6. 总结 Xu et al. 2024 和 Ivison et al. 2024 关于 DPO 与 PPO 的结论。
7. 你有 10 万条偏好对和一张 80GB GPU，要对齐一个 13B 模型：给出完整方案，并说明你会监控哪些指标来发现 Q5 中的问题。

---

> **下一章预告**：第 11 章｜GRPO 与推理模型训练——从 PPO 到 GRPO（组内相对优势、去掉 critic）、可验证奖励的 RL（RLVR）、DeepSeek-R1 / R1-Zero 的训练流程，以及 DAPO、Dr. GRPO 等对 GRPO 的修正。
