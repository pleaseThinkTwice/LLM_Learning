# 第 18 章｜Agent 架构

> **本章定位**：RAG（第 16 章）让模型能读，工具调用（第 17 章）让模型能做，本章把它们装进一个能推理、能记忆、能修正的循环。四条主线：
> 1. **错误会相乘**：💭 若单步成功率 95% 且各步独立，20 步只剩约 36%。所以 Agent 工程的重点除了"让模型更强"，还有**减少步数**和**加入验证**。
> 2. **"编译 vs 解释"的取舍**：Plan-and-Execute 先规划后执行，ReAct 边看边走；判据是你对任务结构的了解程度。
> 3. **上下文是有限、有成本、会"腐坏"的资源**，而上下文压缩与 Prefix Caching 存在结构性冲突。
> 4. 📄 Anthropic 的建议：**先找最简单的方案，只在必要时增加复杂度**。Agent 用可预测性换灵活性，很多场景并不需要它。

> **配套章节**：停止标记与控制权交接 → 第 17 章 Q2 / 并行与 DAG → 第 17 章 Q4 / 幂等与人工确认 → 第 17 章 Q5 / Prefix Caching → 第 12 章 Q7 / RAG 技术 → 第 16 章 / 推理模型的 RL → 第 11 章 / 评估 → 第 19 章

---

## 📏 标注约定

| 标记 | 含义 |
|---|---|
| 📄 **[论文/报告]** | 论文或官方工程报告的结论（附出处） |
| 🧮 **[推导]** | 可推导或可计算 |
| 💭 **[直觉]** | 经验、工程惯例或有争议 |
| ❓ **[开放]** | 尚无共识 |

---

## 📌 本章导航

| 序号 | 题目 | 难度 | 类型 |
|---|---|---|---|
| Q1 | **ReAct 循环——Agent 的骨架** | ⭐⭐⭐⭐⭐ | 必问 |
| Q2 | **Plan-and-Execute vs ReAct** | ⭐⭐⭐⭐⭐ | 必问 |
| Q3 | **Reflexion 与自我修正的边界** | ⭐⭐⭐⭐ | 有反面证据 |
| Q4 | **记忆系统：记什么、怎么忘** | ⭐⭐⭐⭐ | 设计 |
| Q5 | **Context Engineering** | ⭐⭐⭐⭐⭐ | 前沿 |
| Q6 | **多智能体：为什么，以及为什么不** | ⭐⭐⭐⭐⭐ | 有争议 |
| Q7 | **可靠性工程：错误累积与状态管理** | ⭐⭐⭐⭐⭐ | 实战 |
| Q8 | **Workflow vs Agent：什么时候不该用 Agent** | ⭐⭐⭐⭐⭐ | 综合 |

---

## Q1：ReAct 循环——Agent 的骨架 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 📄 **ReAct**（Yao et al., ICLR 2023）让模型**交替**输出推理（Thought）和行动（Action），行动的结果（Observation）再反馈给推理。纯 CoT 没有外部反馈，推理错了无法纠正；纯行动没有规划，容易盲目。💭 用控制论的话说：**CoT 是开环，ReAct 是闭环**。今天的 ReAct 已经不靠 prompt 里手写 `Thought:/Action:` 格式，而是内化进了 chat template：Thought 是 assistant 的普通文本，Action 是 `tool_call`，Observation 是 `role="tool"` 的消息，循环的"心跳"是第 17 章 Q2 的停止标记。

### 📐 循环骨架

```python
async def react_agent(query, tools, max_steps=10):
    messages = [{"role": "user", "content": query}]
    for step in range(max_steps):
        resp = await llm(messages, tools=tools, max_tokens=2048)
        if resp.finish_reason == "length":            # 第 17 章 Q2：防截断
            raise TruncatedError(step)
        messages.append(resp)
        if not resp.tool_calls:                       # 模型不再调用工具 → 结束
            return resp.content
        results = await asyncio.gather(*[execute_tool(tc) for tc in resp.tool_calls],
                                       return_exceptions=True)
        for tc, r in zip(resp.tool_calls, results):   # 按调用顺序固定排列
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": readable(r)})
    messages.append({"role": "user", "content": "已达到最大步数，请基于已有信息给出最终回答。"})
    return (await llm(messages)).content              # 不再传 tools：结构上只能作答
```

💭 最后一步不传 `tools`，比在 prompt 里请求"请直接回答"可靠：**用结构约束行为，而不是用措辞请求行为**。

### 📐 ReAct 论文的关键发现

- 📄 在 HotpotQA 上分析失败案例，**CoT 的失败中 56% 属于幻觉**（事实或推理编造）；ReAct 的幻觉明显减少，但出现了新的失败模式：检索结果不相关时推理被带偏、陷入重复。
- 📄 ReAct 与 CoT-SC 的组合（一方失败时切换到另一方）效果最好。
- 📄 去掉 Thought 只保留行动（Act-only）表现下降。

💭 **外部信息是双刃剑**：它能纠正幻觉，也能带偏推理（第 16 章 Q1 讨论过"相关但错误的上下文"最危险）。在多轮 Agent 中这个效应会被放大。

### 🔥 高频追问

**Q：Thought 有什么用，能省掉吗？**

A：💭 三层理由：
1. **计算量**：每生成一个 token 就做一次完整的前向，Thought 等于给模型额外的串行计算（与 CoT 有效的机制相同）。
2. **显式化中间状态**：把"理解问题、选工具、抽参数"拆成多步，而不是挤在一次前向里。
3. **可观测性**：Thought 是调试 Agent 时理解"它为什么这么做"的主要窗口。

反过来，对简单明确的调用（"北京天气"），Thought 只增加延迟。现代模型在训练中学会了自适应：简单任务直接调用，复杂任务先想。

**Q：为什么现在很少看到手写的 ReAct prompt？**

A：早期做法是在 prompt 里规定 `Thought / Action / Observation` 格式，再用正则解析。问题：正则脆弱；模型未必训练过这种格式；**prompt 里出现 `Observation:` 的模式，模型会倾向于自己续写 Observation**，只能靠字符串级的 stop 截断。现在这些都由原生的工具调用接口和停止标记处理。💭 这是一个普遍规律：**有效的 prompt 技巧，往往最终被吸收进模型训练或基础设施**（ReAct → 原生工具调用；"Let's think step by step" → 推理模型的原生思考；"请输出 JSON" → 约束解码与 Structured Outputs）。

**Q：ReAct 有哪些典型失败？**

A：① **编造 Observation**：模型在没有真正调用工具的情况下写出结果，常见原因是训练时没有 mask observation（第 17 章 Q1）或旧式 prompt 的格式缺陷；② **循环**：反复调用同一工具（第 17 章 Q5）；③ **短视**：只看下一步，没有全局规划，这是 Plan-and-Execute 的动机（Q2）。

### 📚 出处
- Yao et al. *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023. arXiv:2210.03629
- Anthropic. *Building Effective Agents*. 2024（工程博客）

---

## Q2：Plan-and-Execute vs ReAct ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> ReAct 是**解释执行**：每一步都调用一次 LLM 决定下一步，灵活但慢；🧮 如果每轮都要带上全部历史，总 token 数随步数**二次增长**。Plan-and-Execute 是**编译执行**：先一次规划，再执行（执行阶段可以用小模型甚至不用 LLM）。📄 **ReWOO**（Xu et al. 2023）让规划器用变量占位（`#E1`）而**不看任何观测结果**，规划只需一次调用，在 HotpotQA 上报告约 5× 的 token 效率提升，准确率还提高约 4 个百分点；代价是不能根据中间结果调整。💭 判据是**你对任务结构的了解程度**：知道路就规划，不知道路就探索；实践中多是混合：**规划 → 执行 → 发现偏差 → 重新规划**。

### 📐 ReAct 的 token 复杂度（🧮）

设固定部分（system + tools + query）长 $L$，每次观测长 $o$，第 $k$ 轮的 prompt 长约 $L + (k-1)o$：

$$\text{Total} = \sum_{k=1}^{N}\big(L + (k-1)o\big) = NL + o\thinspace \frac{N(N-1)}{2} = O(N^2)$$

| 轮数（ $L=5000$、 $o=2000$） | ReAct 总输入 token |
|---|---|
| 5 | 45,000 |
| 10 | 140,000 |
| 20 | 480,000 |

> 🧮 **Prefix Caching 改变了多少？** 每轮的 prompt 是上一轮的 prompt 加新观测，前缀可以完整命中缓存，**需要真正计算的只有增量部分**，prefill 成本接近线性。但有三点不变：① decode 不被缓存，N 次调用的输出照付；② 每轮仍是一次完整的 LLM 往返，**E2E 延迟仍随步数线性增长**；③ **上下文空间照样被占满**（Q5）。缓存省的是钱，不是延迟和上下文。

### 📐 ReWOO：规划不看观测

```
Plan（一次调用）：
  #E1 = search["A 公司 2023 营收"]
  #E2 = search["B 公司 2023 营收"]
  #E3 = compare[#E1, #E2]          ← 变量占位，此时还不知道值
Worker：依次执行，用实际结果填充 #E1、#E2、#E3
Solver（一次调用）：汇总
```

💭 这类似编译中的"符号引用 + 链接"：规划器生成带未解析符号的计划，执行器负责填入实际值。

### 📐 四种范式

| | ReAct | Plan-and-Execute | ReWOO | LLMCompiler（第 17 章 Q4） |
|---|---|---|---|---|
| LLM 调用 | 每步一次 | 1 次规划 + 执行 | 2 次（规划、汇总） | 1–2 次 |
| 根据中间结果调整 | 完全可以 | 需要重新规划 | 不能 | 有限 |
| 并行 | 否 | 部分 | 可以 | DAG 并行 |
| 适合 | 探索性任务 | 通用 | 可预先规划且要省 | 可预先规划且要快 |

### 🔥 高频追问

**Q：怎么判断该重新规划？**

A：触发条件：工具硬失败或返回空（免费检测）；结果与该步目的不符（可用小模型判断）；计划依赖的假设被推翻。💭 关键原则是**检测成本不能超过它省下的成本**：如果每一步都用大模型判断"要不要重规划"，就退化成了 ReAct。另外，**如果某类任务总要重规划好几次，说明它本质上是探索性的**，应该直接用 ReAct。

**Q：规划本身错了怎么办？**

A：规划是在信息最少时做的，只能靠模型的先验去猜。缓解：
1. **先探索再规划**：先调用一两个"看看有什么"的工具（如列出可用数据），把"猜"变成"看"。
2. **分段规划**：💭 每次只规划接下来的 $k$ 步。 $k=1$ 就是 ReAct， $k=N$ 就是纯规划， $k$ 是"编译 ↔ 解释"之间的**连续旋钮**，类似 JIT 介于预编译和解释之间。
3. **可恢复**：检查点与回滚（Q7）；有副作用的步骤要按第 17 章 Q5 分级确认，因为规划错误可能不可逆。

**Q：工业上用哪个？**

A：💭 常见形态是**外层规划、内层 ReAct**：例如深度研究类产品先把问题分解成子问题，每个子问题内部用 ReAct 检索；编程 Agent 以 ReAct 为主（代码任务高度探索性），同时维护一个 TODO 列表作为轻量规划。TODO 列表还有一个产品价值：**它把模型的内部规划变成用户可见、可干预的界面**，部分补回了 Agent 损失的可预测性。

### 📚 出处
- Xu et al. *ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models*. 2023. arXiv:2305.18323
- Wang et al. *Plan-and-Solve Prompting*. ACL 2023. arXiv:2305.04091
- Kim et al. *An LLM Compiler for Parallel Function Calling*. ICML 2024. arXiv:2312.04511

---

## Q3：Reflexion 与自我修正的边界 ⭐⭐⭐⭐

### 🎯 一句话标答

> 📄 **Reflexion**（Shinn et al., NeurIPS 2023）让模型在失败后写一段"为什么失败、下次怎么做"的文字反思，存入记忆，下次尝试时放进上下文；作者称之为"语言强化学习"：不改权重，用文字更新上下文。它在 HumanEval 上报告了 91% 的 pass@1。**但它有效的前提是有可靠的外部反馈**（单元测试、环境信号）：📄 Huang et al.（ICLR 2024, *Large Language Models Cannot Self-Correct Reasoning Yet*）发现，**没有外部反馈的"内在自我纠正"会让推理任务的表现下降**，模型常把正确答案改错。💭 判据是**生成-验证差距**：验证比生成容易的任务（代码有测试、数学有答案）自我修正有效；验证和生成一样难的任务（开放写作、价值判断）往往无效甚至有害。

### 📐 Reflexion 的机制

```
尝试 1：写代码 → 跑测试 → 失败：IndexError at line 5
反思：  "我在循环中用 i+1 做索引但没检查边界，下次应使用 range(len(arr)-1) 或加边界检查。"
尝试 2：上下文 = 任务 + 上次的反思 → 重写 → 通过
```

| | 传统 RL | Reflexion |
|---|---|---|
| 更新什么 | 权重 | 上下文 |
| 用什么更新 | 梯度 | 文字 |
| 需要训练 | 是 | 否 |
| 反馈的信息量 | 一个标量奖励 | 一段诊断 |

### 📐 反面证据与生成-验证差距

📄 Huang et al. 2024 的核心发现：在没有外部反馈、只提示模型"检查并改进你的答案"时，GSM8K、CommonSenseQA 等任务上的表现**下降**；先前一些自我纠正工作的提升，部分来自使用了标准答案来决定何时停止，或者初始 prompt 本身不够好。💭 一种解释是："再检查一遍"本身暗示答案有错，模型倾向于迎合这个暗示。

| 任务 | 生成 | 验证 | 自我修正 |
|---|---|---|---|
| 代码 | 难 | 跑测试，便宜 | 有效（有外部反馈时） |
| 数学（有标准答案） | 难 | 对答案，便宜 | 有效 |
| 事实问答 | 中 | 检索核对 | 有效（📄 CRITIC 用搜索、计算器等外部工具做验证） |
| 开放写作 | 难 | 一样难 | 往往无效 |

💭 **"验证比生成容易"是一条贯穿全书的原理**：投机解码（大模型并行验证比自回归生成便宜，第 14 章）、Best-of-N 与奖励模型（第 9 章）、推理模型用可验证奖励做 RL（第 11 章）、以及这里的 Reflexion。

### 🔥 高频追问

**Q：没有外部验证器的任务怎么办？**

A：💭 **制造一个验证器**：
1. 把主观任务拆成可检查的子目标（覆盖了这 5 个要点吗？每个论断有引用吗？字数在范围内吗？）。
2. 用另一个模型做评审，至少避开自我偏好（第 19 章 Q2），但它仍不是客观标准。
3. 用人做验证。

一个实用的自检：**如果连人都很难快速判断对错，自我修正大概率没用**，模型凭什么能自己发现错误？

**Q：Reflexion 的记忆会无限增长吗？反思会出错吗？**

A：会增长，需要限制条数、合并或按需检索。更根本的问题是**反思本身可能归因错误**：失败其实是 API 限流，模型却反思成"我的算法有问题"，带着错误的经验下次更糟，而且会持续污染记忆。💭 所以 Reflexion 在**有明确错误信息**（如具体的异常和行号）的场景好用，在只有"失败"一个比特的场景不好用。

**Q：Reflexion 和 RLHF、推理模型的 RL 是什么关系？**

A：

| | RLHF（第 9 章） | Reflexion | 推理模型的 RL（第 11 章） |
|---|---|---|---|
| 改什么 | 权重 | 上下文 | 权重 |
| 反馈 | 人类偏好 → 奖励模型 | 外部验证器 | 可验证的奖励（答案、测试） |
| 何时 | 训练时 | 推理时 | 训练时 |
| 是否持久 | 是 | 只在本次会话 | 是 |

💭 推理模型的长思考中常出现"等等，我算错了""换个思路""把答案代回去验证"等行为，这些是在可验证奖励上用 RL 训出来的（📄 DeepSeek-R1 报告了此类反思行为的涌现）。可以把它看成**把 Reflexion 式的自我检查训进了权重**；而它能训出来，正是因为数学和代码有便宜可靠的验证器。

### 📚 出处
- Shinn et al. *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023. arXiv:2303.11366
- Huang et al. *Large Language Models Cannot Self-Correct Reasoning Yet*. ICLR 2024. arXiv:2310.01798
- Madaan et al. *Self-Refine: Iterative Refinement with Self-Feedback*. NeurIPS 2023. arXiv:2303.17651
- Gou et al. *CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing*. ICLR 2024. arXiv:2305.11738

---

## Q4：记忆系统：记什么、怎么忘 ⭐⭐⭐⭐

### 🎯 一句话标答

> 短期记忆是上下文窗口，长期记忆是外部存储加检索。借用认知科学的分类，长期记忆可分为情景记忆（发生过什么）、语义记忆（事实）、程序记忆（怎么做）。难点不在"存"，而在**记什么**和**怎么忘**。📄 **Generative Agents**（Park et al., UIST 2023）用 **Recency（时间衰减）+ Importance（写入时由 LLM 打 1–10 分）+ Relevance（embedding 相似度）** 加权检索，并定期做 **Reflection** 把底层观察聚合成高层结论；📄 **MemGPT**（Packer et al. 2023）把上下文当作"虚拟内存"，由 LLM 通过函数调用自己换入换出。💭 记忆本质上是"压缩 + 检索"，第 16 章的 RAG 技术基本都能复用。

### 📐 Generative Agents 的检索

$$\text{Score}(m) = \alpha_{\text{rec}}\thinspace \hat r(m) + \alpha_{\text{imp}}\thinspace \hat\imath(m) + \alpha_{\text{rel}}\thinspace \hat s(m)$$

📄 论文设定：Recency 按"距上次被访问的游戏小时数"做指数衰减，衰减因子 0.995；Importance 在**写入时**由 LLM 打 1–10 分；Relevance 是 query 与记忆的 embedding 余弦相似度；三项各自 min-max 归一化到 [0, 1]，三个 $\alpha$ 都取 1。

两个工程细节：
1. **Importance 在写入时打分**，否则每次检索都要给成千上万条记忆打分。💭 把昂贵操作挪到写入路径（写少读多）。
2. **被检索到会刷新访问时间**，常用记忆不衰减，类似 LRU。

💭 **一个可以改进的地方**：三个分数量纲和分布完全不同，min-max 归一化容易被离群值破坏（第 16 章 Q4）。可以考虑用 RRF 按排名融合。

### 📐 记什么：设计 vs 涌现

| | 设计（按 schema 抽取） | 涌现（LLM 判断值不值得记） |
|---|---|---|
| 优点 | 可控、可查询、可审计、便宜 | 灵活，能捕捉预料之外的信息 |
| 缺点 | schema 之外的全部丢失 | 判据模糊、打分不稳定、成本高 |

💭 **"重要性"是相对于用途的属性**：对点餐助手，"用户忌口"是关键信息；对客服助手，它可能毫无用处。让 LLM 打分时，要在 prompt 中说明这个 Agent 是做什么的、打分标准是"对未来完成该任务的帮助"，打分才有意义、才稳定。实践中常用混合：关键字段按 schema 抽取，其余交给 LLM 判断。

### 📐 怎么忘：冲突与更新

💭 遗忘是功能而不是缺陷。冲突的记忆（"住在北京" vs "刚搬到上海"）如果都保留且同等对待，检索时会召回互相矛盾的信息。处理步骤：检测冲突 → 判断是**更新**（覆盖）、**补充**（共存）、**例外**（共存并标注条件）还是**旧记录错误**（删除）→ 执行 → **保留历史**（标记为被取代而不是物理删除，便于审计，"以前住北京"本身也可能有用）。

💭 判断类型需要理解时序与因果，LLM 并不可靠。一个稳妥的兜底：**不确定时两条都保留并带时间戳，让 Recency 自然地把新的排在前面**。能用客观信号（时间）解决的，就不要依赖模型判断。

### 📐 方案对比

| | 全部放进上下文 | 摘要 | Generative Agents | MemGPT | Mem0 等记忆层 |
|---|---|---|---|---|---|
| 检索 | 无 | 无 | 三分数加权 | LLM 自己换入换出 | 混合检索 |
| 遗忘 | 无 | 一次性丢细节 | 时间衰减 | 换出到外部存储 | 冲突消解 |
| 对 Prefix Caching | 友好（只追加） | 破坏 | 破坏（每次检索不同） | 破坏 | 破坏 |

### 🔥 高频追问

**Q：记忆和 Prefix Caching 冲突吗？**

A：冲突。检索到的记忆每次不同，而它逻辑上是"背景"，通常放在对话历史之前，一变就让后面的缓存全部失效。缓解：稳定的用户档案固定放前面，动态检索的部分放后面；**只在会话开始时检索一次**，会话内不再变化（用一点召回率换整个会话的缓存）；或者干脆把检索到的记忆放在最后，💭 利用 Lost in the Middle 中"结尾位置利用率高"的特点，效果需要实测。

### 📚 出处
- Park et al. *Generative Agents: Interactive Simulacra of Human Behavior*. UIST 2023. arXiv:2304.03442
- Packer et al. *MemGPT: Towards LLMs as Operating Systems*. 2023. arXiv:2310.08560
- Chhikara et al. *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory*. 2025. arXiv:2504.19413

---

## Q5：Context Engineering ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 上下文是**有限、有成本、会腐坏**的资源。📄 Anthropic 和 Chroma 的报告都指出 **context rot**：随着输入变长，模型准确利用其中信息的能力下降，所有模型都有这个现象，只是程度不同。常见手段可归为四类（📄 LangChain 的分类）：**Write**（写到上下文之外：笔记、记忆）、**Select**（按需拉入：RAG、工具检索）、**Compress**（压缩：摘要、裁剪）、**Isolate**（隔离：子 Agent 各自有独立上下文）。其中 **Compress 与 Prefix Caching 有结构性冲突**：压缩要改写前缀，缓存要求只在末尾追加。💭 折中是**在固定检查点压缩**，而不是每轮滑动窗口。

### 📐 为什么"塞得越多"不一定越好

1. 📄 **Lost in the Middle**（第 16 章 Q1）：中部信息利用率低。
2. 📄 **Context Rot**（Chroma 2025）：评测 18 个模型（包括 GPT-4.1、Claude 4、Gemini 2.5、Qwen3），发现即使在简单任务上，表现也随输入长度变化而变得越来越不可靠，模型并不均匀地使用上下文。
3. 📄 **RULER**（Hsieh et al. 2024）：所测的号称支持 ≥32K 的模型中，约一半在 32K 上还能保持令人满意的表现。
4. 💭 **注意力分散**：softmax 的权重和为 1，上下文越长，相关信息与大量无关 token 竞争注意力。📄 Scalable-Softmax（Nakanishi 2025）从这个角度提出随长度缩放 softmax 的方案。
5. 💭 **训练分布**：长上下文能力通常靠位置编码外推加少量长样本训练获得（第 2 章），接近上限的长度上模型相当于在分布边缘工作。

> 💭 实践建议：在自己的任务上实测有效长度，把关键信息放在不同位置，看准确率从哪里开始下降。

### 📐 四类手段与缓存的关系

| | Write | Select | Compress | Isolate |
|---|---|---|---|---|
| 例子 | 笔记文件、长期记忆 | RAG、Tool RAG | 摘要、裁剪 | 子 Agent |
| 信息损失 | 无（可读回） | 可能漏召回 | 有损 | 通信损失 |
| 对 Prefix Caching | **友好**（只追加句柄） | 破坏 | **破坏** | **友好**（各自独立） |

💭 所以在可行时优先用 Write 和 Isolate。

### 📐 压缩与缓存的冲突（🧮）

```
压缩前：[sys][tools][轮1]...[轮10]        ← 全部可命中
压缩后：[sys][tools][摘要(新)][轮9][轮10]  ← 只有 sys+tools 命中
滑动窗口：每轮丢掉最早一轮 → 开头每轮都变 → 每轮缓存全部失效
```

**检查点压缩**：前 10 轮只追加（缓存全程有效）→ 第 11 轮压缩一次（失效一次）→ 之后在新前缀上继续只追加。💭 把"连续的破坏"变成"离散的、可预期的破坏"。前面各章的冲突大多有巧妙化解（分页解决碎片、chunked prefill 让瓶颈互补、PD 分离解开 SLO 耦合），这一个没有免费的解法，只能选折中点。

📄 Anthropic 的 *Effective context engineering for AI agents* 给出的实践：
- **Just-in-time**：不预加载全部内容，给模型工具按需读取（像人用 `ls`、`grep`、打开文件，而不是把整个文件系统读进脑子；文件名、目录结构本身就是有用的元数据）。
- **Compaction**：接近上下文上限时总结并在新上下文中继续，"通常是上下文工程的第一个杠杆"。
- **Structured note-taking**：Agent 在上下文之外维护笔记。
- **Sub-agent**：把探索的脏活交给子 Agent，只返回凝练的结论。

### 📐 外化：最省的压缩

```python
doc_id = store(huge_document)               # 5 万 token 存到外部
messages.append({"role": "tool", "content": json.dumps({
    "doc_id": doc_id,
    "summary": "这份文档讨论了 X、Y、Z 三个主题，共 50 页",
    "hint": "用 read_doc(doc_id, section='X') 读取具体章节"})})
```

只花几十个 token，而且是**追加**，不影响缓存；需要细节时模型自己去读。💭 这与第 16 章的 Small-to-Big、第 17 章 Q5 的大结果处理是同一个模式：**放一个适合决策的表示，详情另开工具**。

### 🔥 高频追问

**Q：Agent 的上下文怎么组织？**

A：

```
① system prompt            永不变                 ┐
② tools schema             顺序固定、序列化稳定     │ 缓存命中区
③ 示例                      固定                   │
④ 用户档案/长期记忆          会话开始时检索一次后固定 ┘
⑤ 压缩后的早期历史           只在检查点变化（变化时⑥⑦⑧失效）
⑥ 最近 N 轮原文             只追加
⑦ 当前观测                  只追加
⑧ 用户新输入                只追加（也处于利用率较高的结尾位置）
```

两条常见事故：**在 system prompt 开头放时间戳**（每次请求前缀都不同，缓存命中率归零且没有任何报错，只能靠监控命中率发现；时间应放在末尾或做成工具）；**工具按相关性动态排序**。

**Q：预加载还是 Just-in-time？**

A：💭 JIT 用延迟换上下文质量：预加载一次 prefill（且可缓存），JIT 可能要多轮工具调用才找到所需内容。按**"用到的概率 × 大小"**做判断：小而几乎必用的（项目 README、目录结构、常用工具）预加载；大或不确定的（具体文件内容、大检索结果）按需读取。

### 📚 出处
- Anthropic. *Effective context engineering for AI agents*. 2025（工程博客）
- Hong et al. *Context Rot: How Increasing Input Tokens Impacts LLM Performance*. Chroma Research, 2025
- LangChain. *Context Engineering for Agents*. 2025（博客，Write/Select/Compress/Isolate 分类）
- Hsieh et al. *RULER*. 2024. arXiv:2404.06654
- Nakanishi. *Scalable-Softmax Is Superior for Attention*. 2025. arXiv:2501.19399

---

## Q6：多智能体：为什么，以及为什么不 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 💭 多 Agent 最实际的理由往往不是"专业化"，而是**上下文隔离**：每个子 Agent 在自己干净的上下文里读几十个网页，主 Agent 只收凝练的结论。📄 Anthropic 报告，在其内部研究类评测上，Claude Opus 4 主控 + Sonnet 4 子 Agent 的多 Agent 系统比单 Agent Opus 4 好 90.2%；但 token 消耗约为普通对话的 15 倍（单 Agent 约 4 倍），而且在 BrowseComp 上**仅 token 用量就解释了约 80% 的性能方差**。📄 Cognition 则在 *Don't Build Multi-Agents* 中反对：子 Agent 互相看不到对方的决策，结果会不一致。💭 两者都对，判据是：**子任务之间需不需要知道对方在做什么**。

### 📐 双方的论证

**支持（Anthropic，深度研究）**：子 Agent 并行探索不同方向，各自压缩信息，主 Agent 在干净的上下文中综合。对广度优先、可并行的查询（例如找出某个指数所有成分公司的董事会成员）尤其有效。

**反对（Cognition，编程）**：📄 他们的例子：任务"做一个 Flappy Bird 克隆"被拆成"做背景和管道"与"做小鸟"两个子任务，一个子 Agent 做出了超级马里奥风格的背景，另一个做出的小鸟风格不匹配，两部分无法协调。核心论点：**上下文隔离既是优势，也是缺陷**：隔离带来干净，也带来失联。他们主张单线程、长上下文，必要时做压缩。

| 任务 | 子任务需要互相了解吗 | 倾向 |
|---|---|---|
| 调研 A、B、C 三家公司 | 不需要 | 多 Agent |
| 从大量网页中找答案 | 读网页互相独立 | 多 Agent |
| 重构一个代码库 | 每处改动互相影响 | 单 Agent |
| 写一份风格统一的长报告 | 需要一致的论点和风格 | 单 Agent 或强协调 |

💭 Anthropic 做的是研究（子任务独立），Cognition 做的是写代码（子任务深度耦合），**他们在讨论不同的任务**。

### 📐 成本（📄 Anthropic）

| | token 消耗（相对普通对话） |
|---|---|
| 普通对话 | 1× |
| 单 Agent | 约 4× |
| 多 Agent | 约 15× |

> ❌ 旧版写"多 Agent 比单 Agent 贵 4 倍"，原文的 4× 是单 Agent 相对普通对话。

💭 因此只在**任务价值足够高、且可以并行分解**时使用。❓ 由于 token 用量本身解释了大部分方差，"**在相同 token 预算下**，多 Agent 比单 Agent 加更长思考还剩多少优势"是一个有价值的开放问题。

### 📐 编排模式

| 模式 | 结构 | 说明 |
|---|---|---|
| Orchestrator–Worker | 主管动态分派 | 最常用 |
| Hierarchical | 多层主管 | 超大任务 |
| Network / Swarm | 任意互相调用 | 难以控制 |
| Sequential / Pipeline | 固定流水线 | 💭 流程固定时，这其实是 Workflow（Q8） |
| Debate | 多个 Agent 辩论 | 需要多视角时 |

💭 **"多 Agent"一词常被滥用**：如果没有任何一个 Agent 在运行时动态决定下一步，那只是几个 prompt 串起来的 Workflow。

### 🔥 高频追问

**Q：子 Agent 返回原始数据还是摘要？**

A：返回原始数据会让主 Agent 的上下文爆掉，失去多 Agent 的意义；返回摘要则把判断权下放给子 Agent。💭 更危险的是**摘要会抹掉不确定性**：原始材料里"一个来源说 895 亿、另一个说 890 亿"，摘要里可能变成一个看起来确定的数字。缓解：结构化返回，包含摘要、来源、置信度、冲突说明和原始数据的句柄（主 Agent 有疑问时可以自己去读）。

**Q：怎么防止子 Agent 递归爆炸？**

A：限制深度（通常主 Agent 加一层子 Agent 就够）；**不给子 Agent 创建子 Agent 的工具**（💭 从能力上杜绝，比 prompt 里写"请不要"可靠得多）；全局 token 预算；限制一次派出的数量。

**Q：主 Agent 和子 Agent 之间怎么通信？**

A：给子 Agent 的不应只有任务，还要有**目的和输出要求**（"查 A 公司 2023 年的营收和净利润，要和 B 公司对比，请给出可比口径"），子 Agent 遇到口径冲突时才知道怎么取舍。📄 Anthropic 也强调主 Agent 需要给子 Agent 明确的目标、输出格式、工具和任务边界，否则会重复工作或遗漏。子 Agent 之间默认不通信；💭 如果非通信不可，说明任务分解得不干净。合并结果时，把"**指出各子结果之间的矛盾并说明采信哪个**"作为主 Agent 的显式任务，这是对 Cognition 批评的直接工程回应。

### 📚 出处
- Anthropic. *How we built our multi-agent research system*. 2025（工程博客）
- Cognition. *Don't Build Multi-Agents*. 2025（博客）

---

## Q7：可靠性工程：错误累积与状态管理 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 🧮 若每步独立、成功率为 $p$， $N$ 步的成功率约为 $p^N$： $0.95^{20}\approx 36$%。三条应对路线：**提高单步准确率**（边际收益递减）、**减少步数**（💭 杠杆最大、最常被低估：并行、DAG、更强的组合工具）、**加入验证与恢复**（检查点、回滚）。Agent 是有状态的长任务，还需要**持久化执行**（崩溃后能续跑）、**补偿（Saga）**（有副作用的步骤要能撤销）和**完整的 trace**（失败是路径依赖的，必须能重放整条路径）。

### 📐 错误累积（🧮）

| 单步成功率 | 5 步 | 10 步 | 20 步 | 50 步 |
|---|---|---|---|---|
| 90% | 59% | 35% | 12% | 0.5% |
| 95% | 77% | 60% | 36% | 7.7% |
| 99% | 95% | 90% | 82% | 61% |
| 99.9% | 99.5% | 99% | 98% | 95% |

要让 $N$ 步任务达到成功率 $T$，单步需要 $p \ge T^{1/N}$：20 步要 90% 成功率，单步需约 99.5%。

**减少步数 vs 提高单步**：步数从 20 减半到 10，成功率 36% → 60%；单步从 95% 提到 96%，20 步只到 44%。

> 💭 **这个模型的两个强假设**：各步独立（真实中错误高度相关）；一步错就全盘失败（真实 Agent 常能恢复）。所以 $p^N$ 是偏悲观的直觉工具，不是预测公式。
> 📄 **METR 的经验视角**（Kwa et al. 2025, *Measuring AI Ability to Complete Long Software Tasks*）：用"模型以 50% 成功率能完成的任务，人类专家需要多长时间"（time horizon）衡量 Agent 能力，发现该指标在 2019–2025 年间约**每 7 个月翻一倍**；成功率随任务时长的下降可以用 logistic 曲线拟合。

**加入验证**：🧮 一个玩具模型：每 $K$ 步做一次验证，失败时以概率 $r$ 被发现并重跑一次（重跑仍以 $p^K$ 成功），假设验证器本身不出错，则每段成功率为 $p^K + (1-p^K)\thinspace r\thinspace p^K$。代入 $p=0.95$、 $N=20$、 $K=5$、 $r=0.8$：每段约 0.914，四段约 **70%**，高于不验证的 36%。前提是有可靠的验证器（Q3）。

💭 **"更强的工具"常被低估**：与其让 Agent 用三个原子工具组合出一个操作，不如直接提供组合好的工具；给 Agent 一个 SQL 工具，往往比给它十个增删改查工具更可靠，因为一步能做完的事不需要五步。

### 📐 状态管理

- **持久化执行**：每步保存检查点（如 LangGraph 的 checkpointer、Temporal 一类的工作流引擎），进程崩溃后从断点恢复，而不是重跑所有 LLM 调用。
- **Saga 补偿**：每个有副作用的操作都配一个补偿操作；失败时**按逆序**执行补偿（先退款再取消订单）。💭 它与第 17 章 Q5 的幂等性是一对：幂等保证"重试安全"，补偿保证"回滚安全"。
- **Trace**：记录每一步的 prompt、响应、工具调用、结果、延迟、成本。单次调用的问题看输入输出就够，Agent 的问题可能第 3 步埋下、第 7 步才显现。

### 🔥 高频追问

**Q：怎么定位是哪一步出的错？**

A：
1. **Trace**：人工回看整条路径。
2. **首次出错步数的分布**：💭 集中在前两步 → 理解或规划问题；集中在中后期 → 上下文变长导致的退化或错误累积；均匀分布 → 单步能力不足。
3. **Teacher forcing 归因**：把怀疑出错的那一步替换成正确的，重跑后续步骤；成功则确认是这一步的问题。💭 类似 `git bisect`，把"单步能力"和"错误累积"解耦（与第 17 章 Q8 的多轮评估方法相同）。

**Q：成本怎么控制？**

A：减少步数（同时降成本、提成功率，是唯一的"双赢"）；Prefix Caching；模型分级（规划用大模型，执行和子 Agent 用小模型）；每任务 token 预算；缓存只读工具的结果（注意时效）。

**Q：什么指标能提前发现 Agent 变差？**

A：💭 **平均步数**（变长说明模型在挣扎，比成功率更早、而且实时可见）；同参数重复调用的比例（死循环先兆）；参数校验失败率（schema 或模型变了）；Prefix Cache 命中率突降（prompt 模板被改坏）；"无法完成"的比例；用户中途放弃率。

### 📚 出处
- Kwa et al. *Measuring AI Ability to Complete Long Software Tasks*. METR, 2025. arXiv:2503.14499
- Richardson. Saga pattern（microservices.io）
- LangGraph 文档：Persistence

---

## Q8：Workflow vs Agent：什么时候不该用 Agent ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 📄 Anthropic 的定义：**Workflow** 是 LLM 和工具通过**预定义的代码路径**编排；**Agent** 是 LLM **动态决定**自己的流程和工具使用。区别在于**谁决定流程**。核心建议："找到最简单的方案，只在必要时增加复杂度"。复杂度阶梯：单次 LLM 调用（加检索、工具）→ Workflow（prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer）→ Agent。💭 只有当任务开放、步数不可预测、需要根据中间结果调整、能信任模型的决策、且错误代价可控时，才值得上 Agent。

### 📐 五种 Workflow 模式（📄 Anthropic *Building Effective Agents*）

| 模式 | 结构 | 适合 | 与本书其他章节 |
|---|---|---|---|
| Prompt chaining | A → B → C | 任务能分解为固定步骤 | — |
| Routing | 分类后分流 | 输入有明显类别 | 第 16 章 Q6 的路由 |
| Parallelization | 分工并行或多次投票 | 可并行或需要多视角 | — |
| Orchestrator–workers | 主管动态分派子任务 | 子任务数量不确定 | 最接近 Agent 的 Workflow |
| Evaluator–optimizer | 生成 → 评审 → 修改 | 有明确评价标准 | Q3：前提是可靠的验证 |

💭 Orchestrator–workers 与 Agent 的界线：前者由主管动态决定分派几个 worker，但 worker 的类型和流程是固定的；后者连"下一步做什么"都由模型决定。

### 📐 复杂度的代价（💭 量级示意）

| | 单次调用 | Workflow | Agent | 多 Agent |
|---|---|---|---|---|
| 成本 | 1× | 数倍 | 更高 | 📄 约为对话的 15× |
| 可预测性 | 完全 | 高 | 低 | 很低 |
| 可调试、可测试 | 易 | 易 | 难（需要 trace，路径不唯一） | 很难 |
| 能力上限 | 低 | 中 | 高 | 最高 |

💭 **可预测性本身有价值**：成本可预测才能做预算，延迟可预测才能定 SLO，行为可预测才能测试和审计，失败可复现才能调试。**Agent 是用可预测性换灵活性。**

### 🔥 高频追问

**Q：怎么判断我的场景该用哪个？**

A：
1. 任务步骤固定吗？固定 → Workflow，到此为止。
2. 需要根据中间结果改变后续步骤吗？不需要 → Plan-and-Execute（Q2）。
3. 错误代价可控吗？不可控 → Agent 加严格的人工确认（第 17 章 Q5）。
4. 任务价值值得 Agent 的成本吗？不值得 → 退回 Workflow，接受能力上限。

💭 一个实用的检查：**如果你能用流程图画出 Agent 该怎么做，你需要的就是把这张流程图写成代码。**

**Q：Agent 和 Workflow 能混合吗？**

A：💭 能，而且这是主流：**Workflow 做骨架，Agent 处理少数复杂请求**。例如客服系统先做意图路由：查询类走单次 LLM + RAG，操作类走固定流程加人工确认，只有少数复杂请求交给 Agent。反过来，**Agent 内部可以把可靠的 Workflow 包装成工具调用**，提高单步准确率（Q7）。这体现了本书反复出现的模式：**分级处理，只在必要时付出昂贵的成本**（RRF 粗排 → rerank 精排、小模型猜 → 大模型验、低维粗筛 → 高维精排）。

**Q：为什么"从简单开始"这么难被采纳？**

A：💭 三个偏见：Agent 更"有技术含量"；**demo 是单次采样，生产是期望值**（演示时跑一次、挑最好的那次，而 $p^N$ 和 pass^k 说明生产中的方差很大，第 17 章 Q8）；以及"更通用就更好"的错觉（实际流量往往集中在少数几种情况）。📄 Anthropic 的原话："Success in the LLM space isn't about building the most sophisticated system. It's about building the right system for your needs."

### 📚 出处
- Anthropic. *Building Effective Agents*. 2024（工程博客）
- Anthropic. *Writing effective tools for agents*. 2025（工程博客）

---

## 🔬 研究前沿与开放问题

- 📄 **从 prompt 编排到端到端 RL**：编程 Agent 的进展越来越多来自在真实环境中用结果奖励做 RL，例如 SWE-RL（Wei et al., NeurIPS 2025）用开源软件演化数据做 RL。💭 本章不少"架构技巧"可能被训进权重，正如推理模型把 Reflexion 式的自我检查训了进去（Q3）。
- 📄 **评测在快速演化**：SWE-bench Verified（人工筛选的 500 题子集）、τ-bench（多轮、有状态、pass^k，第 17 章）、GAIA、OSWorld、METR 的 time horizon。💭 共同趋势是从单轮正确率转向**长程、有状态、看可靠性**的指标。
- ❓ **错误的相关性**： $p^N$ 假设独立，真实失败高度相关。如何从 trace 数据估计"可恢复错误"与"致命错误"的比例？这决定了检查点和验证器该放在哪里。
- ❓ **等预算下的多 Agent 增益**：既然 token 用量解释了大部分性能方差，固定总 token 预算后，多 Agent 相对"单 Agent + 更长思考"还有多少优势？
- ❓ **记忆的评测**：长期记忆系统缺少公认的评测（何时该记住、何时该忘记、冲突如何处理），现有评测多是合成的长对话问答。
- 🧪 **实验设想（文字）**：在一个可验证任务集（如 SWE-bench Verified 子集）上，固定总 token 预算，比较 ① 单 Agent ② 单 Agent + 检查点压缩 ③ 主从多 Agent，记录成功率、平均步数和 Prefix Cache 命中率，可以同时检验 Q5（压缩与缓存）和 Q6（隔离的收益）。

---

## 📝 本章小结

| 关键点 | 你必须能说清楚 | 性质 |
|---|---|---|
| ReAct | 交替推理与行动；CoT 开环、ReAct 闭环；已内化进原生工具调用 | 📄 + 💭 |
| Thought | 额外的串行计算；显式化中间状态；可观测性 | 💭 |
| ReAct 成本 | 输入 token 随步数二次增长；Prefix Caching 省钱不省延迟和上下文 | 🧮 |
| ReWOO | 规划不看观测，变量占位；约 5× token 效率 | 📄 |
| 分段规划 | $k$ 是"编译 ↔ 解释"的连续旋钮 | 💭 |
| Reflexion | 文字形式的反馈；有效的前提是外部验证器 | 📄 |
| 内在自我纠正 | 没有外部反馈时会让推理表现下降 | 📄 |
| 生成-验证差距 | 验证比生成容易，自我修正才有效 | 💭 |
| 记忆 | 三分数检索 + Reflection；重要性由用途定义；遗忘与冲突处理 | 📄 + 💭 |
| Context rot | 有效利用率随长度下降 | 📄 |
| 四类手段 | Write / Select / Compress / Isolate；Write 和 Isolate 对缓存友好 | 📄 + 🧮 |
| 压缩 vs 缓存 | 结构性冲突；检查点压缩；滑动窗口最伤缓存 | 🧮 |
| 多 Agent | 理由是上下文隔离；90.2% 与 15×；token 用量解释约 80% 方差 | 📄 |
| Anthropic vs Cognition | 判据：子任务是否需要互相了解 | 💭 |
| 错误累积 | $p^N$；减少步数杠杆最大；模型假设偏悲观 | 🧮 + 📄 |
| 状态管理 | 检查点、Saga 补偿、trace | 💭 |
| Workflow vs Agent | 谁决定流程；用可预测性换灵活性；能画流程图就写成代码 | 📄 + 💭 |

## ✅ 自测题

1. CoT 和 ReAct 的本质区别是什么？为什么说外部信息是双刃剑？
2. 用"每个 token 一次前向"解释 Thought 的作用。什么时候 Thought 是浪费？
3. 推导 ReAct 的输入 token 复杂度。ReWOO 为什么能降下来？Prefix Caching 在成本、延迟、上下文三方面各改变了什么？
4. 什么是生成-验证差距？为什么"内在自我纠正"可能有害？各举三个差距大和差距小的任务。
5. Generative Agents 的三个检索分数是什么？直接相加有什么问题？你会怎么改？
6. 为什么"这条记忆重要吗"是一个不完整的问题？
7. 为什么上下文压缩和 Prefix Caching 不可兼得？检查点压缩是怎么折中的？为什么滑动窗口最糟？
8. Anthropic 说多 Agent 好 90.2%，Cognition 说别做多 Agent。用一句话调和，并说明"token 用量解释约 80% 方差"对这个结论意味着什么。
9. 计算 $0.95^{20}$，说明为什么减少步数比提高单步更有效，并指出 $p^N$ 模型的两个强假设。
10. 你的 Agent 成功率只有 40%，你的前三个排查动作是什么？
11. **设计题**：设计一个企业财报分析助手（"对比 A、B、C 三家公司 2023 年的营收和利润率并分析差异原因"；数据源：财报 PDF 库、结构化财务数据库、新闻）。给出架构选型及理由、上下文组织、记忆策略、错误处理、成本估算和评测方案，并说明每一处的取舍。

---

> **下一章预告**：第 19 章｜评估体系——评估的层次与 Goodhart 定律、LLM-as-Judge 的偏差与校准、benchmark 污染、评测集的构建、端到端归因与 oracle 实验，以及离线到线上的完整评估链。第 16–18 章讲"怎么做"，第 19 章讲"你怎么知道你做对了"。
