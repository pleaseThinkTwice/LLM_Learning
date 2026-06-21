# 第 7 章｜SFT 指令微调

> **本章定位**：SFT 是预训练到对齐的桥梁，**数据工程是 90% 的工作量**——LIMA 的"少即是多"、Self-Instruct、Magpie 等是必懂概念。**Loss Masking** 是工程必考细节，多轮对话格式是落地常见坑。本章重点：数据策略 + 工程实现。

> **配套章节**：预训练 → 第 6 章 / PEFT → 第 8 章 / RLHF → 第 9 章

---

## 📌 本章导航

| 序号 | 题目 | 难度 | 频率（字节/阿里/Meta/DeepSeek） | 类型 |
|---|---|---|---|---|
| Q1 | SFT 是什么？和预训练的本质区别 | ⭐⭐⭐ | 4 / 3 / 2 / 2 | 基础 |
| Q2 | **SFT 数据：质量 vs 数量（LIMA）** | ⭐⭐⭐⭐⭐ | **7 / 5 / 4 / 4** | 必问 |
| Q3 | Self-Instruct/Evol-Instruct/Magpie | ⭐⭐⭐⭐ | 5 / 3 / 3 / 4 | 数据工程 |
| Q4 | **Loss Masking 的实现与必要性** | ⭐⭐⭐⭐⭐ | **6 / 4 / 3 / 4** | 易答错 |
| Q5 | Chat Template 与多轮对话格式 | ⭐⭐⭐⭐ | 5 / 3 / 2 / 4 | 工程 |
| Q6 | SFT 中的灾难性遗忘 | ⭐⭐⭐⭐ | 4 / 3 / 2 / 3 | 实战 |
| Q7 | SFT 超参选择经验 | ⭐⭐⭐ | 3 / 2 / 2 / 2 | 工程 |
| Q8 | Instruction Tuning vs SFT 的细微区别 | ⭐⭐⭐ | 2 / 2 / 2 / 1 | 概念 |

---

## Q1：SFT 是什么？和预训练的本质区别 ⭐⭐⭐

### 🎯 一句话标答

> SFT（Supervised Fine-Tuning）= **用指令-响应对的格式继续训练 base model**，让模型学会**听指令并输出符合期望的格式**，本质上还是 next-token prediction，但**数据形态**和预训练完全不同。

### 🗣️ 30 秒口语版

"SFT 和预训练的核心相同 + 关键不同——

**相同点**：都用 next-token prediction 这个目标，loss 都是交叉熵，训练机制完全一致。

**不同点**有三个：

第一，**数据形态**——预训练用海量无标注文本（几 T token），SFT 用结构化的'指令-响应'对（几万到几百万条）。

第二，**目标不同**——预训练学'语言和知识'，SFT 学'怎么响应指令'。比如预训练见过'巴黎是法国首都'这样的陈述，但 SFT 让模型学会'用户问"法国首都是哪里？"，要回答"巴黎"'。

第三，**Loss Masking**——预训练对所有 token 算 loss，SFT 通常**只对响应部分算 loss**——指令部分不算梯度。

实际工业流程：**Pretrain → SFT → RLHF/DPO**。

成本对比：预训练几百万到几千万美元，SFT 几千到几万美元——便宜两到三个数量级。"

### 🔍 SFT 与预训练对比

| 维度 | 预训练 | SFT |
|---|---|---|
| 训练目标 | Next-token prediction | Next-token prediction（同） |
| 数据形态 | 无标注文本 | 指令-响应对 |
| 数据规模 | T 级 token | 万-百万样本 |
| 训练时长 | 几月 | 几小时-几天 |
| Loss 计算 | 所有 token | 仅 response 部分 |
| 目标能力 | 语言理解 + 知识 | 指令遵循 + 格式 |
| 评估指标 | PPL、Benchmark | 指令遵循质量 + 下游任务 |

### 🔥 高频追问 Top 3

**Q：SFT 之后模型为什么还需要 RLHF？**

A：因为 SFT 只学了**"什么样的回答是合格的"**——但合格不等于**优秀**。SFT 数据通常是人写的"标准答案"，模型最多模仿到这个水平。

RLHF/DPO 用**偏好数据**（A 比 B 好）训练，让模型超越人类标注者的平均水平。这是 ChatGPT 之所以比纯 SFT 模型强的关键。

**Q：能不能跳过 SFT 直接 RLHF？**

A：**几乎不行**。RLHF 是基于"已经能产生大致合理回答的模型"做精修——base model 还不会听指令，产生的输出大多是噪声，RLHF 没法工作。必须先 SFT 让模型学会指令格式，再 RLHF 优化偏好。

近期有"SFT-free"的尝试（如直接用 DPO + 大量偏好数据），但效果不如 SFT + DPO 组合。

**Q：SFT 一定要全参微调吗？**

A：不一定。**全参微调**调整所有参数，效果好但贵；**PEFT**（LoRA 等）只调少量参数，性价比高。

工业实践：
- 小公司/资源有限：LoRA + SFT
- 大模型团队：全参 SFT（如 LLaMA-2 的 Chat 版）
- 极致质量场景：全参 SFT + RLHF

### ⚠️ 常见陷阱

1. 不要把 SFT 等同于"微调"——Fine-tuning 是泛称，SFT 特指指令格式的微调
2. 要懂 Loss Masking——不答这点 SFT 答得不完整
3. 不要把 SFT 和 RLHF 混为一谈

---

## Q2：SFT 数据：质量 vs 数量（LIMA） ⭐⭐⭐⭐⭐（必问）

### 🎯 一句话标答

> LIMA 论文（Meta 2023）证明 **1000 条高质量样本就能 SFT 出强模型**——预训练已经塞进了所有知识，SFT 只需要少量样本"激活"指令遵循能力。**质量 >> 数量**是行业共识。

### 🗣️ 30 秒口语版

"这道题是 SFT 章节的'核心题'，必须能完整讲清 LIMA 的实验和启示。

**LIMA 论文的关键发现**——

Meta 团队用 LLaMA-65B 做实验：分别用 **1000 条**和 **30 万条**指令数据做 SFT。结果——
- 1000 条精选数据训出的模型，**在人工评估中胜过用 30 万条 Alpaca 数据训的模型**
- 1000 条版本和 GPT-4 在某些任务上 tie

**LIMA 的核心论断**——'Superficial Alignment Hypothesis'（表面对齐假说）：

> 一个模型的知识和能力**几乎完全来自预训练**。对齐阶段只是教模型**用什么样的格式和风格来呈现这些已有知识**——所以少量高质量样本就够了。

**实际启示**：
- 不要堆量——10 万条噪声数据不如 1000 条精品
- 要保证 **多样性**：覆盖不同任务类型、不同长度、不同风格
- 要保证 **质量**：每条都是人写或精修的高质量回答

**这之后的工业实践**：
- LIMA：1000 条
- LLaMA-2 Chat：约 28000 条精选 SFT 数据
- DeepSeek-V3：1.5M 条 SFT 数据，经过严格筛选

业界共识——**SFT 数据规模在 10K-1M 量级，关键是质量**。"

### 📐 LIMA 实验细节

**数据来源**：
- 750 条来自 Stack Exchange / wikiHow 等高质量社区，人工筛选
- 250 条由作者手工撰写

**评估结果**（300 个测试提示上 head-to-head）：

| 对手 | LIMA 胜率 | 平手 | LIMA 负 |
|---|---|---|---|
| Alpaca-65B (52K 数据) | **62%** | 21% | 17% |
| Davinci-003 | 26% | 35% | 39% |
| GPT-4 | 19% | 38% | 43% |

**结论**：1000 条精选 ≈ 52000 条 Alpaca；接近 Davinci-003 水平。

### 💡 "表面对齐假说"的深层意义

**Pre-LIMA 思路**：堆量、多样化——更多数据 = 更好对齐。

**Post-LIMA 思路**：
- 知识来自**预训练**（已经塞进去）
- SFT 只是**激活**这些知识的"开关"
- 所以**少量、高质量、多样化的样本**最有效
- 数据**配比和多样性 >> 绝对数量**

**进一步推论**：
- 如果预训练充分，SFT 应该越少越好（避免污染知识）
- 数据质量的边际收益远大于数量
- 这也是 DeepSeek、Mistral 等团队投资数据 curation 的原因

### 🔥 高频追问 Top 3

**Q：LIMA 之后，工业上为什么还用几十万到几百万条 SFT 数据？**

A：因为 LIMA 是**学术理想化设定**。工业场景更复杂——
- 需要覆盖**很多领域**（数学、代码、推理、对话、写作...）
- 需要**多语言**支持
- 需要**特定行为**（拒答、安全、格式...）

1000 条覆盖不了这么多维度。**实际工业数据规模 50K-1M，但仍然遵循 LIMA 原则——每一条都精挑细选**。

**Q：LIMA 的结论在小模型上也成立吗？**

A：**部分成立**。LIMA 用 65B 大模型——预训练知识足够"充分"，SFT 只激活。小模型（如 7B）预训练知识不够，可能需要 SFT 数据"补充"知识，所以 SFT 数据量需求相对大一点。

但即便 7B 模型，**几千条高质量样本仍胜过几十万条噪声样本**。质量优先的原则不变。

**Q：怎么评估 SFT 数据质量？**

A：几个维度——
- **指令多样性**：通过 embedding 聚类，看覆盖了多少不同任务
- **响应质量**：人工抽样评估，看响应是否准确、完整、得体
- **长度分布**：避免全是短答案或全是长答案
- **格式正确性**：JSON、代码块、Markdown 等格式是否正确
- **没有矛盾**：同样的指令不应该有冲突的响应

实际工业 pipeline：用 GPT-4 / Claude 等大模型给每条数据打分（**LLM-as-judge**），过滤低分。

### ⚠️ 常见陷阱

1. 要能脱口而出 LIMA 的 1000 条——这是数字 anchor
2. 要懂"表面对齐假说"——这是关键概念，能加分
3. 不要绝对化"越少越好"——工业上还是要覆盖全面

### 🏢 大厂偏好

- **字节 / 阿里 SFT 数据团队**：必问，会让你详述数据 pipeline
- **Meta / DeepSeek**：会问"LIMA 假说在你工作中怎么验证"

### 📚 延伸阅读
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) - 必读
- [DeepSeek-V3 SFT 部分](https://arxiv.org/abs/2412.19437)

---

## Q3：Self-Instruct / Evol-Instruct / Magpie 对比 ⭐⭐⭐⭐

### 🎯 一句话标答

> 这三种是**自动生成 SFT 数据**的代表方法——**Self-Instruct** 让 LLM 自己生成指令-响应对、**Evol-Instruct** 用 LLM 把简单指令进化成复杂指令、**Magpie** 利用 chat template 让 LLM 自己"招供"训练数据。一代比一代精巧。

### 🗣️ 30 秒口语版

"这三种方法解决同一个问题——**人工写 SFT 数据太贵**。

**Self-Instruct（2022）**：
- 给 LLM 一些**种子指令**（175 条手写）
- 让 LLM 生成新指令 + 生成响应
- 经过过滤（去重、低质量过滤），形成 SFT 数据集
- **Alpaca 用 Self-Instruct + GPT-3.5 生成了 52K 数据**

**Evol-Instruct（WizardLM, 2023）**：
- 从简单指令开始，让 LLM **进化**它
- 五种进化操作：增加约束、深化、具体化、增加推理步骤、广度变换
- 迭代多轮，得到更难、更多样的指令
- WizardLM 用这个方法做出顶尖开源模型

**Magpie（2024）**：
- 利用**对齐模型的 chat template** 的特性
- 给模型只输入 `<|user|>` 这个开头标记
- 模型会自动'补全'——生成它训练时见过的'用户指令'
- 然后再生成响应
- **完全无监督地从对齐模型中提取它的训练数据分布**

三者关系——**Self-Instruct → Evol-Instruct（更复杂）→ Magpie（更优雅）**。"

### 📐 Self-Instruct 流程

```
1. 种子集：175 条人工编写的指令-响应对

2. 指令生成：
   Prompt: "参考下面的例子，生成 8 条新的不同的指令：
            [示例1]: ...
            [示例2]: ...
            新指令："
   LLM 生成 → 8 条新指令

3. 任务分类：判断是分类还是生成任务

4. 实例生成：
   对每条新指令，让 LLM 生成响应

5. 过滤：
   - ROUGE-L 相似度 < 0.7（去重）
   - 长度合理
   - 关键词过滤

6. 加入数据池，迭代生成
```

### 📐 Evol-Instruct 的五种进化

给定原始指令 `I = "翻译: hello → ?"`，可以进化为：

**Add Constraints**: "翻译 hello 成法语，且使用古老的表达方式"

**Deepening**: "证明二加二等于四，使用集合论的方法"

**Concretizing**: "讲一个发生在 1920 年代上海，主角是一位记者的悬疑故事"

**Increased Reasoning Steps**: "篮子里有 8 个苹果，又加了 3 个，然后取出 2 个，再加 1.5 倍数量的橘子，问篮子里水果总数"

**In-breadth Evolving**: 换主题/换形式生成完全新的指令

### 📐 Magpie 的"招供"机制

正常使用：
```
输入: "<|user|>什么是 Python?<|assistant|>"
输出: "Python 是..."
```

Magpie 反向利用：
```
输入: "<|user|>"  ← 只给开头
输出: 模型"补全"——继续生成它训练时看过的 user query 分布！
比如生成: "什么是 Python? 它有哪些主要特性?<|assistant|>Python..."
```

**核心 insight**：**对齐模型在训练时见过大量真实指令，让它"招供"这些指令，就得到了真实分布的数据**。完全不需要种子集或人工设计。

### 🔍 三种方法的对比

| 方法 | 来源数据 | 生成方式 | 质量 | 多样性 | 成本 |
|---|---|---|---|---|---|
| Self-Instruct | 175 种子 + LLM | LLM 模仿生成 | 中 | 中 | 低 |
| Evol-Instruct | 原始指令 + LLM | LLM 进化变换 | 高 | 高 | 中 |
| Magpie | 仅对齐模型 | 模型"招供" | 高 | 高 | 低 |

### 🔥 高频追问 Top 3

**Q：自动生成数据有哪些坑？**

A：
- **模式塌缩**：生成的数据风格趋同，多样性低
- **错误传播**：LLM 生成的回答可能是错的（hallucination），训练后强化错误
- **数据污染**：可能生成训练分布外的"奇怪"内容
- **过滤成本**：需要后续过滤、人工抽查
- **重复**：自然语言的句法多样性有限，容易生成相似数据

应对——用更强的 LLM（如 GPT-4）生成 + 多轮过滤 + 人工抽样质检。

**Q：用 GPT-4 生成 SFT 数据训自己的模型，存在什么风险？**

A：
- **法律风险**：OpenAI 的 ToS 禁止用 GPT 输出训竞争模型（但实际执行难）
- **能力上限**：模型最多达到生成数据的来源模型水平——**徒弟难超过师傅**（除非加 RLHF/RLAIF 等增强）
- **风格继承**：会继承 GPT 的写作风格、价值观、拒答偏好
- **数据污染**：GPT 的训练数据可能有偏差，传递给学生模型

业界做法：
- 大公司：自建数据（开源模型 + 人工）
- 中小团队：用 Mistral / LLaMA 这类开源模型生成数据（无法律限制）
- 强烈不推荐用 GPT-4 数据训商业模型

**Q：怎么衡量自动生成数据的质量？**

A：几个工具——
- **困惑度过滤**：用 base model 给每条数据打 PPL，过滤太奇怪的
- **LLM-as-judge**：用强模型打分
- **聚类多样性**：用 sentence embedding + KMeans，看聚类数和均衡度
- **人工抽样**：1% 数据人工审核
- **最终下游任务**：用生成数据训完，看 benchmark

### ⚠️ 常见陷阱

1. 不要只说 Self-Instruct——要懂演化路线（Self-Instruct → Evol-Instruct → Magpie）
2. Magpie 是 2024 新方法——跟上前沿能加分
3. 要懂自动生成的"局限"——不能盲信

### 🏢 大厂偏好

- **数据团队**：必问，可能让你设计一个完整的数据 pipeline
- **字节 / DeepSeek**：可能问"如何提升数据多样性"

### 📚 延伸阅读
- [Self-Instruct](https://arxiv.org/abs/2212.10560)
- [WizardLM (Evol-Instruct)](https://arxiv.org/abs/2304.12244)
- [Magpie](https://arxiv.org/abs/2406.08464)

---

## Q4：Loss Masking 的实现与必要性 ⭐⭐⭐⭐⭐（易答错）

### 🎯 一句话标答

> SFT 训练时**只对 response 部分算 loss**，instruction 部分的 loss 被 mask 掉——避免让模型学"生成问题"（meaningless），让它专注学"生成答案"（关键）。

### 🗣️ 30 秒口语版

"这是 SFT 实现的核心细节，很多新手会忽略。

**问题场景**：SFT 数据是 `[INSTRUCTION] + [RESPONSE]` 的串联。如果直接算 next-token prediction loss——

模型会**同时学**：
- 给定 [前 i 个 token]，预测第 i+1 个 token（在 instruction 里）→ **学'生成问题'，没意义**
- 给定 [INSTRUCTION + 前 j 个 response token]，预测下一个 response token → **学'答问题'，是目标**

前者是噪声，后者是信号。混在一起会**稀释训练信号**，效果差。

**Loss Masking 的做法**：
- 把 instruction 部分的 label 设为 -100（PyTorch 的忽略值）
- `F.cross_entropy(logits, labels, ignore_index=-100)` 自动跳过这些位置
- 只有 response 部分参与 loss 计算

**效果对比**：
- 不做 mask：训得慢，质量差（loss 信号被稀释 50%+）
- 做 mask：训得快，质量好

多轮对话的处理更复杂——见 Q5。"

### 💻 实现示例

**单轮 SFT 数据**：

```python
instruction = "请翻译: hello"
response = "你好"

# 拼接
full_text = instruction + response

# Tokenize
tokens = tokenizer(full_text).input_ids  # 假设 [101, 102, 103, 104, 105, 106]

# 构造 labels
inst_len = len(tokenizer(instruction).input_ids)  # 假设 4
labels = tokens.copy()
labels[:inst_len] = [-100] * inst_len  # mask instruction 部分
# labels = [-100, -100, -100, -100, 105, 106]

# Loss 计算
loss = F.cross_entropy(logits, labels, ignore_index=-100)
# 只对位置 4, 5 算 loss
```

**多轮对话**的处理：
- A. 只算最后一轮 assistant：简单
- B. 算所有 assistant 轮：信号更密，主流做法

### 🔥 高频追问 Top 3

**Q：为什么 -100 是 PyTorch 的默认忽略值？**

A：纯约定俗成——PyTorch 的 `F.cross_entropy` 和 `nn.CrossEntropyLoss` 默认 `ignore_index=-100`。选 -100 是因为：
- 必须是负数（label 是非负 token ID，所以负数一定是"非法 label"，安全）
- -100 足够"远离"任何可能的 token ID

也可以自己设其他值（如 -1），但默认 -100 不要乱改，保持习惯。

**Q：能不能给 instruction 一个小的 loss 权重（比如 0.1）而不是完全 mask？**

A：可以尝试，但**不主流**。完全 mask 的逻辑很清晰：instruction 是用户输入，模型不需要学。给小权重的逻辑是"让模型也对 instruction 有点感知"，但实际效果**没明显增益**，反而引入超参（权重选多少？）。

业界主流：**完全 mask instruction**。

**Q：多轮对话怎么做 Loss Masking？**

A：多轮数据格式：
```
<user>问1<assistant>答1<user>问2<assistant>答2...
```

两种策略——

**策略 A：只对最后的 assistant 算 loss**——简单但浪费前面信息

**策略 B：对所有 assistant 轮次算 loss**（主流）

```python
labels = []
for token, role in zip(tokens, roles):
    if role == "user" or role == "system":
        labels.append(-100)
    else:  # assistant
        labels.append(token)
```

### ⚠️ 常见陷阱

1. 必须用 ignore_index=-100，不要乱改
2. Tokenize 时要单独算 instruction 长度：拼接后再 tokenize 和分开 tokenize 长度可能不同（subword 合并问题），要小心
3. 多轮对话用策略 B（对所有 assistant 算 loss）

### 🏢 大厂偏好

- **训练工程师岗**：必问
- **字节 / Meta**：可能会让你现场写多轮对话的 mask 实现

---

## Q5：Chat Template 与多轮对话格式 ⭐⭐⭐⭐

### 🎯 一句话标答

> Chat Template 是模型识别**对话角色**的特殊格式——用 `<|user|>`, `<|assistant|>` 等特殊 token 标记每段话的发起者；**不同模型的 template 不同，训练和推理必须一致**，否则模型会乱套。

### 🗣️ 30 秒口语版

"Chat Template 是工程落地的关键细节，很多 bug 都在这里。

**核心问题**：base model 是续写器，不知道'用户'和'助手'的区别。要让它学对话，必须用**特殊 token 显式标记角色**——这就是 chat template。

每个模型的 template 不同——LLaMA-3 用 `<|start_header_id|>` 包围角色名，Qwen 用 `<|im_start|>...<|im_end|>`，DeepSeek 用 `User:` 和 `Assistant:` 加特殊 token。

**关键原则**：
- 训练用什么模板，推理就用什么模板——一致性是铁律
- 通常用 `tokenizer.apply_chat_template()` 自动处理
- 自己拼字符串容易出错（漏特殊 token、空格不对）

**为什么不同模型 template 不同**？历史原因——各家在 SFT 时自定义了格式，已经训成了肌肉记忆，没法统一。所以使用任何模型前必须查它的 template。

**常见 bug**：训练用 ChatML 格式（Qwen 的），推理时用了 LLaMA 模板，模型表现像 base model（不会回答），就是 template 不匹配。"

### 📐 主流模型的 Chat Template

**LLaMA-3**：
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>
```

**Qwen-2.5 (ChatML)**：
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

**DeepSeek-V3**：
```
<|begin_of_sentence|>You are a helpful assistant.

User: Hello!Assistant: Hi there!
```

注意 LLaMA-3 在角色名后有**两个换行**，Qwen 是**一个换行**——细节错就崩。

### 🔥 高频追问 Top 3

**Q：如果训练和推理用了不同模板会怎样？**

A：**模型表现会"退化到 base model"**——可能不会停止生成、不会按 assistant 角色回答、混用角色等。表现糟糕但又不会完全错——这种 silent failure 是常见的 bug 来源。

调试技巧：把训练时的一条数据完整 print 出来，再把推理的输入完整 print 出来，**逐字符对比**。

**Q：怎么自定义 chat template？**

A：在 HuggingFace 上，每个模型的 `tokenizer_config.json` 里有 `chat_template` 字段，是 Jinja2 模板：

```python
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")
print(tokenizer.chat_template)  # 看官方模板
```

自定义时直接覆盖这个字段。**强烈建议用官方模板，不要自己造**。

**Q：System prompt 怎么处理？**

A：大部分现代模型支持 system role，放在最前面：

```
<|im_start|>system
你是一个有用的助手...<|im_end|>
<|im_start|>user
...<|im_end|>
```

实践要点：
- System prompt 通常**不算 loss**（和 user 一样 mask）
- 训练时要混合**有/无 system prompt** 的数据，让模型学会处理两种情况
- 不要给太长的 system prompt——会挤占有效上下文

### ⚠️ 常见陷阱

1. **训练推理 template 不一致**——最常见 bug
2. **手动拼接字符串容易漏特殊 token**——用 `apply_chat_template`
3. **EOS token 的处理**：训练时是否在 response 后加 EOS 决定模型是否会停止生成

### 🏢 大厂偏好

- **训练工程师 / SFT 团队**：必问
- **AI 部署岗**：会问推理时如何确保 template 一致

---

## Q6：SFT 中的灾难性遗忘 ⭐⭐⭐⭐

### 🎯 一句话标答

> SFT 用窄分布的指令数据训练，会让模型**遗忘预训练中学到的通用知识**——通过**数据混合、低 lr、早停、PEFT** 等手段缓解。

### 🗣️ 30 秒口语版

"灾难性遗忘是 SFT 的隐形杀手。

**现象**：用某个领域 SFT 数据微调后，模型在该领域表现提升，但在其他领域（如基础知识、推理、代码）显著下降。原因——

**SFT 数据分布窄**：通常只覆盖部分任务类型。
**梯度方向偏**：训练时只学新数据，原有能力对应的参数被'冲走'。
**学习率敏感**：lr 太大或训练步数太多，加速遗忘。

**应对策略**：

**1. 数据混合**：把通用数据混入 SFT 集（如 OpenHermes 等通用 SFT 集），保持广度。
**2. 较低学习率**：通常 1e-5 到 5e-5，比预训练小一个量级。
**3. 早停**：监控通用 benchmark（如 MMLU），下降超过阈值就停。
**4. PEFT**：LoRA 等只调少量参数，对原模型影响小。
**5. 重放（Replay）**：少量预训练数据混入 SFT。

**实际经验**：
- 全参微调 + 单一领域数据：遗忘严重
- 全参微调 + 通用数据混合：遗忘适中
- LoRA + 单一领域：遗忘很少（参数冻结）

DeepSeek-V3 等大模型 SFT 时混合大量通用对话数据，就是为防遗忘。"

### 🔥 高频追问 Top 3

**Q：怎么定量评估遗忘？**

A：**遗忘曲线监控** —— SFT 训练过程中定期评估一组**保留 benchmark**：
- 通用知识：MMLU
- 推理：GSM8K, MATH
- 代码：HumanEval
- 对话：MT-Bench

如果某个指标下降超过阈值（如 5%），可能在过拟合特定领域。

**Q：LoRA 真的比全参微调更抗遗忘吗？**

A：**通常是**。LoRA 只更新少量低秩参数，**原始权重几乎冻结**——大部分预训练知识保留。

但代价是表达力上限低，对**完全新领域**（如模型未见过的语言、代码语法）效果不如全参。

经验：
- 新任务/新格式：LoRA 够用
- 新领域/新语言：可能需要全参 + 数据混合

**Q：能用什么方法"恢复"已经遗忘的能力？**

A：几条路——
- **混合预训练数据继续训**：用少量原始预训练数据 + 新 SFT 数据
- **从 base model 重新做 SFT**：抛弃已经过拟合的版本，用更好的数据配方
- **多任务 SFT**：一次性混合多个领域的数据
- **模型融合**：LoRA 多个版本融合，平衡不同能力

### ⚠️ 常见陷阱

1. 不要只看 SFT loss——要看通用 benchmark 变化
2. 学习率太大是遗忘主因——SFT 用 1e-5 量级
3. PEFT 抗遗忘但有上限——不是万能药

---

## Q7：SFT 超参选择经验 ⭐⭐⭐

### 🎯 一句话标答

> SFT 超参经验：**学习率 1e-5 到 5e-5**，**batch size 32-128**，**1-3 epochs**，**warmup 3-10%**，**bf16 + AdamW**——比预训练保守得多。

### 🗣️ 30 秒口语版

"SFT 超参的核心原则是'**比预训练保守**'——

**学习率**：1e-5 ~ 5e-5。预训练用 1e-4 ~ 6e-4，SFT 小一个量级。lr 太大会破坏预训练学到的表示。

**Batch size**：32-128（global batch）。比预训练小很多——SFT 数据量小，大 batch 没意义。

**Epoch 数**：1-3。**不要超过 3**——过拟合迅速。LIMA 用 15 epoch（数据极少），是特例。

**Warmup**：总步数的 3-10%。SFT 步数少，warmup 也短。

**优化器**：AdamW，β1=0.9, β2=0.95, weight_decay=0.1。和预训练一致。

**精度**：bf16 混合精度。

**LR Schedule**：Cosine 或 Linear decay 到 0。

**最大长度**：根据数据，通常 2K-8K。

**Gradient Clipping**：max_norm=1.0。

实战经验——SFT 超参不是太敏感，**只要 lr 不太大、epoch 不太多，结果都差不多**。质量主要看数据。"

### 🔍 SFT vs 预训练超参对比

| 超参 | 预训练 | SFT |
|---|---|---|
| 学习率 | 1e-4 ~ 6e-4 | 1e-5 ~ 5e-5 |
| Batch size (global) | 1M+ tokens | 32-128 samples |
| Epoch | 1（数据多到训不完） | 1-3 |
| Warmup | 0.3-3% | 3-10% |
| Sequence length | 2K-8K | 2K-32K |
| 优化器 | AdamW | AdamW（同） |

### 🔥 高频追问 Top 3

**Q：SFT epoch 数怎么定？**

A：**通常 2-3**。
- 1 epoch：可能欠拟合（数据没学透）
- 2-3 epoch：通常最优
- 4+ epoch：开始过拟合，遗忘加剧

监控**验证集 loss** 决定何时停。如果训练 loss 持续下降但验证 loss 上升，立即停。

**Q：lr 太大会有什么具体表现？**

A：
- 训练初期 loss 飙升（梯度 spike）
- 模型输出变得"奇怪"——重复、乱码
- 通用 benchmark 大幅下降
- 即使最终 loss 收敛，效果也比小 lr 差

如果观察到这些，**立即调小 lr** 重新训。

**Q：长上下文 SFT 有什么特殊处理？**

A：
- **序列长度**：根据应用场景，可能 8K-128K
- **Sequence Parallel**：需要 SP 分担激活显存
- **数据**：要包含真实长序列，不能只 padding 短序列
- **lr 可能要更小**：长序列梯度方差大

### 🏢 大厂偏好

- 这题是**铺垫题**，会作为更深入超参/工程问题的起点

---

## Q8：Instruction Tuning vs SFT 的细微区别 ⭐⭐⭐

### 🎯 一句话标答

> **Instruction Tuning** 是 2021 年 FLAN 提出的概念——把多种 NLP 任务转换成指令格式，让模型学会泛化到新指令；**SFT** 是更广义的微调流程。**Instruction Tuning ⊂ SFT** 的一种形式。

### 🗣️ 30 秒口语版

"这两个词常被混用，但严格说有区别——

**Instruction Tuning（FLAN, 2021）**：
- Google 提出的概念
- 把**已有的 NLP 任务**（分类、问答、翻译...）**重写成自然语言指令**
- 比如：把 SST-2 分类任务改成"判断这句话情感：'今天天气很好'"
- 用大量这种任务训练，模型学会**指令遵循的一般能力**
- **关键**：用大量已标注的 NLP 任务数据

**SFT（更广义）**：
- 任何用指令格式做的有监督微调
- 数据可以是：自然指令、对话、特定领域任务
- 数据来源：人工写、自动生成、从对话日志收集
- 不一定来自传统 NLP 任务

简而言之——**Instruction Tuning 是 SFT 的早期形态**。FLAN 启发了后来的 Alpaca / Vicuna 等用 LLM 生成数据的 SFT 路线。

到 GPT-4 / Claude 3 时代，**SFT 已经不限于 NLP 任务**，扩展到对话、写作、代码、推理——所以现在大家说 SFT 更多，instruction tuning 这个词渐渐淡化。"

### 🔍 演化关系

```
FLAN (2021)
  ├─ Instruction Tuning 概念
  ├─ 用 NLP 任务模板化
  └─ T5/PaLM 上验证

  ↓ 启发

InstructGPT (2022)
  ├─ OpenAI 用真实用户指令
  ├─ SFT + RLHF
  └─ ChatGPT 前身

  ↓ 扩展

Alpaca/Vicuna (2023)
  ├─ Self-Instruct 生成数据
  ├─ Llama base + SFT
  └─ 开源仿制 ChatGPT

  ↓ 完善

现代 LLM Pipeline
  Pretrain → SFT (含多种数据) → RLHF/DPO
```

### 🔥 高频追问 Top 3

**Q：FLAN 和现代 SFT 最大区别是什么？**

A：**数据来源**——
- FLAN：基于已有 NLP 数据集（分类、问答等），人工写指令模板
- 现代 SFT：自然指令（用户真实问题），无固定模板，覆盖更宽

效果上现代 SFT 模型更"像人"——能处理开放式问题，不只是固定任务。

**Q：Instruction Tuning 现在还有用吗？**

A：**有用，但融入了 SFT**。现代 SFT 数据集（如 OpenHermes）通常包含——
- FLAN 风格的传统任务（分类、QA、摘要等）
- 自然对话指令
- 代码、数学等专门数据

两者**混合使用**效果最好。

**Q：Pre-training + Instruction Tuning 能不能跳过 RLHF？**

A：**FLAN 时代是的**——只用 Instruction Tuning，没有 RLHF。但现代 LLM 已经证明：**RLHF 后的对话质量显著高于纯 SFT**。所以现在的 pipeline 几乎都是 SFT + RLHF/DPO。

### 🏢 大厂偏好

- 这题是**铺垫题**，更多体现你的概念清晰度

### 📚 延伸阅读
- [FLAN](https://arxiv.org/abs/2109.01652) - Instruction Tuning 的起源
- [InstructGPT](https://arxiv.org/abs/2203.02155) - 现代 SFT 的范本

---

## 📝 本章小结

| 关键点 | 你必须能脱口而出 |
|---|---|
| SFT 本质 | next-token prediction，只是数据是指令-响应对 |
| LIMA 数字 | 1000 条精选 ≈ 52000 条 Alpaca |
| 表面对齐假说 | 知识来自预训练，SFT 只是激活格式 |
| 数据生成路线 | Self-Instruct → Evol-Instruct → Magpie |
| Loss Masking | 用 -100 mask instruction，只算 response loss |
| Chat Template | 训练推理必须一致，否则模型崩 |
| 灾难性遗忘 | 混合通用数据 + 小 lr + 早停 |
| SFT 超参 | lr 1e-5~5e-5, batch 32-128, 1-3 epoch |
| Instruction Tuning | FLAN 提出，SFT 早期形态 |

## ✅ 自测题

1. 用 50 行 PyTorch 代码实现一个带 Loss Masking 的 SFT 训练 step（单轮和多轮各一版）
2. 解释 LIMA 的"表面对齐假说"，并讨论它在小模型上是否成立
3. 比较 Self-Instruct、Evol-Instruct、Magpie 三种数据生成方法的优劣，并选一种为你的场景设计 pipeline
4. 如果你训完 SFT 发现 MMLU 下降了 10%，怎么排查和恢复？
5. LLaMA-3 和 Qwen-2.5 的 chat template 有什么区别？混用会出什么 bug？

---

> **下一章预告**：第 8 章｜PEFT 与 LoRA 系列——LoRA 推导是必考题，QLoRA 的 NF4 + Double Quantization 是 DeepSeek 风格的细节题。
