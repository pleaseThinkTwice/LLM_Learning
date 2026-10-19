# 第 7 章｜SFT 指令微调

> **本章定位**：SFT 把一个"续写器"变成一个"助手"。本章围绕三个问题展开：**数据**（要多少？从哪来？LIMA 的"表面对齐假说"在多大程度上成立？）、**实现**（chat template、loss masking、packing、EOS 的细节）、**边界**（SFT 能不能教新知识？和 RL 相比学到的是什么？）。近两年的研究对这几个问题都给出了比"质量胜过数量"更细致的答案。

> **配套章节**：预训练与继续预训练 → 第 6 章 / LoRA → 第 8 章 / RLHF → 第 9 章 / 推理模型的蒸馏与 RL → 第 11 章 / 分词与特殊 token → 第 4 章

> **标记约定**：📄 论文结论 / 🧮 可推导 / 💭 直觉或有争议 / ❓ 开放问题

---

## 📌 本章导航

| 序号 | 题目 | 难度 | 类型 |
|---|---|---|---|
| Q1 | SFT 是什么？和预训练、RL 的关系 | ⭐⭐⭐ | 基础 |
| Q2 | **SFT 数据：LIMA 与表面对齐假说（及其反驳）** | ⭐⭐⭐⭐⭐ | 必问 |
| Q3 | 合成数据：Self-Instruct / Evol-Instruct / Magpie / 蒸馏 | ⭐⭐⭐⭐ | 数据工程 |
| Q4 | **Loss masking：为什么、怎么做、一定要做吗** | ⭐⭐⭐⭐⭐ | 易答错 |
| Q5 | Chat template、EOS 与 packing | ⭐⭐⭐⭐ | 工程 |
| Q6 | 遗忘、对齐税与"SFT 能否教新知识" | ⭐⭐⭐⭐ | 研究 |
| Q7 | SFT 超参数 | ⭐⭐⭐ | 工程 |
| Q8 | Instruction tuning 的历史：FLAN → InstructGPT → Alpaca | ⭐⭐⭐ | 概念 |
| Q9 | 研究前沿：数据选择、SFT vs RL、推理蒸馏 | ⭐⭐⭐⭐⭐ | 研究视角 |

---

## Q1：SFT 是什么？和预训练、RL 的关系 ⭐⭐⭐

### 🎯 一句话标答

> SFT（监督微调）在"指令 → 回答"格式的数据上继续做 next-token prediction，通常**只在回答部分计算 loss**。目标函数和预训练相同，区别在于数据形态和目的：让模型学会对话格式、遵循指令、以期望的风格输出。它通常是 RLHF / RL 之前的一步，但 📄 DeepSeek-R1-Zero 证明了在特定任务上可以**跳过 SFT 直接做 RL**。

### 🔍 对比

| 维度 | 预训练 | SFT | RLHF / RL |
|---|---|---|---|
| 目标 | next-token | next-token（通常只在回答部分） | 最大化奖励（带 KL 约束） |
| 数据 | 无标注文本，T 级 token | 指令–回答对，千到百万条 | 偏好对 / 可验证奖励 |
| 数据来源 | 网页、书籍、代码 | 人工撰写、合成、蒸馏 | 模型**自己生成**的回答 |
| 学到什么 | 语言、知识 | 格式、风格、行为模式 | 在自身分布上优化偏好与正确性 |

### 🔥 高频追问

**Q：SFT 之后为什么还要 RLHF？**

A：
- 💭 SFT 是模仿学习：模型学习**示范数据**的分布，上限受示范者水平限制，而且只见过"好的回答"，没见过"坏的回答"。
- RLHF 用的是**比较**信号（A 比 B 好），对标注者来说比写出完美回答更容易；并且训练数据来自模型**自己的**输出（on-policy），能纠正模型自己特有的错误。
- 📄 InstructGPT（Ouyang et al. 2022）中，1.3B 的 InstructGPT（SFT + RLHF）在人类偏好评估中胜过 175B 的 GPT-3；并且在各个模型规模上，PPO 模型都比同规模的 SFT 模型更受偏好。

**Q：能不能跳过 SFT 直接做 RL？**

A：
- 📄 **DeepSeek-R1-Zero**（2025）直接在 base 模型 DeepSeek-V3-Base 上用基于规则的奖励（答案正确性 + 格式）做 RL，推理能力大幅提升，并自发出现了长链思考、自我验证等行为。
- 但 R1-Zero 存在**可读性差、语言混杂**等问题，因此最终的 DeepSeek-R1 仍然先用少量"冷启动"长思维链数据做 SFT，再做 RL（→ 第 11 章）。
- 💭 结论：在**有可验证奖励的任务**上可以跳过 SFT；对开放式对话等任务，SFT 仍是让 RL 有一个合理起点的常用方法。

> ❌ 旧版说"几乎不行，必须先 SFT"，在 2025 年之后已不成立。

**Q：有没有把 SFT 和偏好优化合并的方法？**

A：📄 ORPO（Hong et al. 2024）在 SFT 的 loss 上加一个基于**几率比**的偏好项，一个阶段同时完成指令学习和偏好对齐，不需要参考模型（→ 第 10 章）。

### 📚 出处
- Ouyang et al. *Training language models to follow instructions with human feedback*（InstructGPT）. NeurIPS 2022. arXiv:2203.02155
- DeepSeek-AI. *DeepSeek-R1*. 2025. arXiv:2501.12948
- Hong et al. *ORPO*. EMNLP 2024. arXiv:2403.07691

---

## Q2：SFT 数据：LIMA 与表面对齐假说（及其反驳） ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 📄 LIMA（Zhou et al., NeurIPS 2023）只用 **1000 条**精选数据微调 LLaMA 65B，在人类评估中有 43% 的情况与 GPT-4 持平或更好（对 Bard 为 58%，对 DaVinci-003 为 65%）。作者据此提出**表面对齐假说**：知识和能力几乎都来自预训练，对齐只是学习与用户交互的"格式和风格"。📄 但后续研究发现，对**数学、多跳推理**等任务，性能随 SFT 数据量呈**幂律增长**，并未很快饱和（Raghavendra et al. 2024），说明这个假说至少是过度简化的。

### 🗣️ 30 秒口语版

"LIMA 是 Meta 2023 年的工作。他们从 Stack Exchange、wikiHow、Reddit 等社区挑了约 750 条高质量问答，再手写约 250 条，一共 1000 条，微调 LLaMA 65B。人类评估中，LIMA 有 43% 的时候被认为不差于 GPT-4，65% 的时候不差于用 RLHF 训练过的 DaVinci-003。

由此他们提出表面对齐假说：模型的知识基本都在预训练里，对齐只是教它用什么格式和风格说话，所以少量高质量数据就够了。

但这个假说后来受到了挑战。2024 年 Scale AI 的研究发现，数学、代码、多跳推理等任务的性能随 SFT 样本数呈幂律提升，少量样本主要教会了风格，并没有让推理能力饱和。所以更准确的说法是：**少量数据就能教会格式，但能力的提升仍然需要数据量**。工业界的 SFT 数据量也确实大得多，比如 LLaMA 2-Chat 用了约 2.7 万条高质量标注，DeepSeek-V3 用了 150 万条。"

### 📄 LIMA 的细节

- 模型：LLaMA 65B。
- 数据：约 750 条来自社区问答（Stack Exchange、wikiHow、Reddit 写作板块等，人工筛选和编辑），约 250 条由作者撰写（包括一些多样性提示和安全相关的示例）。
- 训练：15 个 epoch，学习率 $10^{-5}$ 线性衰减到 $10^{-6}$，并在残差连接上使用逐层递增的 dropout。
- 评估：人类偏好评估 + GPT-4 作为评判的偏好评估，两者结论一致。
- 消融：在数据量相同的情况下，**提高质量和多样性**比单纯增加数量更有效；在 Stack Exchange 数据上把数量翻倍到 2000、4000 条并没有带来提升。

> ❌ 旧版的 LIMA 对比表格（对 Alpaca 胜率 62%、对 GPT-4 胜率 19% 平局 38% 等）与论文摘要的数字不符，已替换。

### 📄 支持与反驳

| 观点 | 证据 |
|---|---|
| 支持：对齐主要改变风格 | 📄 **URIAL**（Lin et al., ICLR 2024, *The Unlocking Spell on Base LLMs*）：比较 base 模型和对齐后模型的 token 分布，发现差异主要集中在少数**风格类 token**（如话语标记、安全声明）上；仅用 3 个精心设计的上下文示例 + 系统提示，base 模型就能达到接近 SFT/RLHF 模型的对话质量 |
| 反驳：推理能力随数据持续提升 | 📄 **Raghavendra et al. 2024**（*Revisiting the Superficial Alignment Hypothesis*）：在 Llama-2/3、Mistral 上，数学、代码、指令遵循、多跳推理的性能随微调样本数呈幂律增长；少量样本主要教风格，推理能力随数据显著提升 |
| 推理蒸馏的"少量样本" | 📄 **s1**（Muennighoff et al. 2025）：1000 条精选的推理轨迹微调 Qwen2.5-32B-Instruct，配合"预算强制"，取得了很强的数学推理表现；用全部 5.9 万条数据训练并没有显著优于这 1000 条。📄 **LIMO**（Ye et al. 2025）用 817 条样本得到类似结论 |

💭 一种调和的解释：当 base 模型**已经具备**某种能力时，少量高质量数据就能把它"激活"出来（LIMA、s1、LIMO 使用的都是很强的 base 模型）；当能力需要真正被**学习**时，数据量依然重要。

### 🔥 高频追问

**Q：工业界的 SFT 数据量是多少？**

A：
- 📄 InstructGPT：SFT 数据约 1.3 万条提示。
- 📄 LLaMA 2-Chat：作者发现第三方 SFT 数据多样性和质量不足，最终使用了 **27,540 条**自有的高质量标注，并发现"少量高质量数据就足够"。
- 📄 DeepSeek-V3：约 **150 万条**，覆盖多个领域；其中推理数据由 DeepSeek-R1 生成后经过筛选，非推理数据由 DeepSeek-V2.5 生成并由人工核验。
- 📄 Tülu 3（AI2, 2024）：公开了完整的 SFT 数据混合（约 94 万条）和消融实验，是研究 SFT 数据配方的好资料。

**Q：怎么评估 SFT 数据质量？**

A：
- 多样性：用 embedding 聚类看覆盖面；按任务类型、难度、长度统计分布。
- 正确性：对数学、代码等可验证任务，用执行或答案匹配做过滤（拒绝采样）。
- LLM 打分：📄 AlpaGasus（Chen et al. 2024）用 ChatGPT 给 Alpaca 的 52K 数据打分，只保留约 9K 条高分数据，训练效果反而更好。
- 难度/价值：📄 IFD（Li et al. 2024, *From Quantity to Quality*）用"有指令时和无指令时回答的困惑度之比"衡量一条数据对模型的难度，挑选高 IFD 的样本。

### 📚 出处
- Zhou et al. *LIMA: Less Is More for Alignment*. NeurIPS 2023. arXiv:2305.11206
- Lin et al. *The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning*. ICLR 2024. arXiv:2312.01552
- Raghavendra et al. *Revisiting the Superficial Alignment Hypothesis*. 2024. arXiv:2410.03717
- Muennighoff et al. *s1: Simple test-time scaling*. 2025. arXiv:2501.19393
- Ye et al. *LIMO: Less is More for Reasoning*. 2025. arXiv:2502.03387
- Touvron et al. *Llama 2*. arXiv:2307.09288（§3.1）
- Lambert et al. *Tülu 3*. 2024. arXiv:2411.15124
- Chen et al. *AlpaGasus*. ICLR 2024. arXiv:2307.08701
- Li et al. *From Quantity to Quality*. NAACL 2024. arXiv:2308.12032

---

## Q3：合成数据：Self-Instruct / Evol-Instruct / Magpie / 蒸馏 ⭐⭐⭐⭐

### 🎯 一句话标答

> 📄 **Self-Instruct** 从 175 条种子任务出发，让模型生成新指令和回答，再过滤去重；📄 **Evol-Instruct**（WizardLM）用模型把已有指令"进化"得更复杂、更多样；📄 **Magpie** 只给对齐模型输入"用户回合的开头"，让它自己续写出用户问题，再生成回答，不需要种子数据。更普遍的做法是**用强模型蒸馏**：用更强的模型生成回答，再经过过滤。

### 📐 Self-Instruct（Wang et al. 2023）

```
1. 种子：175 条人工编写的任务（每条包含指令和示例）
2. 从任务池中采样 8 条（6 条人工 + 2 条模型生成）作为示例，让模型生成新指令
3. 判断新指令是否为分类任务（决定先生成输入还是先生成输出）
4. 为新指令生成输入–输出实例
5. 过滤：与已有指令的 ROUGE-L > 0.7 的丢弃；过滤无法由语言模型完成的指令（如涉及图片）
6. 加入任务池，迭代
```

📄 Stanford Alpaca 使用类似流程，用 text-davinci-003 生成了 52K 条数据微调 LLaMA 7B。

### 📐 Evol-Instruct（Xu et al. 2023, WizardLM）

- **深度进化**：增加约束、深化问题、具体化、增加推理步骤、使输入更复杂。
- **广度进化**：在同一领域生成全新的、更罕见的指令。
- **淘汰**：去掉进化失败的指令（例如没有信息增量、模型无法回答）。

### 📐 Magpie（Xu et al. 2024）

```
输入：<|start_header_id|>user<|end_header_id|>\n\n      ← 只给"用户回合"的模板前缀
输出：（模型自动续写出一条用户问题）
然后：把这条问题按正常格式输入，让模型生成回答
```

💡 原理：对齐模型在训练中见过大量"用户回合"，因此在用户位置上续写时，会从它学到的**用户问题分布**中采样。📄 作者用 Llama-3-Instruct 生成了数百万条数据，筛选后的子集微调 Llama-3-8B-Base，在 AlpacaEval 等对齐基准上可以与官方的 Llama-3-8B-Instruct 相当。

### 🔥 高频追问

**Q：用强模型的输出训练自己的模型，有什么风险？**

A：
- 📄 **模仿的假象**：Gudibande et al. 2023（*The False Promise of Imitating Proprietary LLMs*）发现，用 ChatGPT 输出训练的模仿模型能很好地学到**风格**，人类评估者也常常被骗，但在事实性和基础能力上与原模型差距没有缩小。结论是"想要缩小能力差距，需要更好的 base 模型，而不是模仿数据"。
- **错误传递**：教师模型的幻觉和错误会被学生学走，因此需要验证（可执行代码、可验证答案）和过滤。
- **许可**：部分商业模型的服务条款禁止用其输出开发竞争模型，需要注意合规。
- **学生能否超过老师**：📄 Burns et al. 2023（OpenAI, *Weak-to-Strong Generalization*）发现，用弱模型的标签训练强模型时，强模型的表现可以**超过**弱模型的标签水平，但还达不到用真实标签训练的水平。

**Q：合成数据怎么保证多样性？**

A：常用方法：用多样的种子（人物设定、主题分类、难度）驱动生成，例如 📄 Persona Hub（Ge et al. 2024）用 10 亿个合成"人物设定"来生成多样化的指令；对生成结果做 embedding 去重；分领域控制配比。

### 📚 出处
- Wang et al. *Self-Instruct*. ACL 2023. arXiv:2212.10560
- Taori et al. *Stanford Alpaca*（2023，GitHub）
- Xu et al. *WizardLM*. ICLR 2024. arXiv:2304.12244
- Xu et al. *Magpie*. ICLR 2025. arXiv:2406.08464
- Gudibande et al. 2023. arXiv:2305.15717
- Burns et al. *Weak-to-Strong Generalization*. 2023. arXiv:2312.09390
- Ge et al. *Scaling Synthetic Data Creation with 1,000,000,000 Personas*. 2024. arXiv:2406.20094

---

## Q4：Loss masking：为什么、怎么做、一定要做吗 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 标准做法是把 system / user 部分的 label 设为 `-100`，**只在 assistant 回答部分计算 loss**，让模型专注学"怎么回答"而不是"怎么提问"。📄 但它不是金科玉律：Shi et al.（NeurIPS 2024）发现，当**指令很长、回答很短**或**训练样本很少**时，在指令部分也计算 loss 反而更好，原因是减少了对指令数据集的过拟合。

### 💻 实现（推荐基于 chat template 逐段构造）

```python
def build_example(messages, tokenizer):
    input_ids, labels = [], []
    for i, msg in enumerate(messages):
        # 渲染到当前消息为止的前缀，取增量部分，避免"整体 tokenize 与分段 tokenize 结果不一致"
        prefix = tokenizer.apply_chat_template(messages[:i], tokenize=False) if i else ""
        full = tokenizer.apply_chat_template(messages[:i + 1], tokenize=False)
        seg_ids = tokenizer(full[len(prefix):], add_special_tokens=False).input_ids
        input_ids += seg_ids
        labels += seg_ids if msg["role"] == "assistant" else [-100] * len(seg_ids)
    return input_ids, labels
```

要点：
1. **assistant 回答末尾的结束 token**（如 `<|eot_id|>`、`<|im_end|>`）必须参与 loss，否则模型学不会停止（Q5）。
2. assistant 回合的**角色头**（如 `<|im_start|>assistant\n`）通常不计算 loss，因为推理时它由模板提供。
3. 注意 BPE 合并可能跨越"指令/回答"边界：如果先拼接字符串再整体 tokenize，然后按字符长度切分 mask，边界附近可能错位。上面的做法以模板渲染出的段落为单位处理；更稳妥的方式是使用 tokenizer 的 offset mapping 或框架自带的实现（如 TRL 的 `assistant_only_loss`）。

### 🔥 高频追问

**Q：不做 masking 一定更差吗？**

A：不一定。
- 📄 **Shi et al., NeurIPS 2024**（*Instruction Tuning With Loss Over Instructions*）：提出"指令建模"（在指令和提示部分也计算 loss）。在 21 个基准上的实验表明，效果取决于两个因素：**指令长度与回答长度之比**、**训练样本数量**。在指令长、回答短的数据集上，或者只有少量样本（类似 LIMA 的设置）时，指令建模明显更好；作者分析认为原因是**减少了对 SFT 数据集的过拟合**。他们并不主张取代标准做法，而是给出了低资源场景下的建议。
- 📄 Huerta-Enochian & Ko 2024（*Instruction Fine-Tuning: Does Prompt Loss Matter?*）发现，对**回答很短**的数据，给提示部分一个较小的 loss 权重（而不是完全 0）有时能提升效果。

> ❌ 旧版说"不做 mask 训得慢、质量差（信号被稀释 50%+）"，没有出处，并且与上述研究不完全一致。

**Q：多轮对话怎么 mask？**

A：主流做法是**对所有 assistant 回合都计算 loss**（信号更多），而不是只算最后一轮。但要注意：如果前面的 assistant 回合质量较差（例如来自真实对话日志），可能会学到不好的行为，此时可以只在高质量回合上计算 loss。

**Q：loss 怎么归一化？**

A：和第 5 章 Q1 一样，要明确是"按 token 平均"还是"按样本平均"。按 token 平均时，长回答的权重更大；梯度累积和 packing 时要用全局的有效 token 数归一化，否则会出现 bug。

**Q：为什么 ignore_index 默认是 -100？**

A：PyTorch `CrossEntropyLoss` 的默认值。只要是一个不会出现的 token id（负数）即可；关键是和框架保持一致。

### 📚 出处
- Shi et al. NeurIPS 2024. arXiv:2405.14394
- Huerta-Enochian & Ko. 2024. arXiv:2401.13586

---

## Q5：Chat template、EOS 与 packing ⭐⭐⭐⭐

### 🎯 一句话标答

> Chat template 用特殊 token 标记角色和回合边界。**训练和推理必须使用完全相同的模板**（包括空格、换行、BOS）。SFT 必须让模型学会在回答结束时输出**结束 token**，否则推理时不会停止。Packing 可以减少 padding 浪费，但要处理好样本之间的注意力隔离。

### 📐 主流模型的模板

**LLaMA 3**：
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>
```

**Qwen（ChatML）**：
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

**DeepSeek-V3**（注意特殊 token 中的竖线和下划线是全角字符 `｜` 和 `▁`）：
```
<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>Hello!<｜Assistant｜>Hi there!<｜end▁of▁sentence｜>
```

**LLaMA 2-Chat**：
```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

Hello! [/INST] Hi there! </s>
```

> ❌ 旧版的 DeepSeek-V3 模板写成了 `User: Hello!Assistant:` 的纯文本形式（这更接近更早版本 DeepSeek 模型的格式）。DeepSeek-V3 使用 `<｜User｜>`、`<｜Assistant｜>` 特殊 token。**以 HF 仓库中 `tokenizer_config.json` 的 `chat_template` 为准**。

### 🔥 高频追问

**Q：模板不一致会出什么问题？**

A：💭 常见现象包括：不停止生成、重复角色标记、回答质量下降、安全行为失效。这类问题往往不会直接报错，所以排查时要把**训练样本**和**推理输入**都渲染成字符串并逐字符比较，再比较 token id。

**Q：EOS 有哪些坑？**

A：
- 回答末尾的结束 token 必须在 label 中保留（不能被 mask 成 -100）。
- 如果 pad token 与 EOS 共用同一个 id，并且用"label 中 pad 位置一律设为 -100"的方式处理，就会**把真正的 EOS 也 mask 掉**，导致模型学不会停止。这是一个很常见的 bug。
- 推理时的停止条件要包含模板中的回合结束 token（如 LLaMA 3 的 `<|eot_id|>`），而不只是 `<|end_of_text|>`。

**Q：Packing 怎么做？**

A：把多条样本拼接成一个固定长度的序列，减少 padding。
- **朴素 packing**：直接拼接，样本之间可以互相 attention，存在"跨样本污染"。
- **隔离 packing**：使用块对角的因果 mask（FlashAttention 的变长接口），并让每个样本的位置编码从 0 开始（第 1 章 Q8）。HF Transformers 和 TRL 支持这种方式（如 `DataCollatorWithFlattening`）。
- loss 归一化要按全局有效 token 数计算（Q4）。

**Q：system prompt 怎么处理？**

A：system 部分通常不计算 loss。训练数据中要包含有 system prompt 和没有 system prompt 的样本，以及多种不同的 system prompt，否则模型可能只在训练时见过的那种 system prompt 下表现正常。

---

## Q6：遗忘、对齐税与"SFT 能否教新知识" ⭐⭐⭐⭐

### 🎯 一句话标答

> SFT 数据分布窄，可能导致通用能力下降（**遗忘**）；对齐后在某些 NLP 基准上的性能下降被称为**对齐税**。📄 一个更微妙的问题是：通过 SFT 让模型学习它**原本不知道的事实**，学得很慢，而且会**增加幻觉**（Gekhman et al. 2024）。

### 📄 关键研究

- 📄 **对齐税**：InstructGPT 发现 RLHF 后模型在部分公开 NLP 数据集上的性能下降，提出 **PPO-ptx**：在 RL 训练中混入预训练梯度来缓解。
- 📄 **新知识与幻觉**：Gekhman et al., EMNLP 2024（*Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?*）在闭卷问答上做了受控实验：包含模型"不知道"的事实的样本，在微调中**学得比已知样本慢得多**；一旦这些样本最终被学会，模型产生幻觉的倾向会**线性增加**。作者认为模型主要通过预训练获得知识，微调更多是教它更高效地使用已有知识。
- 📄 **"遗忘"可能是表面的**：Zheng et al., ICLR 2025（*Spurious Forgetting in Continual Learning of Language Models*）发现，在连续指令微调中观察到的很多"遗忘"，实际上是**任务对齐**的丢失，而不是知识的丢失，用少量数据就能恢复。
- 📄 **LoRA 少学少忘**：Biderman et al. 2024（第 6 章 Q8，第 8 章）。

> ❌ 旧版说"DeepSeek-V3 等大模型 SFT 时混合大量通用对话数据，就是为了防遗忘"，没有出处，已删除。

### 🔥 高频追问

**Q：怎么缓解遗忘？**

A：
- 混入通用领域的 SFT 数据（多任务混合）。
- 降低学习率、减少 epoch。
- 使用 LoRA 等参数高效方法（以学得少为代价）。
- 混入少量预训练数据（replay）或在 RL 阶段加入预训练损失（PPO-ptx）。
- 模型融合：把领域模型和通用模型的权重做插值或更复杂的合并（如 TIES、DARE，→ 第 8 章）。

**Q：怎么监控遗忘？**

A：在训练过程中定期评估一组覆盖通用知识、推理、代码、对话、安全的基准，并与 base 模型对比。💭 阈值要根据业务需求设定，没有通用的"下降 5% 就停"规则。

### 📚 出处
- Ouyang et al. *InstructGPT*. arXiv:2203.02155
- Gekhman et al. EMNLP 2024. arXiv:2405.05904
- Zheng et al. *Spurious Forgetting in Continual Learning of Language Models*. ICLR 2025. arXiv:2501.13453
- Biderman et al. *LoRA Learns Less and Forgets Less*. TMLR 2024. arXiv:2405.09673

---

## Q7：SFT 超参数 ⭐⭐⭐

### 🎯 一句话标答

> SFT 的学习率比预训练小一到两个数量级（全参微调常见 $10^{-6}$ 到 $2\times10^{-5}$），训练 1–3 个 epoch（小数据集可以更多），通常不需要很大的 batch。最可靠的方法是参考公开配方，再在自己的数据和评测上做小范围扫描。

### 📄 公开配方

| 来源 | 学习率 | Epoch | 其他 |
|---|---|---|---|
| 📄 LLaMA 2-Chat | $2\times10^{-5}$，cosine | 2 | batch 64、序列长度 4096、weight decay 0.1；多条样本拼接，只在回答部分计算 loss |
| 📄 LIMA | $10^{-5}$ → $10^{-6}$ 线性 | 15 | 1000 条数据；逐层递增的残差 dropout |
| 📄 Tülu 3 | 不同规模分别调参 | 2 | 公开了完整超参数和消融 |

> 💭 Epoch 多少合适和数据量有关：LIMA 只有 1000 条数据，用了 15 个 epoch，并发现验证困惑度与生成质量不相关，因此根据人工评估选择 checkpoint。

### 🔥 高频追问

**Q：有什么简单有效的 SFT 技巧？**

A：📄 **NEFTune**（Jain et al., ICLR 2024）：训练时在输入 embedding 上加均匀噪声（幅度按 $\alpha/\sqrt{Ld}$ 缩放），在 AlpacaEval 上带来了明显提升，几乎不增加成本。作者分析认为它减少了对训练数据的过拟合，生成的回答更长、更详细。

**Q：学习率过大有什么表现？**

A：💭 通用基准大幅下降、输出重复或格式混乱、训练 loss 下降很快但生成质量变差。

---

## Q8：Instruction tuning 的历史：FLAN → InstructGPT → Alpaca ⭐⭐⭐

### 🎯 一句话标答

> 📄 FLAN（Wei et al. 2021）把 60 多个已有 NLP 数据集改写成自然语言指令，微调 137B 的 LaMDA-PT 后，在**未见过的任务类型**上的零样本能力明显提升，提出了"instruction tuning"的概念。📄 InstructGPT（2022）改用**真实用户提示**和人工撰写的回答做 SFT，再做 RLHF。📄 Alpaca（2023）则用合成数据低成本复现了指令遵循能力，开启了开源 SFT 的浪潮。

### 🔍 演化

| 时间 | 工作 | 关键点 |
|---|---|---|
| 2021 | 📄 FLAN（Google） | NLP 数据集 + 指令模板；在 LaMDA-PT 137B 上验证；小模型上 instruction tuning 反而有害，效果随规模出现 |
| 2021 | 📄 T0（Sanh et al.） | 多任务提示训练 T5，零样本泛化 |
| 2022 | 📄 InstructGPT（OpenAI） | 真实 API 用户提示；SFT → RM → PPO |
| 2022 | 📄 Flan-T5 / Flan-PaLM（Chung et al.） | 扩展到 1800+ 任务，加入思维链数据，系统研究了任务数量、模型规模的 scaling |
| 2023 | 📄 Alpaca、Vicuna | Self-Instruct 合成数据、ShareGPT 用户对话 |
| 2024– | 📄 Tülu 3、OLMo 2 等 | 完全开放的后训练配方（SFT + DPO + RLVR） |

### 🔥 高频追问

**Q：FLAN 式数据和现代 SFT 数据的区别？**

A：FLAN 式数据来自已有的标注数据集，任务边界清晰、回答通常很短；现代 SFT 数据更多是开放式的真实用户需求，回答更长、更详细。📄 Chung et al. 2022 发现两者可以互补，在 FLAN 混合中加入思维链数据，对推理任务有明显帮助。

### 📚 出处
- Wei et al. *Finetuned Language Models Are Zero-Shot Learners*（FLAN）. ICLR 2022. arXiv:2109.01652
- Sanh et al. *Multitask Prompted Training Enables Zero-Shot Task Generalization*（T0）. ICLR 2022. arXiv:2110.08207
- Chung et al. *Scaling Instruction-Finetuned Language Models*. 2022. arXiv:2210.11416
- Jain et al. *NEFTune*. ICLR 2024. arXiv:2310.05914

---

## Q9：研究前沿：数据选择、SFT vs RL、推理蒸馏 ⭐⭐⭐⭐⭐

### 方向 1：数据选择

- 📄 **LESS**（Xia et al., ICML 2024）：用低秩梯度特征估计每条训练数据对目标任务的影响（类似影响函数），只选 5% 的数据就能达到甚至超过用全部数据训练的效果。
- 📄 **Deita**（Liu et al., ICLR 2024）：综合复杂度、质量、多样性三个维度挑选数据。
- ❓ 开放问题：在一个模型上选出的数据，换一个模型是否依然最优？

### 方向 2：SFT 学到了什么，RL 学到了什么

- 📄 **Chu et al., ICML 2025**（*SFT Memorizes, RL Generalizes*）：在文本规则变体的纸牌算术游戏和视觉导航任务上，基于结果奖励的 RL 能泛化到未见过的规则和视觉变体，而 SFT 更倾向于记住训练数据。但 SFT 仍然有用：它稳定了输出格式，让后续 RL 能够发挥作用。
- 💭 这为"SFT → RL"的两阶段流程提供了一种解释：SFT 负责格式和起点，RL 负责泛化。

### 方向 3：推理蒸馏

- 📄 DeepSeek-R1 用 R1 生成的约 80 万条数据蒸馏 Qwen、Llama 系列的小模型，效果显著；论文还发现，对小模型而言，**蒸馏比直接对小模型做大规模 RL 更有效**（→ 第 11 章）。
- 📄 s1、LIMO：少量高质量推理轨迹 + 强 base 模型即可获得强推理能力（Q2）。
- ❓ 开放问题：蒸馏得到的能力和 RL 得到的能力在鲁棒性和泛化上有何不同？

### 🧪 动手验证思路

1. 用一个 1B 左右的 base 模型，分别用 1K、10K、100K 条 SFT 数据训练，在指令遵循（如 IFEval）和数学（如 GSM8K）上画出性能随数据量的曲线，检验表面对齐假说。
2. 在同一数据集上比较"只算回答 loss"和"指令也算 loss"，并按"指令/回答长度比"对数据分组分析（复现 Shi et al. 2024 的趋势）。
3. 故意制造"pad = EOS 且 mask 掉所有 pad"的 bug，观察模型是否学不会停止。
4. 实现带块对角 mask 的 packing，与朴素 packing 对比训练效果。

### 📚 出处
- Xia et al. *LESS*. ICML 2024. arXiv:2402.04333
- Liu et al. *What Makes Good Data for Alignment?*（Deita）. ICLR 2024. arXiv:2312.15685
- Chu et al. *SFT Memorizes, RL Generalizes*. ICML 2025. arXiv:2501.17161
- DeepSeek-AI. *DeepSeek-R1*. arXiv:2501.12948

---

## 📝 本章小结

| 关键点 | 你必须能脱口而出 | 性质 |
|---|---|---|
| SFT 本质 | next-token，通常只在回答部分算 loss | 🧮 |
| 跳过 SFT | DeepSeek-R1-Zero 在可验证任务上直接 RL；R1 仍用冷启动 SFT | 📄 |
| LIMA | 1000 条；对 GPT-4 43% 不差于；表面对齐假说 | 📄 |
| 假说的反驳 | 推理类任务随 SFT 数据量呈幂律提升 | 📄 |
| 合成数据 | Self-Instruct → Evol-Instruct → Magpie；模仿只学到风格 | 📄 |
| Loss masking | 标准做法；指令长/回答短/样本少时，指令也算 loss 可能更好 | 📄 |
| 模板与 EOS | 训练推理一致；EOS 必须计算 loss；pad=EOS 的坑 | 🧮 |
| 新知识 | SFT 学新事实慢，并会增加幻觉 | 📄 |
| 超参数 | LLaMA 2-Chat：2e-5、2 epoch；NEFTune | 📄 |
| SFT vs RL | SFT 偏记忆、RL 偏泛化；SFT 为 RL 稳定格式 | 📄 |

## ✅ 自测题

1. 实现一个基于 chat template 的多轮 loss masking，并说明如何处理 BPE 跨边界合并的问题。
2. 解释表面对齐假说，并给出一个支持它和一个反驳它的研究。
3. 什么情况下在指令部分也计算 loss 会更好？为什么？
4. 列出三个与 EOS / 模板相关的常见 bug，以及排查方法。
5. 为什么通过 SFT 注入新知识可能导致幻觉？你会怎样给模型注入新的领域知识？
6. 你要为一个垂直领域构建 SFT 数据集：说明数据来源、合成方法、质量控制、配比和评估方案。

---

> **下一章预告**：第 8 章｜PEFT 与 LoRA 系列——LoRA 的推导与初始化、QLoRA 的 NF4 与双重量化、DoRA / PiSSA 等变体，以及"LoRA 学得少、忘得少"背后的原因。
