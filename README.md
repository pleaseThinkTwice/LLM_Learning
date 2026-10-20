# LLM 学习笔记

> 从 Transformer 到预训练、对齐（SFT / RLHF / DPO / GRPO），再到推理部署、RAG 与 Agent，以及 MoE、多模态、可解释性等研究专题的系统性笔记。目标不只是应付面试，而是**能讲清证据链、能提出研究问题**。

## 📖 章节导航

| 章节 | 内容 | 核心主题 |
|---|---|---|
| [00-全书知识地图](00-全书知识地图.md) | 跨章主线、锚点、反直觉结论 | 复习索引 |
| **第一部分：模型是怎么来的** | | |
| [01-Transformer 深度解析](01-Transformer深度解析.md) | Self-Attention、MHA/MQA/GQA/MLA、Causal Mask | 地基 |
| [02-位置编码专题](02-位置编码专题.md) | Sinusoidal、RoPE、ALiBi、PI/NTK/YaRN | 长度外推 |
| [03-归一化与激活](03-归一化与激活.md) | LN/RMSNorm、Pre/Post/Peri-LN、SwiGLU、DeepNorm、QK-Norm | 训练稳定性 |
| [04-分词器](04-分词器.md) | BPE、WordPiece、SentencePiece | 工程基础 |
| [05-损失与优化器](05-损失与优化器.md) | CrossEntropy、AdamW、学习率调度 | 训练核心 |
| [06-预训练](06-预训练.md) | Scaling Law、数据配比、并行策略 | 大模型基础 |
| **第二部分：模型是怎么被驯服的** | | |
| [07-SFT 指令微调](07-SFT指令微调.md) | SFT 数据构建、训练技巧、评估 | 对齐第一步 |
| [08-PEFT 与 LoRA 系列](08-PEFT与LoRA系列.md) | LoRA/QLoRA/AdaLoRA | 参数高效微调 |
| [09-RLHF 与 PPO](09-RLHF与PPO.md) | 奖励模型、PPO、RLHF 全流程 | 对齐核心 |
| [10-DPO 数学推导专题](10-DPO数学推导专题.md) | DPO 损失推导、Bradley-Terry | 算法进阶 |
| [11-GRPO 与推理模型训练](11-GRPO与推理模型训练.md) | GRPO、DeepSeek-R1、推理模型 | 前沿 |
| **第三部分：怎么把模型跑起来** | | |
| [12-推理基础与 KV Cache 工程](12-推理基础与KVCache工程.md) | Prefill/Decode、KV Cache、PagedAttention | 推理地基 |
| [13-量化专题](13-量化专题.md) | GPTQ/AWQ/SmoothQuant/FP8、KV 量化 | 压缩 |
| [14-解码策略与投机解码](14-解码策略与投机解码.md) | 采样策略、投机解码、EAGLE/Medusa | 加速 |
| [15-推理引擎与服务化](15-推理引擎与服务化.md) | vLLM/SGLang、调度、并行 | 部署 |
| **第四部分：怎么把模型用起来** | | |
| [16-RAG 系统](16-RAG系统.md) | 检索、重排、RAG 评估 | 知识增强 |
| [17-Function Calling 与工具调用](17-FunctionCalling与工具调用.md) | 工具调用、协议、MCP | Agent 基础 |
| [18-Agent 架构](18-Agent架构.md) | 规划、记忆、多 Agent | Agent |
| [19-评估体系](19-评估体系.md) | Benchmark、LLM-as-Judge、Agent 评估 | 评估 |
| [20-系统设计实战](20-系统设计实战.md) | 端到端系统设计题 | 综合 |
| **第五部分：研究专题** | | |
| [21-MoE 架构与训练](21-MoE架构与训练.md) | 路由、负载均衡（aux-loss-free）、细粒度/共享专家、MoE scaling law | 稀疏架构 |
| [22-多模态大模型](22-多模态大模型.md) | ViT、CLIP/SigLIP、连接器、分辨率与视觉 token、原生多模态、多模态幻觉与评测 | 多模态 |
| [23-可解释性](23-可解释性.md) | 探针、logit lens、叠加与 SAE、activation patching 与电路、归因图、steering、知识编辑、CoT 忠实度 | 可解释性 |

## 🎯 使用方式

- **一面速查**：每道题的"一句话标答"
- **二面深挖**：数学推导和"为什么"
- **三面追问 / 研究视角**：论文出处、演化脉络、开放问题

## 📏 事实核查约定

全书采用以下标记：

| 标记 | 含义 |
|---|---|
| 📄 [论文] | 原论文或技术报告中明确写出的结论，附出处 |
| 🧮 [推导] | 可以严格推出的数学事实 |
| 💭 [直觉] | 启发式解释，没有严格证明，或学界有争议 |
| ❓ [开放] | 尚无共识的研究问题 |

**不写没有出处的具体数字**（加速比、层数阈值、实验分数、"大厂考察频率"等）。

## 📝 更新日志

- 2026-10-20：新增第五部分"研究专题"：第 21 章 MoE、第 22 章多模态、第 23 章可解释性。按第 3 章标准重写第 15–20 章（篇幅约减半，核对引用数字，第 20 章容量估算补上 prefill 算力）。重写 `00-全书知识地图.md`，覆盖全部 23 章。
- 2026-10-19：完成第 12–20 章修订。删除全部"频率"打分列与"大厂偏好"小节；重算第 14 章投机解码加速比表；修正 EAGLE-2/3、DistServe、MT-Bench（一致率口径、位置一致性、注水攻击）、Anthropic 多 Agent token 倍数、MCP 规范（SHOULD 而非 MUST）等数字与表述；补充 MagicDec、RAG vs 长上下文实证、METR time horizon、评测统计学等内容；第 15–20 章新增"研究前沿与开放问题"小节。
- 2026-10-19：重写第 1–11 章（同标准：去除无出处数字，补充论文出处、前沿与开放问题）。
- 2026-10-19：重写第 3 章。修正 Sandwich-LN 公式、梯度公式、RMSNorm 历史、QK-Norm 现状；删除无出处的数字；补充 Xiong/Liu 2020、Pre-RMSNorm 等价性、Peri-LN/OLMo 2/Gemma 3、Curse of Depth、DyT 等内容。README 补全第 12–20 章。
- 2026-06-21：初始版本
