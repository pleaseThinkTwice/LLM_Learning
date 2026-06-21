# LLM 面试通关笔记

> 覆盖 Transformer → 预训练 → SFT → RLHF → DPO → GRPO 全链路，面向大模型算法岗面试的系统性学习笔记。

## 📖 章节导航

| 章节 | 内容 | 核心主题 |
|---|---|---|
| [01-Transformer 深度解析](01-Transformer深度解析.md) | Self-Attention、MHA/MQA/GQA/MLA、Causal Mask | 必问地基题 |
| [02-位置编码专题](02-位置编码专题.md) | Sinusoidal、RoPE、ALiBi 等 | 高频对比题 |
| [03-归一化与激活](03-归一化与激活.md) | LayerNorm/RMSNorm/Pre-Norm/Post-Norm、SwiGLU/GeLU | 训练稳定性 |
| [04-分词器](04-分词器.md) | BPE、WordPiece、SentencePiece 等 | 工程基础 |
| [05-损失与优化器](05-损失与优化器.md) | CrossEntropy、AdamW、学习率调度 | 训练核心 |
| [06-预训练](06-预训练.md) | Scaling Law、数据配比、并行策略 | 大模型基础 |
| [07-SFT 指令微调](07-SFT指令微调.md) | SFT 数据构建、训练技巧、评估 | 对齐第一步 |
| [08-PEFT 与 LoRA 系列](08-PEFT与LoRA系列.md) | LoRA/QLoRA/AdaLoRA、参数高效微调 | 落地必备 |
| [09-RLHF 与 PPO](09-RLHF与PPO.md) | 奖励模型、PPO 推导、RLHF 全流程 | 对齐核心 |
| [10-DPO 数学推导专题](10-DPO数学推导专题.md) | DPO 损失推导、Bradley-Terry 模型 | 算法进阶 |
| [11-GRPO 与推理模型训练](11-GRPO与推理模型训练.md) | GRPO 原理、DeepSeek-R1、推理模型 | 前沿热点 |

## 🎯 使用方式

- **一面速查**：每道题有"一句话标答"，30 秒应付开场题
- **二面深挖**：配有数学推导和"为什么"，支撑 3 分钟深度回答
- **三面追问**：链接论文细节和演化脉络，应对延伸追问

## 📝 更新日志

- 2026-06-21：初始版本，覆盖 11 个专题
