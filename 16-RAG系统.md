# 第 16 章｜RAG 系统

> **本章定位**：第四部分回答"怎么把模型用起来"，RAG（检索增强生成）是第一站。贯穿全章的三个判断：
> 1. **RAG 是一条串行链路**：解析 → 分块 → 向量化 → 检索 → 融合 → 重排 → 组装 → 生成。效果不好时，关键是定位到哪一环出了问题，而不是先去调 prompt。
> 2. **检索召回率是整个系统的上限**：没检索到的内容，后面怎么做都补不回来。所以先测检索，再调生成。
> 3. **不同来源的分数不可比**：RRF 只用排名、LLM 评审更适合做相对比较，都是同一个原则。

> **配套章节**：Prefix Caching → 第 12 章 Q7 / 长上下文与 RoPE 外推 → 第 2 章 / 约束解码 → 第 14 章 Q8 / 推理服务与容量 → 第 15 章 / Function Calling → 第 17 章 / Agentic RAG → 第 18 章 / LLM-as-Judge 的偏差 → 第 19 章

---

## 📏 标注约定

| 标记 | 含义 |
|---|---|
| 📄 **[论文]** | 论文或官方报告的结论（附出处） |
| 🧮 **[推导]** | 可推导或可计算 |
| 💭 **[直觉]** | 经验、启发式或有争议 |
| ❓ **[开放]** | 尚无共识 |

---

## 📌 本章导航

| 序号 | 题目 | 难度 | 类型 |
|---|---|---|---|
| Q1 | **RAG 全景：vs 长上下文、vs 微调、七种失败模式** | ⭐⭐⭐⭐⭐ | 必问 |
| Q2 | **Chunking 策略** | ⭐⭐⭐⭐⭐ | 必问 + 实战 |
| Q3 | **Embedding 选型与 ANN 索引（HNSW）** | ⭐⭐⭐⭐⭐ | 必问 |
| Q4 | **混合检索与 RRF** | ⭐⭐⭐⭐⭐ | 必问 |
| Q5 | **Rerank：Bi vs Cross vs ColBERT** | ⭐⭐⭐⭐⭐ | 必问 |
| Q6 | **Query 改写、路由、HyDE** | ⭐⭐⭐⭐ | 实战 |
| Q7 | 高级 RAG：GraphRAG / CRAG / Agentic | ⭐⭐⭐⭐ | 进阶 |
| Q8 | **幻觉归因与 RAGAS 评估** | ⭐⭐⭐⭐⭐ | 必问 |

---

## Q1：RAG 全景——vs 长上下文、vs 微调、七种失败模式 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> RAG（📄 Lewis et al. 2020）在生成时检索外部文档，解决**私有性、时效性、可溯源、更新成本**的问题。它和微调互补：💭 微调更擅长改变"怎么做"（格式、风格、任务能力），RAG 更擅长提供"知道什么"（事实）。长上下文没有取代 RAG：📄 Li et al. 2024 发现资源充足时长上下文模型平均效果优于 RAG，但 RAG 便宜得多；📄 Xu et al. 2023 发现检索对长上下文模型同样有帮助。RAG 是串行链路，**每一环都有自己的失败模式，排查要先定位环节**。

### 📐 为什么要 RAG

1. **私有知识**：企业内部文档，模型从未见过。
2. **时效性**：模型知识停在训练截止时间。
3. **可溯源 / 可审计**：能给出引用，在金融、法律、医疗等场景常是合规要求。
4. **更新成本**：改一份文档，而不是重新训练模型。
5. **减少幻觉**：有依据地回答，但**不能消除幻觉**（见追问）。

### 📐 RAG vs 长上下文

**成本**：🧮 如果每次查询都把 100 万 token 的语料放进上下文，prefill 计算至少按 token 数线性增长（attention 部分是二次的）；RAG 每次只放几千 token，成本可以相差两个数量级。
> 💭 **这条论据在被削弱**：如果是同一份语料被反复询问，Prefix / Context Caching（第 12 章 Q7）可以摊销 prefill，剩下的主要是每次 decode 读长 KV 的访存成本。更准确的说法是：**语料大且常变 → RAG 占优；语料固定、可缓存、需要全局理解 → 长上下文更有竞争力**。

**有效利用率**：
- 📄 **Lost in the Middle**（Liu et al., TACL 2024）：相关信息位于上下文中部时，模型利用它的能力明显下降，呈 U 形曲线（开头和结尾较好）。
- 📄 **RULER**（Hsieh et al. 2024）：所测的号称支持 ≥32K 的模型里，只有约一半在 32K 长度上还能保持令人满意的表现。"支持 128K"不等于"在 128K 上好用"。

**规模与溯源**：企业知识库可能是 TB 级，远超任何上下文；RAG 天然知道每句话来自哪一段。

📄 **实证对比**：
- Xu et al. 2023（NVIDIA, ICLR 2024）：4K 上下文 + 检索可以达到与 16K 长上下文微调模型相当的效果；检索增强的 Llama2-70B-32K 在九个长上下文任务上平均优于 GPT-3.5-turbo-16k，也优于不带检索的同一模型。
- Li et al. 2024（EMNLP 2024 Industry）：资源充足时长上下文平均优于 RAG，但 RAG 成本低得多；提出 **Self-Route**，由模型自我判断该走 RAG 还是长上下文，以较低成本接近长上下文的效果。

💭 **长上下文改变了 RAG 的形态**：上下文变长后，chunk 可以更大、top-k 可以更多，"宽召回、让模型自己筛"变得可行，对 rerank 精度的要求下降；主要矛盾转向**成本、延迟和摆放顺序**。

### 📐 RAG vs 微调

| | 微调 | RAG |
|---|---|---|
| 擅长 | 格式、风格、领域术语、任务能力 | 事实、知识 |
| 知识更新 | 需要重新训练 | 改文档即可 |
| 可溯源 | 否 | 是 |
| 推理成本 | 不需要额外上下文 | 每次都要放入检索内容 |

💭 判断：模型**不知道**某个事实 → RAG；**知道但答得不对**（格式、风格）→ 微调；**听不懂领域问题** → 微调甚至继续预训练。📄 第 7 章 Q6 讨论了"SFT 能否教新知识"：Gekhman et al. 2024 发现通过微调学习新知识较慢，且会增加幻觉倾向，这是"事实交给 RAG"的一个证据。

### 📐 七个环节的失败模式

| # | 环节 | 典型失败 | 怎么查 |
|---|---|---|---|
| 1 | 解析 | PDF 表格乱码、多栏串行、扫描件没 OCR | **直接看入库的文本** |
| 2 | 分块 | 切断语义；chunk 缺少上下文（"它"指谁） | 抽查 chunk 是否自包含 |
| 3 | Embedding | 领域术语没见过；漏加 instruction 前缀；用错对称/非对称模型 | 测 Recall@k |
| 4 | 检索 | 型号、错误码等精确匹配失败；索引参数太保守 | 测 Recall@k |
| 5 | 融合/重排 | 相关文档被排到 top-k 之外 | 测 MRR / NDCG |
| 6 | 组装 | 关键内容放在长上下文中部；上下文过长 | 检查摆放顺序 |
| 7 | 生成 | 有依据仍编造；忽略上下文用参数知识 | 测 Faithfulness（Q8） |

> 💭 排查顺序：**先测检索**（Recall@k，免费且快），召回足够高再去看生成。召回只有 60% 时，调 prompt 救不回剩下的 40%。

### 🔥 高频追问

**Q：RAG 能消除幻觉吗？**

A：不能，它改变了幻觉的来源：
1. **检索失败**：没检索到，模型只能靠参数知识或编造。
2. **生成失败**：检索到了，但模型忽略、误读或"补充"了上下文中没有的信息。📄 Xie et al. 2024（*Adaptive Chameleon or Stubborn Sloth*, ICLR 2024）发现：外部证据连贯、有说服力时，模型会接受它，即使与自身记忆冲突；但当证据中有部分与自身记忆一致时，模型表现出强烈的确认偏误。
3. **知识库本身有错**：RAG 会忠实地传播错误，而且因为有引用，看起来更可信。这只能靠数据治理。

常用缓解手段：**要求逐句引用**（💭 编造的内容很难给出处，引用也让人可以核对，可配合约束解码保证格式）、**明确允许"无法回答"**（注意过度拒答）、**测 Faithfulness 并归因**（Q8）。

**Q：无关的检索结果一定有害吗？**

A：证据比直觉复杂：
- 📄 Shi et al. 2023（ICML）：在数学题描述中加入无关信息会让模型表现大幅下降；在 prompt 中要求忽略无关信息、自洽解码可以缓解。
- 📄 Yoran et al. 2024（ICLR）：无关段落会降低检索增强模型的表现；用约 1000 条混有相关/无关上下文的数据微调，就能让模型对无关上下文鲁棒。
- 📄 Cuconasu et al. 2024（*The Power of Noise*, SIGIR）：**看起来相关但不含答案的高分文档**最伤效果；反直觉的是，**加入随机文档**反而让准确率最多提升 35%。

💭 合理的读法：最危险的是"**相关但错误**"的上下文，而不是明显无关的噪声。这也是"不该检索时不要检索"（Q6 路由）和"用 rerank 分数做阈值丢弃"的依据。❓ 随机文档为什么有帮助，目前没有公认解释。

**Q：什么时候不该用 RAG？**

A：闲聊（检索内容可能把模型带偏，"不检索"应该是路由的显式选项）；需要全局理解的问题（top-k 看不到全貌，见 Q7）；汇总统计（"有多少个 X"，应走 Text2SQL）；格式风格问题（应微调）；知识量很小（几十条 FAQ 直接放进 system prompt，还能吃到 Prefix Caching）。

### 📚 出处
- Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. arXiv:2005.11401
- Liu et al. *Lost in the Middle: How Language Models Use Long Contexts*. TACL 2024. arXiv:2307.03172
- Hsieh et al. *RULER*. 2024. arXiv:2404.06654
- Xu et al. *Retrieval meets Long Context Large Language Models*. ICLR 2024. arXiv:2310.03025
- Li et al. *Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach*. EMNLP 2024 Industry. arXiv:2407.16833
- Xie et al. *Adaptive Chameleon or Stubborn Sloth*. ICLR 2024. arXiv:2305.13300
- Shi et al. *Large Language Models Can Be Easily Distracted by Irrelevant Context*. ICML 2023. arXiv:2302.00093
- Yoran et al. *Making Retrieval-Augmented Language Models Robust to Irrelevant Context*. ICLR 2024. arXiv:2310.01558
- Cuconasu et al. *The Power of Noise: Redefining Retrieval for RAG Systems*. SIGIR 2024. arXiv:2401.14887
- Gao et al. *Retrieval-Augmented Generation for Large Language Models: A Survey*. 2023. arXiv:2312.10997

---

## Q2：Chunking 策略 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 核心矛盾：chunk 太小，语义不完整、缺少上下文；chunk 太大，**一个向量要概括多件事，对哪件都不够精确**（💭 常说的"embedding 被稀释"），而且塞给模型的噪声多。比起寻找完美的 size，更好的办法是跳出二选一：**Small-to-Big**（小块检索、大块返回）、📄 **Contextual Retrieval**（用 LLM 给每个 chunk 写一段上下文说明再索引，Anthropic 报告 top-20 检索失败率降低 35%，配合 BM25 降低 49%，再加 rerank 降低 67%）、📄 **Late Chunking**（先对全文做 token 级编码，再按 chunk 边界池化）。

### 📐 为什么 chunk 太大不好

💭 embedding 把整个 chunk 压成一个向量。一个讲了五件事的长 chunk，其向量近似这五件事的某种"平均"；用户只问其中一件时，query 向量很"尖"，chunk 向量很"钝"，相似度不高。所以"chunk 大一点召回更全"并不成立。另外，即使检索到了，大量无关文本也会占用上下文、干扰生成。

### 📐 三种跳出二选一的方法

**① Small-to-Big / 父文档检索**：索引时把文档切成小块（如 128 token）做向量，同时记录每个小块所属的大段落；检索用小块（精确），返回父段落（完整）。变体：索引摘要或"这段能回答的问题"，返回原文。

**② Contextual Retrieval**（📄 Anthropic 2024）：对每个 chunk，把整篇文档和该 chunk 一起交给 LLM，生成一两句"这段在全文中的位置和主题"，prepend 到 chunk 前再做 embedding 和 BM25 索引。

| 方案（📄 Anthropic 报告，top-20 检索失败率，基线 5.7%） | 失败率 | 相对降低 |
|---|---|---|
| Contextual Embeddings | 3.7% | 35% |
| + Contextual BM25 | 2.9% | 49% |
| + Rerank | 1.9% | 67% |

> 💭 这是 Anthropic 在自选数据集上的博客结果，未经同行评审，迁移到你的语料需要重测。
> 🧮 成本：每个 chunk 一次 LLM 调用。但整篇文档是**固定前缀**，只有"当前 chunk"在变，Prompt / Prefix Caching（第 12 章 Q7）可以让成本大幅下降。

**③ Late Chunking**（📄 Günther et al. 2024, Jina）：把整篇长文档一次送入长上下文 embedding 模型，得到每个 token 的向量（它们都在全文的 attention 下计算），**再**按 chunk 边界做 mean pooling。不需要额外的 LLM 调用，前提是 embedding 模型支持足够长的输入。

| | Contextual Retrieval | Late Chunking |
|---|---|---|
| 怎么加上下文 | LLM 显式写一段文字 | attention 隐式注入 |
| 额外成本 | LLM 调用（可被缓存摊薄） | 几乎为零 |
| 限制 | 无文档长度限制 | 受 embedding 模型窗口限制 |
| BM25 是否受益 | **是**（加的是文字） | 否（只改向量） |
| 可解释性 | 能看到加了什么 | 黑盒 |

### 📐 Chunking 策略一览

| 策略 | 做法 | 评价 |
|---|---|---|
| 固定长度 + overlap | 每 N token 切，重叠 10–20% | 基线，容易切断语义 |
| 递归字符分割 | 按段落 → 换行 → 句子 → 空格逐级降级 | 简单实用，常见默认 |
| **结构感知** | Markdown 按标题、代码按函数/类、PDF 按段落/表格、法规按条款 | 💭 往往是性价比最高的一步 |
| 语义分块 | 相邻句子 embedding 相似度骤降处切分 | 慢 |
| 命题分块 | LLM 拆成原子命题（📄 Chen et al. 2024, *Dense X Retrieval*） | 精度高、成本高 |
| Small-to-Big | 小块检索、大块返回 | 推荐 |
| Contextual Retrieval | LLM 补上下文 | 高价值语料 |
| Late Chunking | 先全文编码、后池化 | 需长上下文 embedding |

💭 **结构感知被低估**：把 Markdown 的标题层级 prepend 到每个 chunk，相当于免费版的 Contextual Retrieval；表格整张作为一个 chunk；对话按话轮切并保留说话人。

### 🔥 高频追问

**Q：overlap 有用吗？**

A：它防止关键句恰好被切在两块中间，代价是存储和向量数增加、检索结果重复（需要去重，还会挤占 rerank 名额）。💭 常见取 10–20%。如果分块是结构感知的、切在天然边界上，overlap 的必要性就小得多：**overlap 主要是给固定长度切分打的补丁**。

**Q：表格、图片、代码怎么处理？**

A：💭 通用原则是 **"索引一个适合检索的表示，返回一个适合阅读的表示"**：
- **表格**：整表保留为 Markdown；另存一段 LLM 生成的自然语言描述用于检索；结构化数据进数据库，聚合类问题走 Text2SQL。
- **图片**：用视觉语言模型生成描述，索引描述、返回图片；或用多模态 embedding（CLIP 类）直接检索。
- **代码**：按 AST 切到函数/类，prepend 文件路径、类名、import；索引签名和 docstring，返回完整函数。

**Q：实现上有哪些常见坑？**

A：长度按 **token** 而不是字符计算；中文要在分隔符里加入"。"等中文标点，否则递归分割会一路降级到按字符切；表格不要被切开。

### 📚 出处
- Anthropic. *Introducing Contextual Retrieval*. 2024（博客）
- Günther et al. *Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models*. 2024. arXiv:2409.04701
- Chen et al. *Dense X Retrieval: What Retrieval Granularity Should We Use?* EMNLP 2024. arXiv:2312.06648

---

## Q3：Embedding 选型与 ANN 索引（HNSW） ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 检索用的 embedding 模型是 **bi-encoder**：query 和文档分别编码，文档向量可以离线算好，所以快；代价是两者从不交互，**文档被压缩时还不知道 query 会问什么**，这是精度受限的根源，也是 rerank 存在的理由（Q5）。选型要点：用为**非对称检索**（问题 → 段落）训练的模型；按模型说明加 instruction / 前缀；不要只看 MTEB 榜单，要在自己的数据上测。索引方面，📄 **HNSW** 是中等规模的常用选择，`ef_search` 是运行时调节召回与延迟的旋钮；更大规模用 IVF-PQ 或 DiskANN 等。

### 📐 对称 vs 非对称检索

| | 对称 | 非对称 |
|---|---|---|
| 场景 | 句子相似度、找重复问题 | **问题 → 文档段落**（RAG） |
| 两侧 | 同类型 | 疑问句 vs 陈述段落 |
| 前缀 | 通常不需要 | 很多模型要求 query 侧加前缀 |

📄 例如 BGE 系列建议在检索任务的 query 侧加一段指令前缀、文档侧不加；E5 系列要求两侧分别加 `query: ` 和 `passage: `。💭 漏加或加错会明显掉点，具体规则以模型卡为准。这个"query 和文档天然不同构"的问题，也是 HyDE（Q6）想解决的。

### 📐 选型

- 📄 **BGE-M3**（Chen et al. 2024）：一个模型同时输出 dense、sparse（词级权重）和多向量（ColBERT 式）三种表示，支持多语言和长输入。
- 中文常用 BGE、GTE 等；商用有 OpenAI、Cohere、Voyage 等。💭 排行榜（MTEB）有过拟合的风险，排名靠前的模型在你的领域不一定最好，**以自己的 Recall@k 为准**。
- 📄 **BEIR**（Thakur et al. 2021）：在 18 个异构数据集上的零样本评测显示，BM25 是很强的基线，不少稠密检索模型在域外泛化不如它。这是 Q4 混合检索的背景之一。

### 📐 维度与 Matryoshka 表示

🧮 存储：1000 万个向量，fp32 下 768 维约 31 GB，1024 维约 41 GB，3072 维约 123 GB。维度越高成本线性增长，💭 收益通常递减。

📄 **Matryoshka Representation Learning**（Kusupati et al., NeurIPS 2022）：训练时让向量的**前 $k$ 维**本身就是可用的表示，于是一个模型可以按需截断维度；OpenAI 的 text-embedding-3 支持指定输出维度。典型用法是两阶段检索：用低维向量在全库粗筛，再用全维向量精排少量候选（与"召回 → 精排"是同一个思路）。

### 📐 HNSW

📄 Malkov & Yashunin（TPAMI 2018）：多层近邻图，类似跳表。上层稀疏、边长，用于快速接近目标区域；底层包含全部点，用于精细搜索。搜索时从顶层入口贪心下降，到底层后维护一个大小为 `ef_search` 的候选池做宽度搜索。

| 参数 | 含义 | 影响 |
|---|---|---|
| `M` | 每个节点的邻居数 | 召回和内存都随之上升 |
| `ef_construction` | 建索引时的候选池 | 索引质量 vs 建索引时间 |
| `ef_search` | 查询时的候选池 | **运行时**召回 vs 延迟 |

🧮 **图的内存**：底层每个节点约存 $2M$ 个 4 字节邻居 id。 $M=16$ 时约 128 B/向量，相对 1024 维 fp32 向量（4 KB）只占约 3%；向量维度越低，图的占比越高。
> ❌ 旧版写"M=32 时图的内存约等于向量本身的一半"，对常见的高维向量不成立。

其他代价：删除会让图退化，需要定期重建；建索引较慢。

| 索引 | 适用规模（💭 量级） | 特点 |
|---|---|---|
| Flat（暴力） | 十万级以下 | 精确，几毫秒，无需额外组件 |
| HNSW | 十万到千万级 | 召回高，内存大 |
| IVF / IVF-PQ | 更大规模 | 压缩向量，召回有损 |
| DiskANN（Subramanya et al., NeurIPS 2019） | 十亿级 | 图放 SSD |

### 🔥 高频追问

**Q：为什么向量检索做不好精确匹配？**

A：💭 embedding 编码的是"关于什么"，而型号 `XR-2000P` 与 `XR-2000S`、错误码 `E4032`、罕见人名的区别是**符号层面**的；切分成子词后前几个片段完全相同，向量几乎一样。**否定**（"不含糖" vs "含糖"）同样容易被压近，因为两者讨论的是同一话题。解法：BM25、元数据过滤、混合检索（Q4）。

**Q：向量库怎么选？**

A：💭 按数据量和已有技术栈：数据量不大且已用 Postgres，用 pgvector（事务、备份、权限、和业务表 JOIN 都现成）；已用 Elasticsearch，就在 ES 里加向量字段，BM25 和向量在同一个系统里；Faiss 是库而非服务；千万级以上、需要分布式或复杂过滤时再考虑 Milvus、Qdrant 等专用系统。**不要为了少量数据引入一个分布式向量库。**

### 📚 出处
- Malkov & Yashunin. *Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs*. TPAMI 2018. arXiv:1603.09320
- Chen et al. *BGE M3-Embedding*. 2024. arXiv:2402.03216
- Kusupati et al. *Matryoshka Representation Learning*. NeurIPS 2022. arXiv:2205.13147
- Thakur et al. *BEIR*. NeurIPS 2021 Datasets and Benchmarks. arXiv:2104.08663
- Muennighoff et al. *MTEB: Massive Text Embedding Benchmark*. EACL 2023. arXiv:2210.07316

---

## Q4：混合检索与 RRF ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> 向量检索擅长语义匹配（"怎么退货" ↔ "退换货流程"），BM25 擅长精确符号（型号、错误码、人名、否定词），两者互补。融合时**不同检索器的分数不可比**：余弦相似度有界且常挤在窄区间，BM25 无界且依赖文档长度和词频，min-max 归一化又容易被离群值破坏。📄 **RRF**（Cormack et al., SIGIR 2009）只用排名： $\text{RRF}(d)=\sum_r \frac{1}{k+\text{rank}_r(d)}$， $k=60$。排名没有量纲，天然可比。

### 📐 为什么不能直接加权相加分数

🧮 min-max 归一化被离群值破坏的例子：一批 BM25 分数为 [47.3, 3.1, 2.9, 2.8, 2.7]，归一化后变成 [1.0, 0.009, 0.004, 0.002, 0.0]，后四个的区分度几乎消失。

### 📐 RRF 与 $k$ 的作用

$$\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}$$

- 🧮 $k=60$ 时第 1 名与第 2 名的得分是 $1/61$ 与 $1/62$，只差约 1.6%； $k=0$ 时是 1 与 0.5。
- $k$ 越大，各名次差距越小，越看重"**被多个检索器同时召回**"； $k$ 越小，越看重某个检索器的头部。
- 🧮 例如 $k=60$ 时，在两路中都排第 5 的文档得分 $2/65 \approx 0.0308$，高于只在一路中排第 1 的 $1/61 \approx 0.0164$。
- 实现细节：rank 从 1 开始；某路没召回的文档不参与该路求和。

💭 **"相对优于绝对"**在本书多次出现：PPO/GRPO 的优势函数（第 9、11 章）、min-p 的相对阈值（第 14 章 Q1）、RRF 的排名、LLM 评审的成对比较（第 19 章）。共同原因是绝对量的尺度依赖场景、容易被离群值破坏，相对量更稳健。

### 📐 BM25

$$\text{BM25}(q,d) = \sum_i \text{IDF}(q_i)\cdot\frac{f(q_i,d)(k_1+1)}{f(q_i,d) + k_1\left(1-b+b\frac{|d|}{\text{avgdl}}\right)}$$

- **IDF**：罕见词权重高。
- **$k_1$（常取 1.2–2.0）控制词频饱和**：🧮 $f\to\infty$ 时该项趋于 $k_1+1$，出现 100 次并不比 10 次重要多少；TF-IDF 的线性词频则容易被关键词堆砌利用。
- **$b$（常取 0.75）控制文档长度归一化**，避免长文档天然占优。

### 📐 标准流水线

```
query ─┬─ 向量检索 top-50 ─┐
       └─ BM25 top-50     ─┴─ RRF 融合 → top-20 → rerank → top-5 → LLM
          宽召回             廉价粗排            昂贵精排
```

💭 **融合和 rerank 只能重排，不能凭空找回第一阶段漏掉的文档**，所以第一阶段要宽。多返回一些候选对 HNSW 和 BM25 几乎不增加成本，真正的成本在 rerank，RRF 正好充当免费的粗排。

### 🔥 高频追问

**Q：BGE-M3 的稀疏输出能取代 BM25 吗？**

A：📄 BGE-M3 的稀疏表示给每个 token 学一个权重，可以覆盖部分同义词。💭 但它基于模型的词表，罕见型号会被切成多个子词；BM25 零训练、不需要 GPU、有成熟的倒排索引工程。实践中可以让 M3 的 dense 和 sparse 先融合，再与 BM25 做 RRF。

**Q：要不要给各路检索器调权重？**

A：💭 RRF 的优点是基本不用调参。全局固定权重在一类 query 上调好，换一类就不对了；与其调权重，不如**按 query 类型路由**（Q6）：错误码查询以 BM25 为主，语义问题以向量为主。更进一步可以用 learning-to-rank 学一个融合模型，对多数 RAG 场景属于过度工程。

### 📚 出处
- Cormack, Clarke, Büttcher. *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. SIGIR 2009
- Robertson & Zaragoza. *The Probabilistic Relevance Framework: BM25 and Beyond*. FnTIR 2009
- Chen et al. *BGE M3-Embedding*. arXiv:2402.03216

---

## Q5：Rerank——Bi vs Cross vs ColBERT ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> **Bi-encoder** 在看到 query 之前就把文档压成一个向量（"先压缩后匹配"）；**cross-encoder** 把 query 和文档拼在一起过模型，每个 token 互相 attend（"边看边匹配"），更准，但不能预计算，只能用于少量候选；📄 **ColBERT**（late interaction）为文档的每个 token 存一个向量，检索时对每个 query token 取与文档 token 的最大相似度再求和（MaxSim），精度接近 cross-encoder、文档侧仍可预计算，代价是存储（📄 ColBERTv2 用残差压缩把空间降低 6–10×）。标准架构：**bi-encoder + BM25 宽召回 → cross-encoder 精排**。

### 📐 三者对比

$$\text{Bi: } \cos(\text{Enc}(q), \text{Enc}(d)),\quad \text{Cross: } f([q; d]),\quad \text{ColBERT: } \sum_{i\in q}\max_{j\in d} E_{q_i}\cdot E_{d_j}$$

| | Bi-encoder | ColBERT | Cross-encoder |
|---|---|---|---|
| 交互 | 只有最后的点积 | token 级 MaxSim | 每层全 attention |
| 文档侧预计算 | 1 个向量 | 每 token 1 个向量 | 不能 |
| 存储 | 小 | 大（可压缩） | 无 |
| 精度 | 基线 | 接近 cross | 最高 |
| 用途 | 全库召回 | 召回或重排 | 少量候选精排 |

💭 **信息瓶颈的视角**：bi-encoder 的上限是"一个 $d$ 维向量能表达多少与未知 query 相关的信息"；ColBERT 把瓶颈放宽到"每个 token 一个向量"，但交互仍浅；cross-encoder 在已知 query 的前提下阅读文档，没有提前压缩这一步。

### 📐 组装 prompt 的免费技巧

1. **摆放顺序**：💭 依据 Lost in the Middle 的 U 形曲线，把最相关的文档放在开头和结尾，次要的放中间。
2. **加元数据**：来源、章节、日期，让模型能引用、判断时效和权威性。
3. **明确指令**：只依据给定文档；逐句标注来源编号；没有相关信息时明确说明。
4. **检索结果放在 prompt 末尾附近**：system、工具描述、示例等固定内容放前面，每次都变的检索结果放后面，否则会破坏 Prefix Caching（第 12 章 Q7）。

### 🔥 高频追问

**Q：能用 LLM 做 rerank 吗？**

A：可以，三种方式：
- **Pointwise**：逐个打相关性分。简单可并行，但 💭 分数不校准。
- **Pairwise**：两两比较，更稳但调用次数多。
- **Listwise**：📄 RankGPT（Sun et al., EMNLP 2023）一次给模型一组文档让它输出排序，用滑动窗口处理长列表，报告经过合适指令的 GPT-4 在主流 IR 基准上可以达到甚至超过当时有监督的 SOTA；还把排序能力蒸馏到 440M 的小模型，在 BEIR 上超过 3B 的有监督模型。

💭 LLM rerank 适合高价值、低 QPS、延迟预算宽松的场景（法律、医疗、深度研究）；一般场景先用专门的 cross-encoder reranker。

**Q：rerank 太慢怎么办？**

A：💭 按收益排序：
1. **缩小候选数**：用 RRF 先粗排，把要精排的候选从上百个降到几十个。
2. **批处理**：🧮 很多实现逐条循环调用 reranker，每次 batch=1，严重访存受限（第 12 章 Roofline）；一次性把所有 (q, d) 对打包前向，权重只读一次，被整个 batch 摊薄。
3. **用更小的模型**、截断文档长度、缓存重复的 (q, d) 对。
4. 换 ColBERT，文档侧预计算，代价是存储。

### 📚 出处
- Khattab & Zaharia. *ColBERT*. SIGIR 2020. arXiv:2004.12832
- Santhanam et al. *ColBERTv2*. NAACL 2022. arXiv:2112.01488
- Nogueira & Cho. *Passage Re-ranking with BERT*. 2019. arXiv:1901.04085
- Sun et al. *Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents*（RankGPT）. EMNLP 2023. arXiv:2304.09542

---

## Q6：Query 改写、路由、HyDE ⭐⭐⭐⭐

### 🎯 一句话标答

> 用户的 query 常常不适合直接检索。**多轮对话中的指代必须消解**（"它的价格呢？"单独检索几乎没有意义，而且不会报错，只是默默变差）。📄 **HyDE**（Gao et al., ACL 2023）让 LLM 先写一段假想答案，用它的 embedding 去检索：假想答案的事实可能是错的，但它的句式、术语、话题与真实文档相似，💭 相当于把"问题 → 段落"的非对称检索转成"段落 → 段落"的对称检索。**路由**决定每类 query 走哪条路，其中"不检索"必须是显式选项。

### 📐 改写技术

| 技术 | 做法 | 额外开销 | 用途 |
|---|---|---|---|
| 指代消解 | 结合历史改写为独立 query | 1 次（小）模型调用 | **多轮必需** |
| HyDE（📄 Gao et al. 2023） | 生成假想答案，用它检索 | 1 次 LLM 调用 | 零样本、embedding 未在领域微调时 |
| Query2doc（📄 Wang et al. 2023） | 生成伪文档拼接到 query 上（也用于 BM25） | 1 次 LLM 调用 | 同上 |
| Multi-Query / RAG-Fusion | 生成多个改写，分别检索，RRF 融合 | 1 次 LLM + N 次检索 | 表述差异大时提高召回 |
| 分解 | 复杂问题拆成子问题 | 1 次 LLM + N 次检索 | 多跳、对比 |
| Step-back（📄 Zheng et al., ICLR 2024） | 先问一个更抽象的问题检索背景知识 | 额外调用 | 需要原理性背景时 |

💭 **每个技巧都要付延迟**。指代消解在关键路径最前面，它的延迟会直接加到 TTFT 上，适合用小模型甚至规则先过滤（只对含代词或过短的 query 改写）。HyDE、Multi-Query 只有在评测证明有收益时才上，并且最好由路由决定是否启用。

### 📐 路由

| Query 类型 | 例子 | 策略 |
|---|---|---|
| 事实查询 | "退货政策是什么" | 向量 + BM25 混合 |
| 精确查找 | "错误码 E4032" | BM25 为主 / 元数据过滤 |
| 关系、全局 | "这批报告的主要主题" | GraphRAG 或摘要索引（Q7） |
| 汇总统计 | "有多少客户在用 X" | Text2SQL |
| 时效性 | "最新政策" | 时间过滤 |
| 闲聊 | "你好" | **不检索** |
| 需要计算 | "去年增长率" | 工具调用（第 17 章） |

实现：规则（正则识别错误码等，零成本）+ 小分类模型（毫秒级）为主，复杂情况再用 LLM 分类。

### 🔥 高频追问

**Q：路由分错了怎么办？**

A：两类错误都不便宜：该检索没检索 → 模型凭参数知识编；不该检索却检索 → 可能被看起来相关但错误的内容带偏（Q1 追问）。兜底手段：
1. **软路由**：检索后用 rerank 分数做阈值，最高分都很低就丢弃检索结果。💭 这比只看 query 猜意图更可靠，因为它基于实际检索结果。
2. **prompt 兜底**：说明"以下文档可能不相关，不相关时请忽略"。📄 Shi et al. 2023 发现在 prompt 中加入忽略无关信息的指令能缓解干扰。
3. **交给模型决定**：把检索做成工具，由模型决定是否、何时、检索什么（Agentic RAG，Q7 和第 18 章）。

**Q：HyDE 什么时候不值得？**

A：💭 embedding 模型已经在你的领域针对性微调过、query 与文档的表述差异不大时，收益小，不值得多一次 LLM 调用的延迟。生成多个假想答案取平均向量、并与原 query 向量混合，可以降低假想跑偏的风险。

### 📚 出处
- Gao et al. *Precise Zero-Shot Dense Retrieval without Relevance Labels*（HyDE）. ACL 2023. arXiv:2212.10496
- Wang et al. *Query2doc: Query Expansion with Large Language Models*. EMNLP 2023. arXiv:2303.07678
- Zheng et al. *Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models*. ICLR 2024. arXiv:2310.06117
- Rackauckas. *RAG-Fusion: a New Take on Retrieval-Augmented Generation*. 2024. arXiv:2402.03367

---

## Q7：高级 RAG——GraphRAG / CRAG / Agentic ⭐⭐⭐⭐

### 🎯 一句话标答

> 📄 **GraphRAG**（Edge et al. 2024, Microsoft）用 LLM 从文本中抽取实体和关系建图，用 Leiden 算法做层次社区检测，为每个社区生成摘要；回答"整个语料讲了哪些主题"这类全局问题时，对社区摘要做 map-reduce——这类问题 top-k 检索从范式上就答不了。代价是索引成本高、更新困难。📄 **CRAG**（Yan et al. 2024）用一个轻量评估器判断检索质量，差时改用网页搜索；📄 **Self-RAG**（Asai et al., ICLR 2024）训练模型输出反思 token 自行决定检索与否。**Agentic RAG** 把检索做成工具，由模型多轮决定查什么、够不够。💭 多数场景先把"结构感知分块 + 混合检索 + rerank + 改写"做好，高级方案按 query 类型通过路由启用。

### 📐 GraphRAG 为什么能回答全局问题

🧮 一份约 25 万 token 的报告，top-5 个 512 token 的 chunk 只覆盖约 1%。问"主题是什么"，问题不在检索精度，而在 **top-k 采样本身覆盖不到全貌**。

GraphRAG 的 **Global Search**：① 每个社区摘要与 query 一起交给 LLM，生成部分答案并打相关性分（map，可并行）；② 按分数汇总部分答案生成最终答案（reduce）。💭 这是"用层次化的压缩结构换取覆盖率"。**Local Search** 从实体出发沿图扩展邻居，适合"A 和 B 是什么关系"。

📄 论文报告：在百万 token 量级的语料上，对全局性问题，GraphRAG 生成的答案在全面性和多样性上明显优于普通 RAG。

### 📐 对比

| | GraphRAG | CRAG | Self-RAG | Agentic RAG |
|---|---|---|---|---|
| 核心 | 实体图 + 社区摘要 | 检索评估器 + 网页搜索兜底 | 反思 token | 检索即工具 + 循环 |
| 是否改模型 | 否 | 训练一个小评估器 | **需要训练生成模型** | 否 |
| 索引成本 | 高 | 正常 | 正常 | 正常 |
| 查询成本 | 中 | 中 | 低 | 高（多轮） |
| 解决 | 全局、多跳问题 | 检索质量不稳 | 何时检索 | 复杂、多步任务 |

### 🔥 高频追问

**Q：GraphRAG 什么时候值得，有没有更便宜的替代？**

A：💭 值得的条件：语料相对静态、高价值、有大量全局/关系类问题、规模适中。FAQ 和客服知识库通常以事实查询为主，普通 RAG 就够。便宜的替代：全局问题 → 为每篇文档生成摘要并检索摘要；关系问题 → 多跳检索或 Agentic RAG；实体查询 → 实体抽取 + 元数据过滤。降本手段：小模型抽实体；只对核心文档建图；抽取 prompt 的固定前缀可以吃到 Prefix Caching。

**Q：Agentic RAG 的成本怎么控制？**

A：设迭代上限；ReAct 每轮只在末尾追加，💭 前缀高度重合，必须开 Prefix Caching（第 15 章 Q8）；相互独立的子查询并行执行；简单 query 仍走普通 RAG（路由）。

**Q：怎么决定上不上这些高级方案？**

A：先建评测集并测基线；**按 query 类型分组看失败案例**（💭 总体 80% 可能是"事实类 95% + 全局类 10%"）；只对失败的那类 query 引入专用方案并通过路由分发；最后核算延迟和成本。

### 📚 出处
- Edge et al. *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. 2024. arXiv:2404.16130
- Yan et al. *Corrective Retrieval Augmented Generation*. 2024. arXiv:2401.15884
- Asai et al. *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. ICLR 2024. arXiv:2310.11511
- Singh et al. *Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG*. 2025. arXiv:2501.09136

---

## Q8：幻觉归因与 RAGAS 评估 ⭐⭐⭐⭐⭐

### 🎯 一句话标答

> RAG 的错误要先**归因**：**检索失败**（上下文里根本没有答案）还是**生成失败**（有答案但没用、用错或编造）。📄 **RAGAS**（Es et al. 2024）的四个指标正好对应这个二分：**Faithfulness、Answer Relevancy** 衡量生成，**Context Precision、Context Recall** 衡量检索。实践原则：**先测检索**（Recall@k 不需要 LLM、便宜、指向明确），召回足够高后再看生成。

### 📐 四个指标

| 指标 | 测什么 | 需要标准答案 | 做法 |
|---|---|---|---|
| **Faithfulness** | 答案是否有上下文依据 | 否 | 把答案拆成原子陈述，逐条判断能否由上下文推出，取比例 |
| Answer Relevancy | 答案是否切题 | 否 | 由答案反推若干问题，与原问题算 embedding 相似度 |
| Context Precision | 相关内容是否排在前面 | 是 | 加权的 precision@k |
| **Context Recall** | 该检索的是否都检索到 | 是 | 标准答案中的陈述有多少能由上下文支持 |

💭 Faithfulness 和 Answer Relevancy 不需要标准答案，可以对线上**采样流量**持续运行，做幻觉率监控。

### 📐 Faithfulness 的实现细节

```python
def faithfulness(answer, contexts):
    claims = llm(f"把下面的回答拆成独立的原子事实陈述，每行一条：\n{answer}").splitlines()
    supported = 0
    for c in claims:
        v = llm(f"仅根据上下文判断该陈述是否被明确支持，只答'是'或'否'。"
                f"上下文没提到的一律答'否'。\n上下文：{contexts}\n陈述：{c}")
        supported += v.strip() == "是"
    return supported / max(len(claims), 1)
```

1. **"上下文没提到的一律答否"**：否则评审模型会用自己的知识判断陈述是否属实，而我们要测的是"能否从上下文推出"。
2. **拆分粒度决定分数**：拆得越细分数越低。💭 所以绝对值意义有限，**同一套拆分与评审设置下的相对比较**才可靠（LLM 评审的偏差见第 19 章）。

### 🔥 高频追问

**Q：只能测一个指标，测哪个？**

A：**检索召回（Recall@k 或 Context Recall）**。它是上限，便宜，且指向明确。Faithfulness 有一个盲区：**上下文本身是错的时候，模型忠实地复述错误，Faithfulness 仍是 100%，但答案是错的。**

**Q：评测集怎么建？**

A：从文档反向生成问题再人工筛选（快，但分布与真实 query 不同）；从线上真实 query 采样标注（最可靠）。💭 至少覆盖各类 query（事实、多跳、全局、精确查找、闲聊/不该检索），并**按类型分组报告指标**。📄 ARES（Saad-Falcon et al., NAACL 2024）用合成数据微调轻量评审模型，并用少量人工标注做预测驱动推断（PPI）给出置信区间，可以作为 RAGAS 之外的另一种思路。

**Q：线上怎么监控？**

A：
- **免费的客观信号**：top-1 rerank 分数的分布、空召回率、检索延迟、**"无法回答"的比例**（💭 突然上升常常意味着检索链路出了问题，如 embedding 服务异常或知识库更新出错）。
- **采样的 LLM 评审**：对小比例流量跑 Faithfulness。
- **用户信号**：点赞/点踩、追问率、引用点击。💭 引用点击需要结合后续行为解读：用户点开引用可能是在验证一个他不太相信的答案。

### 📚 出处
- Es et al. *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. EACL 2024 Demo. arXiv:2309.15217
- Saad-Falcon et al. *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems*. NAACL 2024. arXiv:2311.09476
- Zheng et al. *Judging LLM-as-a-Judge*. NeurIPS 2023. arXiv:2306.05685

---

## 🔬 研究前沿与开放问题

- 📄 **检索与长上下文的融合**：Self-Route（Li et al. 2024）在两者之间路由；检索头的研究（Wu et al. 2024, *Retrieval Head Mechanistically Explains Long-Context Factuality*）发现长上下文模型内部少数注意力头负责从上下文中检索信息，屏蔽它们会显著损害"大海捞针"类任务。❓ 随着上下文成本下降，外部检索会不会逐渐被模型内部的稀疏注意力替代？
- ❓ **噪声的作用**：*The Power of Noise* 发现加入随机文档反而提高准确率，机制尚不清楚。可能与 attention sink、上下文长度带来的校准变化有关，有待验证。
- ❓ **知识冲突**：当检索内容与模型参数知识冲突时，模型何时该相信哪一方？Xie et al. 2024 揭示了确认偏误，但如何训练出"按证据可信度权衡"的模型仍是开放问题。
- ❓ **端到端优化**：检索器和生成器通常分开训练。如何以最终答案质量为目标联合优化（RAG 原论文做了端到端训练，但在大模型时代成本很高），以及如何用 RL 训练模型主动检索（Search-R1 等 2025 年的工作），是活跃方向。
- 🧪 **实验设想（文字）**：在一个开放域 QA 数据集上，固定生成模型，对比四种上下文：① 只放金标段落 ② 金标 + 看起来相关但不含答案的高分段落 ③ 金标 + 随机段落 ④ 只放干扰段落，并改变金标段落在上下文中的位置。这能同时复现 Lost in the Middle 和 Power of Noise 的核心现象，并检验两者的交互。

---

## 📝 本章小结

| 关键点 | 你必须能说清楚 | 性质 |
|---|---|---|
| RAG vs 长上下文 | 成本、有效利用率（Lost in the Middle、RULER）、规模、溯源；长上下文改变了 RAG 的形态 | 📄 + 🧮 |
| RAG vs 微调 | 互补：事实交给检索，格式与能力交给微调 | 💭 + 📄 |
| 七环失败模式 | 先定位环节；先测检索 | 💭 |
| 幻觉来源 | 检索失败、生成失败、知识库本身错误；相关但错误的上下文最危险 | 📄 |
| Chunking | 太大则一个向量概括多件事；Small-to-Big、Contextual Retrieval、Late Chunking；结构感知最划算 | 📄 + 💭 |
| Bi-encoder | 压缩文档时还不知道 query；这是 rerank 的理由 | 💭 |
| 非对称检索 | query 侧前缀按模型卡加 | 📄 |
| HNSW | 多层图；`ef_search` 是运行时召回/延迟旋钮；图的内存对高维向量占比小 | 📄 + 🧮 |
| 混合检索 | 向量抓语义，BM25 抓符号；BEIR 中 BM25 是强基线 | 📄 |
| RRF | 只用排名； $k=60$；更看重共识 | 📄 + 🧮 |
| 宽召回 | 后续阶段只能重排，不能找回漏掉的 | 💭 |
| Rerank | cross-encoder 精排；批处理；摆放顺序；检索结果放 prompt 后部 | 📄 + 🧮 |
| 改写与路由 | 多轮必须指代消解；HyDE 把非对称转成对称；"不检索"是选项 | 📄 + 💭 |
| GraphRAG | 社区摘要 + map-reduce 回答全局问题；昂贵，按需路由 | 📄 |
| RAGAS | 两个测生成、两个测检索；Faithfulness 高不代表答案对 | 📄 |

## ✅ 自测题

1. "上下文都 100 万了，还要 RAG 吗？"给出论据和反论据，并说明长上下文如何改变了 RAG 的设计。
2. 用户反馈"答得不对"，你的前三个排查动作是什么？
3. 为什么 chunk 太大不好？Small-to-Big、Contextual Retrieval、Late Chunking 分别怎么解决这个问题？
4. 为什么 RRF 用排名而不用分数？举一个 min-max 归一化被离群值破坏的例子，并解释 $k=60$ 的含义。
5. 用信息瓶颈解释 bi-encoder、ColBERT、cross-encoder 的精度差异。
6. rerank 延迟太高，列出优化手段，并说明其中哪一招与第 12 章的 Roofline 有关。
7. HyDE 的假想答案内容是错的，为什么仍然有效？什么时候不值得用？
8. "多检索总比不检索好"这句话对吗？结合 Shi et al. 2023 和 *The Power of Noise* 的发现回答。
9. Faithfulness = 100% 但答案完全错误，可能吗？说明了什么？
10. **设计题**：为客服团队设计一个企业知识库 RAG（10 万份文档，含大量 PDF 表格；query 包括产品参数、错误码、政策解释、"有多少客户在用 X"）。说明每类 query 走哪条路由，以及你的评测方案。

---

> **下一章预告**：第 17 章｜Function Calling 与工具调用——模型怎么学会调用工具、为什么它知道何时停下、schema 设计、并行调用、错误恢复，以及 MCP 中 Tools / Resources / Prompts 的控制权划分。RAG 让模型能"读"，工具调用让模型能"做"。
