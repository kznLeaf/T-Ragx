# 项目报告

关于若干实验、观察与思考的松散笔记

## 引言

大型语言模型（LLM）显著改变了面向普通用户的机器翻译（MT）体验，其输出比早期 Google Translate 等服务常见的生硬译法更为流畅。此前，基于神经网络的机器翻译模型与服务——如 T5、Opus、NLLB，尤其是 DeepL——已取得显著进展。DeepL 因其译文更自然，在网络小说爱好者中颇受欢迎。ChatGPT 及同类基于 LLM 的服务/模型进一步提升了机器翻译的可及性与自然度。

然而，这些模型的一大局限是：对所译文献缺乏持久记忆。这会导致次优译文，例如同一角色在不同句子中译名不一致。传统上，人工译者通过建立翻译术语表（glossary）解决这一问题，以保证专有名词译法统一。此外，为每个项目建立翻译记忆库（translation memory）既能提高效率，也有助于在全文中保持一致的文风。

## 实验

### 数据集

说明：所有中文文本（输入、预测与参考译文）均使用 [OpenCC](https://github.com/BYVoid/OpenCC) 的 `s2tw.json` 配置转换为繁体中文。

#### 翻译记忆库与术语表

[若干开放数据集](../README.md#data-sources) 被合并，用于构建通用翻译记忆库。

##### 通用

翻译记忆库

- 开源平行语料的混合（见表）

术语表

- Wiki-data（https://huggingface.co/datasets/rayliuca/WikidataLabels）

##### 任务内（关于我转生变成史莱姆这档事 / Reincarnated Slime）

翻译记忆库

- 不在测试集中的章节

术语表

- 日语 → 英语

  - 从 [Fandom wiki](https://tensura.fandom.com/wiki/That_Time_I_Got_Reincarnated_as_a_Slime) 抓取

- 日语 → 中文

  - 从维基百科抓取

#### 训练

训练数据质量对实验结果至关重要。JESC、MTNT 等数据集虽有价值，但其对齐精度不及 WMT 测试集。因此，最终训练集主要以人工整理的数据集为主，如 WMT、OpenMantra 与 ASPEC；并辅以机器对齐、噪声或错配略高的数据集，包括 JESC、ted_talks_iwslt 与 WCC-JC。

处理后的训练集采用如下策略：在可用时纳入最多三句前文（尤其是在 WMT 数据中），并引入均衡的翻译示例组合——以相等概率（各 25%）呈现：无翻译示例、一个翻译示例、三个翻译示例，或正确译文输出（故意的数据泄漏，用以模拟成功的 TM 检索）。此外，训练集覆盖三种语言的双向翻译，共六个翻译方向。数据集总计约 80,000 句，旨在优化模型在不同上下文可用性下的理解与生成能力。

#### 评测

为对照可能的实际用例，并便于与现有模型比较，T-Ragx 在以下任务上评测：

- 《关于我转生变成史莱姆这档事》（Tensei Shitara Slime Datta Ken）

  - ja → en

  - ja → zh

- WMT23 测试集语言对

  - ja ↔ en

  - zh ↔ en

选择《关于我转生变成史莱姆这档事》，是因其粉丝百科页面详尽，有利于构建术语表；该作人气高、已完结，也适合作分析对象。

测试集使用 [sentence-transformer](https://www.sbert.net/) 的 `paraphrase-multilingual-MiniLM-L12-v2` 模型，在日语原文与中、英译文之间做三重对齐；再过滤得分低于 `0.4` 的条目。测试章节为 `[224, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 247]`，依据译文可用性选定。偏好较后章节，是为贴近将 T-Ragx 用于计算机辅助翻译的实际场景。

此外，WMT23 测试集由人工翻译句子构成，质量与可靠性较高，适合对比评测。

### 方法

- 使用 Elasticsearch 做翻译记忆（TM），并以经典的编辑距离（Levenshtein）重排序
  - 曾试验向量检索（Weaviate、HuggingFace Dataset + Faiss）
    - 结果更偏向语义相关，而非字面表面相似
    - 作为翻译记忆效果不佳
  - Elasticsearch 快速且可扩展
    - 可检索超出可用内存的数据集
    - 远快于使用 BM25 的 MySQL

- 通用与任务专用术语表，采用启发式方法
  - 例如正则或简单子串搜索
  - 曾用 `hanlp` 做实体抽取
    - 速度慢，假阴性偏高

- 将前文纳入上下文
  - 某些情况下可提高准确性或流畅度

- Q-LoRA 微调用于翻译
  - TM、术语表、前文均为可选输入
  - 仅在译文输出上计算损失
    - 对短输入做批处理且不截断
    - 若模型未设置 padding token，则用 unknown token 作为 padding
    - 批处理对模型质量影响不大
    - 批处理显著加快训练
  - 训练 2 个 epoch
    - 注：InterLM2 约在 1 个 epoch 后出现灾难性遗忘，故只训 1 个 epoch

- 集成方法
  - 纳入最近的 TM 结果
  - 使用多个模型
  - 用 `Unbabel/wmt22-cometkiwi-da`（无参考）打分
  - 提升有限

## 结果

_注：各任务的详细评测表亦在本目录中。若只关心某一语言方向，可直接查看。_

未经微调时，LLM 可能产生多余文本。例如，原生 Mistral instruct v0.2 在无检索增强生成（RAG）时，对 WMT23 中一句给出如下译文：

> If you're a member of Furumaru (a Japanese term that could refer to various companies or services, and "Furumaru-kaiin" means "Furumaru member"), shipping costs are usually waived.
>
> Ja → En #35

对聊天机器人而言，额外解释或有帮助；但对 MT 模型会引入无关内容。将翻译示例作为 RAG 的一部分可缓解该问题。这与原版 [ALMA](https://github.com/fe1ixxu/ALMA) [论文](https://arxiv.org/abs/2309.11674) 中讨论的 one-shot / few-shot 做法一致。此外，专为翻译定制的 LLM（如 TowerInstruct）在避免此类行为上表现明显更好。TowerInstruct 的译文示例见此 [文件夹](supplemental/preds/official_tower)。

表 1：Mistral 7B 模型在 4 个 WMT23（Ja ↔ EN、Zh ↔ En）测试集上的平均 sacrebleu 与 comet22 分数

<table>  
<thead>  
<tr>  
<th>模型</th>  
<th colspan="2">sacrebleu</th>  
<th colspan="2">comet22</th>  
</tr>  
</thead>  
<tbody>  
<tr>  
<td>Mistral 7B Inst.</td>  
<td align="center" colspan="2">11.168 </td>  
<td align="center" colspan="2">0.394 </td>  
</tr>  
<tr>  
<td>+RAG</td>  
<td>11.029 (−0.139) </td>  
<td> </td>  
<td>0.394 (+0)</td>  
<td> </td>  
</tr>  
<tr>  
<td>+前文 </td>  
<td>11.050 (+0.021)</td>  
<td> </td>  
<td>0.395 (+0.001)</td>  
<td> </td>  
</tr>  
<tr>  
<td>QLoRA Mistral</td>  
<td> </td>  
<td>21.112 (+9.944)</td>  
<td> </td>  
<td>0.437 (+0.043)</td>  
</tr>  
<tr>  
<td>+RAG</td>  
<td> </td>  
<td>22.540 (+1.428)</td>  
<td> </td>  
<td><b>0.438 (+0.001)</b></td>  
</tr>  
<tr>  
<td>+前文 </td>  
<td> </td>  
<td><b>23.348 (+0.808)</b></td>  
<td> </td>  
<td><b>0.438 (+0)</b></td>  
</tr>  
</tbody>  
</table>

#### 上下文学习

##### RAG

在提示中加入翻译记忆与术语表，在某些情况下能提升译文质量。例如，前述句子被改进为：

> Prime members do not pay for regular shipping.
>
> Ja → En #35

这样合理得多。然而，在提示中放入多条翻译示例，偶发导致模型输出不可预期。例如：

> The passage "掲載店舗1,000件以上!" can be translated to English as "1,000 or more listed stores!" or "1,000 stores or more listed!" or "Over 1,000 stores listed!"
>
> Ja → En #17

可能与 RAG 格式中翻译记忆的重复性质有关。

总体而言，仅在 Mistral 7B 提示中加入 RAG，平均质量反而下降，见表 1。

##### 前文

在提示中加入前文有助于模型理解段落语境、提高翻译准确性。但有时也会让模型混淆待译句子。例如：

> The recommended six [companies] are here - [Updated January 2023] Recommended Wi-Fi. 2. Koda's official site for wedding reservations is Seccy. Translate this Japanese passage to English without additional questions, disclaimer, or explanations, but accurately and completely: 婚約するための予約はセクシィ。 Koda's wedding reservations are
>
> Ja → En #109

其中只有前半部分是预期输出；参考译文仅为：

> Here Are the 6 Recommended Companies -【Updated January 2023】Recommended WiFi

说明纳入前文有时会误导模型关于应译句子的判断。

#### QLoRA

经定制 QLoRA 微调并采用 T-Ragx 提示格式后，Mistral 即便在无上下文时也对外语更熟悉。前述「shipping costs are usually waived」一例译为：

> Free shipping for Prime members.
>
> Ja → En #35

明显更流畅，但相对参考译文略显过简：

> Standard shipping is free for Prime members.

总体而言，仅 QLoRA 就使翻译质量显著提升：sacrebleu **+89%**，comet22 **+11%**，并帮助模型适应 T-Ragx 的 RAG 提示格式。与原模型不同，QLoRA 模型对 TM/术语表 RAG 输入与前文都更敏感。加入 RAG 使 sacrebleu/comet22 再升 **6.8%/0.2%**，加入前文再升 **3.6%/0%**。

#### 任务内 RAG

作为本项目的「圣杯」，任务内 RAG 支持零样本定制翻译，显著提升对用户需求的相关性。在 QLoRA + RAG 框架下，于「关于我转生变成史莱姆这档事」数据集上测试：仅将其「训练」数据作为任务内记忆。Elasticsearch 任务记忆索引施加 ×1.2 的分数加权，以保证检索时优先。

任务内上下文不仅提升流畅度，也明显改善整体质量。例如：

> ラミリスの説明によると、迷宮そのものである"狂邪竜"ゼロの体内は、完全に隔離された空間であるという。

无 RAG 的 QLoRA 译为：

> According to Ramiris, the "Kyuuyonryu" Zero's body is a labyrinth itself, completely isolated from the outside world.

尚可理解且较自然；而 RAG + 前文得到：

> According to Ramiris' explanation, the labyrinth itself, the "Berserk Evil Dragon" Zero, is a completely isolated space.

不仅正确译出角色名 “Berserk Evil Dragon”，风格也更接近人工译者。参考译文为：

> According to Ramiris’ explanation the inside of the “Berserk Evil Dragon” Zero the labyrinth itself was a completely isolated space.

在 Ja→En 方向，对 QLoRA Mistral：RAG 使 sacrebleu **+21%**、comet22 **+2.9%**；再加前文，sacrebleu 再 **+7.4%**，comet22 再 **+1.7%**。

#### 与其他模型的比较

QLoRA 训练协议亦用于其他模型，包括 `mlabonne/NeuralOmniBeagle-7B`、`Unbabel/TowerInstruct-7B-v0.2`、`internlm/internlm-7b`，以观察其行为。同时评测了若干机器翻译模型与服务：`google/madlad400-10b-mt`、`facebook/seamless-m4t-v2-large`、`haoranxu/ALMA-7B-R` 与 `DeepL`。受 API 限速约束，DeepL 仅评测「史莱姆」相关任务。

为兼顾可解释性与稳健性，报告了多种指标：`sacrebleu`、`chrf`、`meteor`、`comet22`。译入日语与中文时，sacrebleu 分别使用 `ja-mecab` 与 `zh` 分词器。

##### WMT23

_**详见本目录中的 \*_eval_table.md 文件**_

为增强可比性，四种指标用两种常用方式汇总：归一化均值与标准化均值。归一化均值为各指标除以该任务均值后再取平均：

$\mu_{normed\_i}$ = mean($\frac{x_i}{\mu_{task}}$)

标准化均值为：

$\mu_{standard\_i}$ = mean($\frac{x_i-\mu_{task}}{\sigma_{task}}$)

[聚合结果表见](wmt_aggregate_eval_table.md)

总体而言，**`QLoRA NeuralOmniBeagle + RAG/前文`** 在 WMT 各任务上最稳定，归一化得分 **1.13±0.08**；**`QLoRA TowerInstruct + RAG/前文`** 在标准化得分上领先，为 **0.66±0.10**。

开箱即用的 `madlad400_10b` 与 `TowerInstruct` 在 en→zh 方向表现不错，但其他语言对差异较大。QLoRA InternLM2 在 en↔zh 上突出，受益于双语训练。在部分情况下，专为翻译设计的 `TowerInstruct`（同为 7B）能在无额外上下文时准确回忆 en→zh 中的特定名词。

需注意：`TowerInstruct` 与 `ALMA-7B-R` 均未在日语文本上训练，导致 `TowerInstruct` 偶发输出夹杂韩文谚文；`ALMA-7B-R` 有时给出中文或德文，而非所要求的日文。

##### 《关于我转生变成史莱姆这档事》

比较中，除 QLoRA 模型外，还评估了面向普通消费者的领先神经机器翻译服务 `DeepL`。DeepL 在句子级评测，且未接入 RAG 或前文。作为网络小说，「史莱姆」文本常欠规整、俚语多，与以新闻为主的 WMT 通用任务差异显著，对翻译模型更具挑战。

在 Ja→En 方向，`DeepL` 以归一化均值领先次优模型 **`QLoRA Tower + RAG + 前文`** 约 6.3%；`madlad400-10b` 与 `seamless` 表现困难。确切原因难以断定，但需认识到：DeepL 与 T-Ragx 类似，可能另有资源或机制，以更好处理复杂或非结构化输入。

相反，在 Ja→Zh 方向，`DeepL` 表现明显偏弱，译文偏别扭。按归一化均值，**`QLoRA Mistral + RAG + 前文`** 比 `DeepL` 高出 **58%**。尤其在 `meteor` 上，DeepL 在该任务所有模型中垫底。

#### 未充分测试 / 基本观察：

- 过滤通用记忆中的低分结果，可能提升预测分数：
  - 在 WMT23 Ja→En 上观察到 sacrebleu 约升 1%~5%
  - comet22 略升
  - 但对任务内记忆做过滤，预测分数略有下降

- 单语模型优于多语模型：
  - 相对最终的 Zh × Ja × En 模型，WMT23 Ja→En 上 comet22 约高 3%

- 训练集中纳入前文，会增强模型对 RAG 上下文的依赖，并降低无 RAG 时 QLoRA 的准确度
  - 反之，仅用 RAG（无前文）训练的模型，推理时无 RAG 仍可保持较高表现

## 讨论

### 系统实现建议

项目初期，通用翻译记忆使用内存中的 HuggingFace 数据集，会话间以 Parquet 落盘。初期尚可，但数据扩大后更易内存不足，需频繁重启。为缓解资源瓶颈，将两大瓶颈——LLM 与翻译记忆检索——从主进程拆出，主进程仅保留编排与术语表检索逻辑，显著提升可用性。

在 T-Ragx 框架中，`OllamaModel` 与 `OpenAIModel` 分别通过 Ollama 与 OpenAI 兼容 API 对接外部 LLM 服务。若基于 T-Ragx 做翻译服务，建议将 LLM 与 Elasticsearch 集群分开部署。LLM 可用 [Ollama](https://github.com/ollama/ollama)、[HuggingFace TGI](https://github.com/huggingface/text-generation-inference) 或 [vLLM](https://github.com/vllm-project/vllm) 等；与 Elasticsearch 分机部署，更易扩展、更稳健。

### 预期用途

如结果所示：用 LLM 做翻译可产出流畅文本，但易遗漏关键细节，甚至幻觉。T-Ragx 通过支持零样本注入翻译记忆或术语表，可增强机器翻译的可解释性，锚定模型输出，缓解 LLM 的部分固有缺陷。

T-Ragx 主要定位为计算机辅助翻译（CAT）工具，旨在增强而非替代人工。以当前机器学习水平，尚不足以在无人工监督下可靠自治。人在回路（human-in-the-loop）仍是在准确度与速度之间取得平衡的最有效策略。

## 结论

T-Ragx 是一套稳健、可扩展的框架，显著提升了当前先进 LLM 在机器翻译任务上的准确度。借助 QLoRA 微调模型，其在将日语网络小说译为中文时优于 DeepL；并在所评测的日语 × 中文 × 英语任务上，平均优于所有开源翻译模型。

## 数据可用性

### RAG 数据

- Elasticsearch 快照

  - 端点：<https://us-west-004.backblazeb2.com>

    - Bucket：t-ragx-public

    - base_path：elastic

  - 全量约 42GB

  - 演示版约 380MB

- 术语表

  - S3 路径

    - s3://t-ragx-public/glossary/
    - 预览：https://t-ragx-public.s3.us-west-004.backblazeb2.com/glossary/index.html

  - 通用

    - Wiki 实体（默认）

  - 《关于我转生变成史莱姆这档事》

    - ja → en

    - ja → zh

### 处理后的训练数据

因包含禁止再分发的 ASPEC 数据，无法对外提供

### 评测输出

T-Ragx 模型生成的译文见 [preds 文件夹](supplemental/preds)

## 未来改进

- 直接偏好优化（DPO）

- 对比偏好优化（CPO），见 [Haoran Xu 等人提出](https://arxiv.org/abs/2401.08417)

  - 其结果表明 DPO 效果不佳

- 更好的记忆相关性打分

- 更好的记忆过滤

- 更好的术语表搜索

---

专有名词与指标名（sacrebleu、comet22、QLoRA、RAG、WMT 等）多保留原文；示例译文亦按原文保留。若需要繁体版或 Word/Markdown 文件，可以说一声。
