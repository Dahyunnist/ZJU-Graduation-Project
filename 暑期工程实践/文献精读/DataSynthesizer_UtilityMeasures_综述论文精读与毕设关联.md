# DataSynthesizer、Utility Measures 与合成表格数据综述精读拆解及毕设关联说明

本文针对说明书中第 4、5 条经重新检索后最可能对应的三篇文献进行梳理：

1. Ping H., Stoyanovich J., Howe B. **DataSynthesizer: Privacy-Preserving Synthetic Datasets**. SSDBM 2017.
2. Snoke J., Raab G., Nowok B., Dibben C., Slavković A. **General and Specific Utility Measures for Synthetic Data**. JRSS Series A, 2018.
3. Shi R. S., Wang Y., Du M., Shen X., Chang Y., Wang X. **A Comprehensive Survey of Synthetic Tabular Data Generation**. arXiv 2025.

这三篇文献分别补足你毕设中的三个部分：

- **DataSynthesizer**：早期隐私保护合成表格数据系统，帮助理解统计型合成器和隐私-效用权衡。
- **General and Specific Utility Measures**：合成数据质量/效用评估的经典方法，尤其是 pMSE、general utility、specific utility。
- **A Comprehensive Survey**：近年合成表格数据生成综述，帮助把 CTGAN/TVAE、扩散模型、LLM 生成放进完整技术谱系。

---

## 1. DataSynthesizer: Privacy-Preserving Synthetic Datasets

### 1.1 基本信息

正式引用：

> Haoyue Ping, Julia Stoyanovich, Bill Howe.  
> **DataSynthesizer: Privacy-Preserving Synthetic Datasets**.  
> Proceedings of the 29th International Conference on Scientific and Statistical Database Management, SSDBM 2017.  
> DOI: https://doi.org/10.1145/3085504.3091117

说明书中原写法是 `Ping H., et al. Data Synthesis based on Data Utility. ICDE 2019`，但经检索并没有找到完全对应的论文。若按照作者 `Ping H.` 和“早期合成数据质量评估/生成系统”判断，最可能对应的是这篇 DataSynthesizer。

### 1.2 这篇文章解决什么问题

这篇论文关注的是：

> 当真实数据涉及隐私、法律或共享协议限制时，如何生成一份结构和统计特征类似、但不泄露个人敏感信息的合成数据集？

在许多场景中，数据拥有者不能直接把原始数据交给外部分析人员。例如医院、政府机构或社会科学数据通常涉及隐私。传统数据共享协议周期长、成本高，导致协作困难。

DataSynthesizer 的目标是让数据拥有者可以先发布一份合成数据，使外部研究人员能够先开发方法、熟悉数据结构、验证分析流程，同时降低隐私泄露风险。

### 1.3 核心思路

一句话概括：

> DataSynthesizer 先从原始表格中学习带差分隐私保护的数据摘要，再从该摘要中采样生成合成数据，并提供可视化检查工具比较真实数据与合成数据的统计相似性。

系统由三个模块组成：

1. **DataDescriber**：读取原始 CSV，推断字段类型、取值域、分布、缺失率和属性相关性，并在需要时加入差分隐私噪声。
2. **DataGenerator**：根据 DataDescriber 生成的数据描述文件采样，生成指定数量的合成记录。
3. **ModelInspector**：比较原始数据和合成数据的统计摘要，例如直方图、类别分布、KL divergence、pairwise mutual information 和 Bayesian network。

### 1.4 三种生成模式

DataSynthesizer 提供三种模式。

**Random mode**：只生成类型一致的随机值。例如年龄仍是整数，日期仍是日期，但不保证分布相似。它隐私强，但效用很低。

**Independent attribute mode**：每列独立建模。类别列用频率分布，数值列用直方图，然后对每列独立采样。它能保留单列分布，但不能保留列与列之间的关系。

**Correlated attribute mode**：用差分隐私 Bayesian network 建模属性之间的依赖关系，再按网络顺序采样。它比独立模式更能保留多列相关性。

### 1.5 和你毕设的关系

DataSynthesizer 对你的毕设有三点价值。

第一，它是一个早期统计型合成数据生成器。你的基准除了 CTGAN/TVAE 这类深度生成器，也可以考虑加入 GaussianCopula、Bayesian network 或 DataSynthesizer 这类统计方法作为“较易检测”的合成污染来源。

第二，它展示了如何检查合成数据和真实数据是否“像”。ModelInspector 使用直方图、KL divergence、相关矩阵和 Bayesian network 对比，这些思想可以直接服务于你的统计检测基线和数据卡。

第三，它强调隐私与效用的权衡。差分隐私噪声越强，隐私越好，但统计相似性和下游效用可能下降。这和你的毕设中“合成污染对下游效用/数据价值的影响”直接相关。

---

## 2. General and Specific Utility Measures for Synthetic Data

### 2.1 基本信息

正式引用：

> Joshua Snoke, Gillian M. Raab, Beata Nowok, Chris Dibben, Aleksandra Slavković.  
> **General and Specific Utility Measures for Synthetic Data**.  
> Journal of the Royal Statistical Society: Series A, 2018.  
> DOI: https://doi.org/10.1111/rssa.12358

这篇论文非常适合补充说明书里“Data Utility / 早期合成数据质量评估”这一方向。

### 2.2 这篇文章解决什么问题

这篇论文关注的是：

> 合成数据生成之后，如何判断它是否足够有用？

合成数据的目标不是和原始数据逐行相同，而是希望在保护隐私的同时保留足够的分析价值。因此需要评价合成数据的 utility。

论文把 utility 分为两类：

- **General utility**：合成数据整体分布是否接近真实数据。
- **Specific utility**：针对某个具体分析任务，合成数据得出的结论是否接近真实数据。

### 2.3 核心思路

一句话概括：

> 论文用 propensity score mean-squared error（pMSE）衡量真实数据和合成数据的整体可区分性，并用置信区间重叠与标准化系数差异衡量具体分析结果的一致性。

### 2.4 General utility：pMSE

pMSE 的思想非常适合你的课题。

做法是：

1. 把真实数据和合成数据堆在一起。
2. 给真实数据标记 `0`，合成数据标记 `1`。
3. 训练一个分类器预测某条记录来自真实数据还是合成数据。
4. 如果分类器很难区分，说明合成数据整体分布更接近真实数据。
5. 如果分类器很容易区分，说明合成数据与真实数据差异较大。

pMSE 衡量的是预测概率与理论比例之间的均方差。若真实和合成样本数量相同，理想情况下每条记录被判为合成的概率应接近 0.5。

这与本题“样本级 real/synthetic 检测”高度相关。你的检测任务可以看成把 pMSE 的“合成数据质量评估视角”反过来用：如果一个分类器能稳定地区分真实和合成数据，就说明存在可检测的合成污染痕迹。

### 2.5 Specific utility

Specific utility 关注具体分析结果是否一致。

论文使用了两类指标：

- **Confidence interval overlap**：真实数据和合成数据中同一统计量/模型系数的置信区间重叠程度。
- **Standardized difference**：真实数据估计值和合成数据估计值之间的标准化差异。

对你的毕设来说，这对应任务三：

> 合成污染是否改变下游模型性能或分析结论？

你可以用更机器学习化的指标替代原文统计指标，例如 AUROC、F1、accuracy、回归 R² 等。

### 2.6 和你毕设的关系

这篇论文对你的价值很直接：

1. 它提供了 real-vs-synthetic 分类器作为合成数据质量评估的理论依据。
2. 它解释了为什么“能否区分真实/合成”本身就是一种 utility/fidelity 信号。
3. 它提醒你不能只看单一指标，应同时关注整体分布相似性和具体下游任务表现。
4. 它可以作为你阶段一“SDMetrics + sklearn 分类器跑 AUROC”的理论来源之一。

---

## 3. A Comprehensive Survey of Synthetic Tabular Data Generation

### 3.1 基本信息

正式引用：

> Ruxue Shi, Yili Wang, Mengnan Du, Xu Shen, Yi Chang, Xin Wang.  
> **A Comprehensive Survey of Synthetic Tabular Data Generation**.  
> arXiv 2025.  
> https://arxiv.org/abs/2504.16506

这篇综述适合替代说明书第 5 条中“近年相关综述 / 基准，检索确认”的部分。

### 3.2 这篇文章解决什么问题

这篇综述关注的是：

> 合成表格数据生成领域有哪些主要方法、完整流程、评价方式、应用场景和未来挑战？

它试图把分散的研究整合成一个完整框架，尤其补充了近年来扩散模型和 LLM 方法在表格生成中的发展。

### 3.3 核心思路

一句话概括：

> 该综述将合成表格数据生成方法划分为传统方法、扩散模型方法和 LLM 方法，并从生成、后处理、评估、应用和未来方向多个层面梳理整个技术管线。

### 3.4 论文总结的主要挑战

综述指出表格生成面临几类挑战。

**数据质量挑战**：

- 数据量小。
- 类别不平衡。
- 缺失值普遍。

**表格固有特性挑战**：

- 数值列和类别列混合。
- 特征之间依赖复杂而稀疏。
- 单列中可能存在混合类型。

**分布复杂性挑战**：

- 数值特征非高斯。
- 有偏态、长尾、多峰分布。
- 不同列有不同尺度和语义。

这些挑战和 CTGAN 论文中的问题高度一致，也解释了为什么本题不能只做简单随机生成。

### 3.5 方法分类

综述把方法分为三大类。

**传统生成方法**包括 Copula、Bayesian network、SMOTE、Synthpop、VAE、GAN、CTGAN、TVAE、TableGAN、CTAB-GAN 等。

**扩散模型方法**包括 TabDDPM、TabDiff、TabSyn 等，优点是训练稳定、能处理复杂分布，缺点是计算成本通常较高。

**LLM-based 方法**包括 prompt-based 和 fine-tuning 两类，如 GReaT、TabuLa、TabMT、AIGT、DP-LLMTGen 等。它们的优势是可以利用语言模型的语义和常识知识，但也可能产生幻觉、不合法值和格式错误。

### 3.6 后处理与评估

综述特别强调生成后的后处理：

- sample filtering：过滤不合理样本。
- sample correction：根据规则或约束修正样本。
- label enhancement：修正合成样本标签。

这对你的毕设有启发。你构建污染基准时，合成数据不能太假，也不能有明显越界或非法类别，否则检测会变得过于简单。

评估方面，综述将指标分为：

- **ML efficiency**：TSTR，即用合成数据训练，在真实测试集上测试。
- **Fidelity**：列分布、联合分布、Wasserstein、KS、JSD、TVD 等。
- **Alignment**：是否违反领域约束或常识规则。
- **Privacy**：DCR、membership inference、attribute inference 等。

### 3.7 和你毕设的关系

这篇综述可以帮你确定整个课题的技术版图。

它说明：

- CTGAN/TVAE 是经典起点，但不是全部。
- 近年扩散模型和 LLM 生成正在成为重要方向。
- 合成数据评估不能只看统计相似性，还要看下游效用、约束一致性和隐私风险。
- 后处理和质量过滤是构建高质量基准的必要步骤。

对你的毕设而言，这篇综述适合放在文献综述中作为“合成表格数据生成发展脉络”的总览文献，并帮助说明为什么你的基准要覆盖多类生成器、多个污染比例和多种评价维度。

---

## 4. 三篇文献如何补充已有五篇

已有五篇文献中，CTGAN/TVAE 负责深度生成器，模型坍缩负责动机，ZeroED 负责 LLM 检测范式，Data Shapley 负责估值工具。

这三篇补充后，文献体系更完整：

- **DataSynthesizer** 补充早期统计型、差分隐私型合成器。
- **General and Specific Utility Measures** 补充合成数据 utility/fidelity 的经典评价方法。
- **A Comprehensive Survey** 补充近年方法全景，尤其是扩散模型、LLM 生成、后处理与评估。

这样你的文献综述可以形成如下结构：

```text
合成表格数据生成：DataSynthesizer -> CTGAN/TVAE -> 扩散模型/LLM 综述
合成数据质量评估：General/Specific Utility -> SDMetrics -> XAI 诊断
合成污染风险动机：模型坍缩
LLM 检测范式：ZeroED
估值影响分析：Data Shapley
```

---

## 5. 可写进综述的整合段落

可以这样写：

> 早期合成表格数据研究主要围绕隐私保护和统计相似性展开。Ping 等人提出的 DataSynthesizer 通过 DataDescriber、DataGenerator 和 ModelInspector 三个模块，从原始表格中学习带差分隐私保护的数据摘要，并生成结构和统计特征相似的合成数据，为隐私敏感数据共享提供了实用工具。与之相配套，Snoke 等人系统讨论了合成数据的 general utility 和 specific utility，提出基于 propensity score mean-squared error 的整体效用评估方法，并使用置信区间重叠和标准化系数差异衡量具体分析任务的一致性。这些工作为评估合成数据是否“像真实数据”以及是否支持下游分析提供了基础。
>
> 近年来，合成表格数据生成方法不断扩展。Shi 等人的综述将现有方法归纳为传统生成方法、扩散模型方法和 LLM-based 方法，并指出表格数据由于混合类型、类别不平衡、缺失值、复杂依赖关系和非高斯分布等特点，仍然难以高质量生成。该综述还强调了后处理、ML utility、fidelity、alignment 与 privacy 等评估维度。本文在上述工作的基础上，不再只关注如何生成合成数据，而进一步研究合成数据混入真实数据后的检测、比例估计及其对数据价值的影响。

