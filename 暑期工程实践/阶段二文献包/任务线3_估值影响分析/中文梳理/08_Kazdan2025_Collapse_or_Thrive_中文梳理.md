# Collapse or Thrive: Perils and Promises of Synthetic Data in a Self-Generating World（中文梳理）

## 文献信息

Kazdan J., Schaeffer R., Dey A., et al. ICML 2025，PMLR 267：29469–29494。原文：[PDF](../论文PDF/08_Kazdan2025_Collapse_or_Thrive.pdf)；[PMLR官方页面](https://proceedings.mlr.press/v267/kazdan25a.html)。

## 研究问题

论文进一步检验模型崩塌是否取决于真实数据和合成数据的使用流程，而不只是“是否使用合成数据”。它比较全量替换、真实与合成持续累积、固定预算抽样三种训练工作流，并覆盖多元高斯、核密度估计和语言模型微调。

## 主要结论

逐代用纯合成数据替换真实数据会发生崩塌；保留历史真实数据并累积合成数据时，测试损失可以保持稳定；固定训练预算下从不断扩大的混合集合抽样，则表现为较缓慢而非爆炸式退化。因此合成数据的影响高度依赖数据保留与抽样机制。

## 与本课题的关系

这篇ICML正式论文是Gerstgrasser等COLM工作的更完整后续，应成为第三条任务线的重要影响证据。它不是记录级估值算法，但直接指导实验：相同污染比例也要区分一次性混合、逐代替换和累积保留，不能把所有情景混为一谈。

## 复现定位

主线先完成单代污染率—下游效用曲线、KNN-Shapley和Data-OOB；随后可用较轻量的高斯/KDE实验复现三种工作流趋势。若资源不足，只把它作为扩展实验与讨论依据，不冒充“已复现估值算法”。

