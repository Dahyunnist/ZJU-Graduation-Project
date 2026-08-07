# The Real Deal Behind the Artificial Appeal（中文梳理）

## 文献信息

Decruyenaere A., Dehaene H., Rabaey P., et al. UAI 2024，PMLR 244：966–996。原文：[PDF](../论文PDF/07_Decruyenaere2024_Inferential_Utility_Tabular_Synthetic.pdf)；[PMLR](https://proceedings.mlr.press/v244/decruyenaere24a.html)。

## 研究问题

论文不只问合成数据是否能训练预测模型，还问用合成数据做统计推断时，估计值、置信区间和假设检验是否可靠。作者强调“预测效用”和“推断效用”必须区分。

## 主要发现

模拟研究表明，把合成数据当作真实独立观测进行朴素推断，即使点估计看似无偏，也可能严重低估不确定性并导致一类错误率膨胀。合成过程带来的额外不确定性不能被忽略。

## 与本课题的关系

第三条任务线不能只比较下游分类Accuracy。若数据面向分析使用，还应加入回归系数偏差、置信区间覆盖率和type-I error等指标。合成污染比例上升时，这些指标可能比Accuracy更早恶化。

## 复现建议

选择一个简单线性/逻辑回归模拟：真实数据存在已知零效应和非零效应，按比例加入合成记录，重复生成与拟合，统计偏差、覆盖率和假阳性率。该实验可作为Data Shapley类估值之外的“影响分析”补充，而非对比算法。

