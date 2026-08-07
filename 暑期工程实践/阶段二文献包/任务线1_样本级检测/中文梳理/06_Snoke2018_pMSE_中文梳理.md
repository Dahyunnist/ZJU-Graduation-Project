# General and Specific Utility Measures for Synthetic Data（中文梳理）

## 文献信息

Snoke J., Raab G. M., Nowok B., Dibben C., Slavković A. JRSS A，2018，181(3)：663–688。原文：[PDF](../论文PDF/06_Snoke2018_General_and_Specific_Utility_Measures.pdf)；[DOI](https://doi.org/10.1111/rssa.12358)。

## 核心概念

论文区分general utility与specific utility。前者判断合成数据整体分布是否接近真实数据；后者判断具体统计分析所得系数、区间等是否接近。pMSE将真实/合成来源作为标签，拟合倾向得分模型，再计算预测来源概率偏离混合先验的均方误差。

## 关键贡献

作者针对合成数据情形推导正确合成模型下pMSE的期望分布，并提出标准化思路，使不同数据规模和倾向模型复杂度下的结果更可比较。同时比较置信区间重叠、汇总统计标准差异等specific utility指标。

## 与检测的边界

pMSE依赖一个来源分类器，但最终输出是表级总体可区分度，并非专门为未知混合表输出每条记录标签。因此它适合任务线一的经典统计基线和质量度量，也可辅助解释任务线三的效用，却不能代替记录级检测器或污染比例估计器。

## 复现要点

实现原始pMSE、零模型期望校正/标准化pMSE，并固定倾向模型自由度。报告不同分类器下结果，避免把“模型太弱”误解为合成质量高。

