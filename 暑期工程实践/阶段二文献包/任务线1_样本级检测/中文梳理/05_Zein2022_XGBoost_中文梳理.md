# Tabular Data Generation: Can We Fool XGBoost?（中文梳理）

## 文献信息

Zein E.-H., Urvoy T. NeurIPS 2022 Table Representation Learning Workshop。原文：[PDF](../论文PDF/05_Zein2022_Can_We_Fool_XGBoost.pdf)；[OpenReview](https://openreview.net/forum?id=tTQzJ6TJGVi)。

## 方法

论文用XGBoost训练真实/合成二分类器，测试多种先进生成模型是否能制造“不可区分”的表格记录。作者进一步分析特征重要性，发现混合类型列、分布不规则的数值列和不恰当编码常暴露生成痕迹，并探索可逆的逐列编码来改善生成质量。

## 主要结论

即使常见边际统计或下游任务指标看起来不错，强树模型仍可近乎完美地区分某些合成数据。这说明低阶统计相似不代表联合分布一致，也证明XGBoost是非常有竞争力的同表检测基线。

## 本课题用法

C2ST-XGBoost列为必复现算法，并输出SHAP/增益特征重要性定位“检测依据”。同时需要做伪影审计：移除行号、源标志、格式差异；统一真实和合成记录的类型、精度、缺失值表示。否则模型可能只学到工程痕迹。

## 局限

该工作更偏向生成器质量评价和同表场景，不能代表跨表检测。Workshop论文的实验范围也不足以单独支撑SOTA结论，应与Kindji系列协议联合使用。

