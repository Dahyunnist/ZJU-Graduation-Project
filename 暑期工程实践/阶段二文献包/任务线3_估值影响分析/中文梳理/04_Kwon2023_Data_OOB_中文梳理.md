# Data-OOB: Out-of-Bag Estimate as a Simple and Efficient Data Value（中文梳理）

## 文献信息

Kwon Y., Zou J. ICML 2023，PMLR 202：18135–18152。原文：[PDF](../论文PDF/04_Kwon2023_Data_OOB.pdf)；[PMLR](https://proceedings.mlr.press/v202/kwon23e.html)。

## 方法

Bagging中每个弱学习器只在bootstrap样本上训练。对某条未被该弱学习器抽中的记录，可获得out-of-bag预测。Data-OOB聚合这些现成预测来衡量该记录对模型的有益或有害程度，无需像Shapley那样重新训练大量子集模型。

## 优势与结果

方法计算成本低，可扩到百万样本，并具有与infinitesimal jackknife影响函数相关的理论解释。论文在错误标签识别、训练样本筛选等任务上优于多种高成本估值基线。

## 本课题用法

它非常适合Random Forest/Bagging树模型和中等以上数据规模，列为第一优先级。比较真实与合成记录的Data-OOB分布，并做低价值删除/高价值保留曲线，观察删除合成低价值样本是否改善纯真实测试集性能。

## 局限

估值依赖bagging模型，未必代表其他下游模型；少数样本可能缺少足够OOB预测。需固定弱学习器数量并报告每条记录的OOB覆盖次数，与KNN-Shapley结果做排序相关性比较。

