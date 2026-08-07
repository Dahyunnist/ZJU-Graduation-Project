# Continuous Sweep for Binary Quantification Learning（中文梳理）

## 文献信息

Kloos K., Karch J. D., Meertens Q. A., Scholtus S., de Rooij M. Journal of Classification，2026-05-20。原文：[PDF](../论文PDF/09_Kloos2026_Continuous_Sweep.pdf)；[出版社页面](https://link.springer.com/article/10.1007/s00357-026-09548-3)。

## 方法创新

Continuous Sweep由Median Sweep改造而来：用参数化类条件分数分布估计各阈值的TPR/FPR，并用一组Adjusted Count估计的均值代替中位数。这样可以推导偏差与方差，并据此选择使方差最小的阈值集合。

## 实验结论

三组模拟中，在分布设定合理时通常优于Median Sweep；分布错设会削弱优势。在LeQua2022的5000个测试集合上，它平均优于Median Sweep、与DyS接近，但略逊于SLD/EMQ。论文因此将其定位为有理论依据、具竞争力的新二分类量化器，而非无条件最优。

## 限制

方法当前只针对二分类，并依赖能较好拟合正负分数的参数分布。底层分类器—量化器组合也影响公平比较。论文还提醒旧QuaPy的Median Sweep实现曾有bug，复现必须固定新版。

## 本课题安排

这是检索截止日最近发表的新方法，直接体现滚动更新要求。列为第二优先级SOTA候选：先核对作者公开代码，复现同域污染率估计；再观察跨生成器时参数分布失配是否导致退化，并与SLD、DyS和Median Sweep同表比较。

