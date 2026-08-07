# A Comparative Evaluation of Quantification Methods（中文梳理）

## 文献信息

Schumacher T., Strohmaier M., Lemmerich F. Journal of Machine Learning Research，2025，26(55)：1–54。原文：[PDF](../论文PDF/07_Schumacher2025_Comparative_Quantification.pdf)；[JMLR官方页](https://jmlr.org/papers/v26/21-0241.html)。

## 研究设计

论文在大量真实数据集和统一协议下比较24种量化方法，覆盖classify-count-correct、直接学习与分布匹配等类别，并考察不同数据规模、类别比例和分类器条件。

## 主要启示

不存在所有数据条件下都绝对最优的方法；Median Sweep、TSMax、DyS/HDy、Forman Mixture Model、Friedman类方法在不同设置中具有竞争力。强方法的排序还会受底层分类器和实现版本影响。

## 对本课题的作用

它提供“为什么选择这些对比算法”的证据：既保留CC/PCC等弱但必要的基线，也必须加入EMQ、DyS/HDy、Median Sweep等经典强方法。不能从几十个方法中随意挑几个容易实现的。

## 局限与精读重点

该比较研究不是合成表格污染专用，数据偏移机制与本课题的跨生成器、跨表偏移不同。应详细记录其数据生成协议、底层分类器、调参方式和宏平均方法，然后在本课题协议中重新验证结论。

