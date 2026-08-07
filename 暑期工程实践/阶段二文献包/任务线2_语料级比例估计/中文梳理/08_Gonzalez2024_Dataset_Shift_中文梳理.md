# Binary Quantification and Dataset Shift: An Experimental Investigation（中文梳理）

## 文献信息

González P., Moreo A., Sebastiani F. Data Mining and Knowledge Discovery，2024，38：1670–1712。原文：[PDF](../论文PDF/08_Gonzalez2024_Binary_Quantification_Dataset_Shift.pdf)；[出版社页面](https://link.springer.com/article/10.1007/s10618-024-01014-1)；[代码](https://github.com/pglez82/quant_datasetshift)。

## 核心问题

量化算法常在类别先验变化下研究，但真实部署还可能出现covariate shift和concept shift。论文系统控制不同偏移类型，研究各量化方法何时失效。

## 主要结论

不同算法对偏移类型的敏感性显著不同，没有方法在所有shift下都稳健。仅在人工改变类别比例的实验中表现好，不能保证跨域场景可靠。应把“偏移类型”视为量化评测的核心维度。

## 与本课题的直接关系

固定表和生成器、只改变真实/合成混合比例，近似prior shift；更换生成器会改变合成类条件分布；更换真实表甚至改变特征空间。这意味着任务线二必须至少区分同域比例变化、跨生成器、跨表三档，而不能只随机混合CTGAN数据。

## 复现安排

使用官方代码理解shift构造和评价；在本课题中对每个量化器报告各协议MAE、偏差和最差组表现。若某算法只在prior shift下有效，应把它写成适用范围而非失败。

