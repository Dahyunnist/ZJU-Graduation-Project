# A Review on Quantification Learning（中文梳理）

## 文献信息

González P., Castaño A., Chawla N. V., del Coz J. J. ACM Computing Surveys，2017，50(5)。原文：[PDF](../论文PDF/06_Gonzalez2017_Review_Quantification_Learning.pdf)；[DOI](https://doi.org/10.1145/3117807)。

## 综述框架

论文系统界定quantification：输出集合中的类别分布，而不是个体分类。方法大致包括分类后计数、基于混淆率的校正、概率校正、分布匹配、直接优化量化损失及多类扩展。

## 关键认识

分类与量化的损失函数不同；单样本准确率高并不保证总体比例准确。量化实验应在多种目标比例上生成测试集合，并使用绝对误差、相对误差或其平滑版本，而不是沿用分类Accuracy。

## 本课题用法

这篇综述用于搭建任务线二理论部分和算法谱系，不作为对比算法。综述中早期算法需回到原始论文引用；研究方案可据此解释为什么CC/PCC只是基线、ACC/PACC和分布匹配是专用量化方法。

## 精读检查点

- 整理各方法对prior shift、概率校准、验证集的假设。
- 提取二分类比例误差公式及边界处理。
- 将综述分类映射到本课题最终复现清单，避免方法堆砌。

