# Quantification via Probability Estimators（中文梳理）

## 文献信息

Bella A., Ferri C., Hernández-Orallo J., Ramírez-Quintana M. J. ICDM 2010：737–742。原文：[PDF](../论文PDF/02_Bella2010_Quantification_via_Probability_Estimators.pdf)；[DOI](https://doi.org/10.1109/ICDM.2010.75)。

## 核心方法

PCC不对样本概率做硬阈值，而是直接平均正类后验概率得到集合比例。PACC进一步用验证数据中正负类的平均预测概率进行校正，可理解为硬混淆矩阵校正ACC的“软”版本。

## 主要认识

概率估计保留了分类器置信度信息，通常比单纯硬计数更平滑；但它依赖概率校准，遇到训练/测试分布偏移或过度自信模型时仍会偏差。因而“使用概率”不是自动可靠，校准方案也应成为实验变量。

## 本课题用法

检测器输出合成概率后，PCC/PACC是几乎零额外成本的基础比例估计器。应与CC/ACC并列，比较Platt、isotonic或原始概率，并确保概率校准数据不进入最终测试集合。

## 精读检查点

- 推导PACC二分类公式并核对分母稳定性。
- 区分概率校准与比例校正两件事。
- 观察不同检测器AUROC相近时，比例误差是否因校准不同而显著变化。

