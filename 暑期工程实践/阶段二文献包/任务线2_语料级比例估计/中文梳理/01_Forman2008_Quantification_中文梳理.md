# Quantifying Counts and Costs via Classification（中文梳理）

## 文献信息

Forman G. Data Mining and Knowledge Discovery，2008，17：164–206。下载版本为内容更完整的HP技术报告。原文：[PDF](../论文PDF/01_Forman2008_Quantifying_Counts_Costs_Trends.pdf)；[DOI](https://doi.org/10.1007/s10618-008-0097-y)。

## 问题定义

量化学习关注测试集合中各类别的比例，而非每个个体标签是否正确。论文指出，分类器即使个体准确率较高，只要假阳性率或假阴性率存在系统偏差，“分类后计数”也会在类别比例变化时产生很大误差。

## 方法谱系

CC直接统计预测为正的比例；Adjusted Count用验证集估计TPR/FPR并反演校正；阈值选择方法寻找更稳定的工作点；Median Sweep对多个阈值的校正估计取中位数；Mixture Model把测试输出看作正负类分布的混合。

## 本课题意义

把“正类”定义为合成记录，测试集合正类比例就是污染率。该文证明任务线二不能被任务线一吞并：最优检测阈值未必产生最准确的污染率。

## 复现要点

复现CC、ACC、Median Sweep，并用独立校准集估计TPR/FPR；当TPR≈FPR时校正会数值不稳定，应截断到[0,1]并报告失败率。评价使用MAE、RMSE、偏差和不同真实污染率下的误差曲线。

