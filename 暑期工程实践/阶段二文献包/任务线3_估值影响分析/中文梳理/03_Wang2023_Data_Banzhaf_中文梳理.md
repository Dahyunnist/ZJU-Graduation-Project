# Data Banzhaf: A Robust Data Valuation Framework（中文梳理）

## 文献信息

Wang J. T., Jia R. AISTATS 2023，PMLR 206：6388–6421。原文：[PDF](../论文PDF/03_Wang2023_Data_Banzhaf.pdf)；[PMLR](https://proceedings.mlr.press/v206/wang23e.html)。

## 研究动机

模型训练含随机初始化、SGD和采样噪声，同一数据子集的测得效用会波动，进而导致Shapley或leave-one-out价值排序不稳定。论文以safety margin刻画价值排序抵抗噪声的能力。

## 方法与结论

Banzhaf值对所有不含目标样本的子集等概率取边际贡献平均。论文证明其在一类semivalue中具有最大的安全边际，并给出随机采样和最大复用训练结果的估计方式。实验显示它在噪声效用下更能稳定识别有害/有益样本。

## 本课题意义

合成数据估值比较很容易受随机训练波动干扰；若只运行一次TMC-Shapley就下结论，证据不稳。Data Banzhaf可作为稳健估值SOTA候选，并促使所有估值算法报告多随机种子排名一致性。

## 复现定位

第二优先级。先在较小训练集上采样子集，保存每次模型效用并复用计算；比较Banzhaf、TMC-Shapley、LOO在识别已知污染样本和删除低价值样本时的表现与方差。

