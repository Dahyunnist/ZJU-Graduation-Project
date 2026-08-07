# Data Shapley: Equitable Valuation of Data for Machine Learning（中文梳理）

## 文献信息

Ghorbani A., Zou J. ICML 2019，PMLR 97：2242–2251。原文：[PDF](../论文PDF/01_Ghorbani2019_Data_Shapley.pdf)；[PMLR](https://proceedings.mlr.press/v97/ghorbani19c.html)；[代码](https://github.com/amiratag/DataShapley)。

## 定义

给定训练数据子集S的效用V(S)（例如验证集Accuracy或负Log-loss），一条训练样本的Data Shapley值是它加入所有可能前置子集时带来的边际效用的加权平均。它满足效率、对称、虚玩家和可加性等合作博弈性质。

## 算法

精确计算需要枚举子集，代价指数级。论文提出TMC-Shapley：随机采样排列，按顺序增量训练模型；当当前子集效用已接近全数据效用时截断后续贡献，以Monte Carlo平均近似各样本价值。

## 本课题用法

把真实与合成记录共同作为训练样本，以独立、只含真实记录的验证/测试集定义效用。比较两类样本价值分布、负价值比例，以及污染率上升时价值排序和模型性能的变化。这样“估值影响”成为可检验问题，而不是笼统讨论。

## 局限与复现

结果依赖模型、验证集和效用函数；训练随机性会造成方差，计算也昂贵。官方代码依赖较旧环境，建议在小数据子集上自行实现TMC-Shapley并与精确小N结果核对，作为第二优先级复现。

