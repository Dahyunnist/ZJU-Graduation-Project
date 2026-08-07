# Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms（中文梳理）

## 文献信息

Jia R., Dao D., Wang B., et al. PVLDB 2019，12(11)：1610–1623。原文：[PDF](../论文PDF/02_Jia2019_KNN_Shapley.pdf)；[PVLDB PDF](https://www.vldb.org/pvldb/vol12/p1610-jia.pdf)。

## 核心贡献

论文针对无权KNN分类/回归推导精确Shapley递推。对每个验证点，将训练样本按距离排序后从远到近递推贡献，即可在O(N log N)量级得到全部训练样本价值，相比通用指数枚举大幅降低成本。

## 优势

算法确定性强、容易用小样本暴力枚举核对，能够扩展到较大数据集。虽然估值绑定KNN任务，但非常适合毕业设计先建立可靠的“合成样本价值是否较低/为负”证据。

## 本课题复现

列为任务线三第一优先级。先统一连续变量标准化、类别变量编码和距离度量；以纯真实验证集计算KNN-Shapley，比较真实/合成样本价值分布、AUROC（用价值区分来源仅作诊断）、低价值删除曲线及Spearman稳定性。

## 局限

结论只对应KNN和指定验证集，不能直接外推到XGBoost或神经网络；高维one-hot距离可能失真。应与Data-OOB和实际下游性能曲线交叉验证。

