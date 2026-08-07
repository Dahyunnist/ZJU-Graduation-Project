# 论文设定与本地runthrough范围

本轮所有结果均为`runthrough/smoke`，用于证明代码闭环，不能作为论文正式性能值。

- D1/D2：本地仅三表mini fold，论文使用更完整跨表语料和协议。
- D3：本地为论文对齐自实现、CPU tiny网络；未声称官方代码或论文AUC复现。
- 量化：每比例3个test bags而非正式100 bags，统一复用D-XGB分数。
- 效用：Adult+GaussianCopula+单seed子集，不是两表三生成器五seed。
- 估值：Adult固定污染子集；KNN递推经过小N全子集核验，Data-OOB为最小bagging闭环。
