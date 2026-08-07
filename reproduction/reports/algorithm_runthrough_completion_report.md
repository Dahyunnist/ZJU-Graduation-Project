# 算法跑通交差版完成报告

> 完成时间：2026-07-14T19:02:14.444871+00:00；成功run累计耗时：114.5s；状态：通过。

## 结论

必跑矩阵共19项，已通过19项，未通过0项：无。所有结果均标记为runthrough/smoke，未进入formal汇总。

## 实际范围

- 经典检测：D-LR、D-XGB；直接论文基线：D-3G、D-FT、D-TT；D3核心：D-DW、D-DWTA。
- 比例估计：CC、PCC、ACC、PACC、EMQ、HDy、DyS、Median Sweep，共用同一D-XGB分数和7档比例。
- 估值：四条件×7比例的LR/XGBoost效用曲线、KNN-Shapley、Data-OOB。
- 跨表mini数据：Adult、Credit、UCI Abalone；新增合成数据仅使用GaussianCopula。

## 代码来源

D1–D3、KNN-Shapley和Data-OOB均标记为论文对齐自实现；QuaPy 0.2.0为官方BSD-3-Clause包；XGBoost/PyTorch使用官方发行包。详见`code_availability_audit.md`。

## 工程与保护

C0–C3关键文件哈希是否不变：True。本轮没有启动正式五种子、14表或完整两表三生成器矩阵。失败尝试保留在runs目录并由status.json标识，不进入成功汇总。

## 必须保留的诊断警告

仅使用缺失数、空字符串数和总序列化长度的格式诊断器AUROC为 **0.6861**，超过0.65警戒线。因此D-XGB在Adult GaussianCopula P1上的极高AUROC不能直接解释为可泛化的语义检测能力；正式阶段必须统一格式，并通过跨生成器、跨表实验复核。详见`algorithm_runthrough/format_artifact_diagnostic.json`。

## 后续正式阶段仍需解决

GPU环境、CTGAN/TVAE正式质量门、两表三生成器五seed、完整跨表表集、正式bag数、论文超参数及置信区间。当前产物足以证明算法代码已经跑通，并支撑回到文献综述和研究方案写作。
