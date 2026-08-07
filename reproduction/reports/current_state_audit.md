# C0 当前状态与 legacy 审计

> 审计日期：2026-07-15。本报告只登记并验证历史状态，不把历史结果升级为 benchmark v1 正式结果。

## 1. 工作区状态

- 工作区根目录存在 `.git` 目录，但 `git status` 和 `git log` 均返回“not a git repository”；因此本轮无法记录有效 commit，后续 run manifest 将 `git` 标记为 `unavailable_invalid_repository_metadata`。
- 未发现 `AGENTS.md`；没有额外的仓库级执行约束。
- 原 `reproduction` 目录包含两个脚本、三份 README、两个依赖清单和四组历史输出目录。
- 本轮在原目录中渐进新增模块化基准，不移动、不删除原文件。

## 2. 历史脚本只读清单

|文件|字节|SHA-256|状态|
|---|---:|---|---|
|`run_minimal_loop.py`|7,586|`495D59D81C40202A762DEC2D001CDC6397E19E2DACDEAB4CFC43D006A0974B6F`|legacy smoke，只读保留|
|`run_adult_baseline.py`|6,729|`F3D08BAE0393F48AE8F9A460B763B34B939234BA6AB6A5650B027EF381196349`|legacy smoke，只读保留|
|`outputs/adult_baseline/baseline_summary.json`|509|`BF7B05721EF78424FE05EC4D279C9830162A32F9CF1900C9A8D94AA1B76422EF`|历史结果，只读保留|
|`outputs/adult_baseline_reuse/baseline_summary.json`|509|`ABF98860C93608DB72E8B9CDD1FA8CD072ECB20CA5A67B76B0E0AF93D26B594F`|历史结果，只读保留|
|`outputs/week0/detector_report.txt`|371|`09AEA371B4AF89836C531ECD10D4388ECB68F2CE55BA2081433E511915E56BD8`|历史结果，只读保留|
|`outputs/week0_smoke/detector_report.txt`|371|`3224C765373ACE07F19C4E8857CF3C25237B1525A6FC90C61729D2799F8DED21`|历史结果，只读保留|

历史输出目录完整保留：`outputs/week0`、`outputs/week0_smoke`、`outputs/adult_baseline`、`outputs/adult_baseline_reuse`。

## 3. 历史结果与本轮验证

历史完整 Adult 基线记录：

- 3,000条真实记录拟合 CTGAN，生成1,000条合成记录；
- CTGAN 20 epochs，`seed=42`；
- SDMetrics overall `0.7733985767`；
- Logistic Regression AUROC `0.780752`。

复用 week0 数据的历史记录：

- SDMetrics overall `0.7732779684`；
- Logistic Regression AUROC `0.848992`。

本轮没有重新训练 CTGAN。使用 `tabpollution` Python 3.11 环境和已有 week0 CSV，在新目录 `legacy/validation_20260715_direct` 重新运行旧评估脚本，得到 SDMetrics `0.7732779684`、AUROC `0.848992`，与历史 reuse 结果一致。`conda run` 包装器因 Windows GBK 无法显示进度字符而报编码错误，但脚本实际已完成；随后直接调用该环境的 Python 再次验证成功。

## 4. 为什么不能进入正式结果表

1. 生成器先在抽取的真实数据上拟合，之后才构造检测训练/测试集，没有冻结 `R_source_train`、`R_detector_train`、`R_detector_val`、`R_final_test`；
2. 只有 `seed=42`，没有5个正式随机种子和方差；
3. CTGAN 只训练20 epochs，按冻结规格只能算 smoke；
4. 只有Adult、CTGAN和LR，不能支撑两表×三生成器的主结论；
5. 没有固定的逐记录 manifest、概率校准、阈值验证和最终测试隔离；
6. 0.780752与0.848992来自不同真实/合成文件组合，不能当成同一协议的重复实验。

## 5. 可复用与不可复用边界

- 可复用：旧脚本的数据加载思路、SDMetrics调用方式、LR预处理管线和结果字段命名；
- 不直接复用：旧的数据划分、seed、输出目录和实验数值；
- benchmark v1 后续算法必须读取新 `data/splits/benchmark_v1` 中的冻结 row_id 映射；
- 原 `outputs` 永久与新 `runs`、`reports` 分开。

## 6. 环境结论

- C0—C1 实际开发/测试环境：Python 3.13.11，pandas 2.3.3，scikit-learn 1.8.0，PyYAML 6.0.3，pytest 9.0.2；XLS读取依赖 `xlrd 2.0.1` 安装于项目本地 `.deps`。
- 历史生成实验环境：Python 3.11.15，SDV 1.37.3，SDMetrics 0.28.0，scikit-learn 1.9.0。
- 硬件可见：32逻辑CPU；NVIDIA GeForce RTX 4070 Laptop GPU，8,188 MiB，驱动576.40。当前开发环境的 PyTorch 为CPU版；本轮没有进行GPU训练。
- 详细信息见 `reports/environment_initial.txt` 与 `reports/environment_legacy_tabpollution.txt`。

## 7. 规格冲突处理

较早的《阶段二_任务定义与评测规范》写有70%/15%/15%的三分区方案；2026-07-14冻结的《第3步_固定基准规格_v1》将其替换为60%/15%/10%/15%四分区。按文档优先级，本轮实现后者，旧方案只作为历史记录。

