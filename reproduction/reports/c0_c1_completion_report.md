# C0—C1 阶段完成报告

> 完成日期：2026-07-15；run_id：`c0-c1-20260715`；状态：**通过，已停止在C1，未执行C2**。

## 1. 完成内容

1. 完成原 `reproduction`、历史脚本、输出和环境审计，原文件哈希复核一致；
2. 建立 `tabpollution` 模块化工程、`pyproject.toml`、CLI和测试目录；
3. 冻结并校验 benchmark v1 配置：两表、三生成器名称、5个正式种子、四分区、7档污染率和比例估计bag参数；
4. 从UCI官方地址取得Adult和Default of Credit Card Clients原始压缩包，记录来源、许可、日期和SHA-256；
5. 完成两表schema校验、类型规范化、完全重复行审计/移除和内容寻址的稳定`row_id`；
6. 对每张表、每个正式seed生成60%/15%/10%/15%分层冻结划分；
7. 生成机器可读/人工可读数据卡、数据registry、resolved config和split manifest；
8. 建立run manifest JSON Schema、Python校验和run_id防覆盖机制；
9. 运行旧基线非破坏性复核、两次数据重建、CLI校验和完整自动化测试。

本轮没有训练GaussianCopula、CTGAN、TVAE、TabDDPM、TabSyn、检测器或Transformer，也没有产生正式模型指标。

## 2. 数据来源、规模与校验值

|数据集|官方来源|许可|原始文件|原始行×特征|去重后行×特征|目标列|
|---|---|---|---|---:|---:|---|
|Adult|UCI ID 2；DOI `10.24432/C5XW20`|CC BY 4.0|620,237 bytes；SHA-256 `7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb`|48,842×14|48,790×14|`income`|
|Default of Credit Card Clients|UCI ID 350；DOI `10.24432/C55S3H`|CC BY 4.0|5,539,494 bytes；SHA-256 `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`|30,000×23|29,965×23|`default_payment_next_month`|

处理文件SHA-256：

- Adult：`9cf6e79f08b62089624828b3b3c60c64950fea93948f5a5b2829d9194724f2d2`；
- Credit：`ee2fa731865931ec6c4c54d7c16e00f1051b9163a0bfeb7aa318e795302a4d29`。

完全重复记录按冻结规格在划分前移除：Adult 52行，Credit 35行。`row_id`由规范化后的特征和目标内容SHA-256构造，不使用DataFrame临时索引；`row_id`、split和provenance均被特征接口排除。

## 3. 冻结分区结果

每个数据集的2026、2027、2028、2029、2030五个seed具有相同分区规模，但记录成员随seed改变。

|数据集|R_source_train|R_detector_train|R_detector_val|R_final_test|总数|
|---|---:|---:|---:|---:|---:|
|Adult|29,273|7,319|4,879|7,319|48,790|
|Credit|17,978|4,495|2,997|4,495|29,965|

逐seed的row_id映射、目标分布、内容哈希和文件哈希位于`data/splits/benchmark_v1/<dataset>/seed_<seed>.{csv,json}`。所有分区两两不相交，并集覆盖清洗后的全部记录。

## 4. 可重复性检查

在不删除第一次产物的情况下再次运行完整prepare流程，比较结果：

- `prepare_summary.json`第一次与第二次SHA-256均为 `228EA6DD6BB34535897518DB46DB2DAEF59B9363B38146592DB01954A93FD2ED`；
- Adult seed=2026 split CSV第一次与第二次SHA-256均为 `47D75A9FC88B09901A020F5EB57F62FADD6C55921AAF63E5FCDCB8A2B904EFF1`；
- 其余每个数据集×seed的assignment内容哈希由集成测试与对应JSON逐一核对；
- CLI validate再次验证处理文件校验值、row_id唯一性、四分区覆盖和全部5个seed。

结论：在相同原始数据、配置和代码下，可重建相同的处理数据与冻结划分。

## 5. 测试结果

最终执行：`22 passed in 2.12s`，失败0，跳过0。JUnit报告为`reports/pytest_c0_c1.xml`。

覆盖内容包括：

- 合法配置、非法比例、重复seed、未知数据集和污染率越界；
- 同数据重复读取产生相同row_id；
- 完全重复记录的row_id唯一且确定；
- 同seed分区哈希相同；
- 分区两两无交集、并集全覆盖、未知row_id失败；
- 60/15/10/15确定性舍入和目标分层容差；
- target、row_id、split和provenance不会进入特征矩阵；
- 处理数据重新加载时，Credit等数值编码的类别字段仍按registry恢复为字符串类别，而不会被pandas误推断为连续数值；
- 未知schema列显式失败；
- run manifest缺字段失败、重复run_id拒绝覆盖；
- Adult、Credit分别完成真实数据prepare→split→card集成检查；
- 保存的全部split哈希与manifest一致。

## 6. 实际运行的主要命令

```powershell
# C1所需轻量依赖放在项目目录，避免修改受限的全局Python
python -m pip install --target .deps xlrd==2.0.1
$env:PYTHONPATH=((Resolve-Path 'src').Path + ';' + (Resolve-Path '.deps').Path)

python -m tabpollution.cli environment capture --output reports/environment_initial.txt
python -m tabpollution.cli data prepare --config configs/benchmark_v1.yaml
python -m tabpollution.cli data validate --benchmark benchmark_v1
python -m pytest -q --junitxml=reports/pytest_c0_c1.xml

# 旧脚本只复用已有CSV，不重新训练CTGAN，也不写入原outputs
& 'D:\Anaconda3\envs\tabpollution\python.exe' run_adult_baseline.py `
  --real-csv outputs/week0/adult_real_sample.csv `
  --synthetic-csv outputs/week0/adult_synthetic_ctgan.csv `
  --output-dir legacy/validation_20260715_direct
```

## 7. 主要新增/修改文件

|位置|用途|
|---|---|
|`configs/benchmark_v1.yaml`|冻结基准总配置|
|`configs/datasets/*.yaml`|Adult/Credit registry及B-small待准备登记|
|`configs/generators/*.yaml`|只注册三种Track A生成器，本轮未训练|
|`src/tabpollution/config.py`|YAML加载与严格配置校验|
|`src/tabpollution/data/`|官方数据加载、schema、清洗、row_id、划分和数据卡|
|`src/tabpollution/manifests/`|run manifest校验与防覆盖|
|`src/tabpollution/pipeline.py`、`cli.py`|prepare/validate命令|
|`tests/unit`、`tests/integration`|18项单元测试、4项真实数据集成测试|
|`manifests/benchmark_v1`|resolved config、数据registry和prepare摘要|
|`data/raw`、`data/processed`、`data/splits`|原始文件、规范化表和冻结row_id映射|
|`reports/data_cards`|两张Markdown数据卡与JSON数据卡|
|`reports/current_state_audit.md`|环境与legacy审计|
|`runs/c0-c1-20260715/run_manifest.json`|本轮可追踪运行记录|

原 `run_minimal_loop.py`、`run_adult_baseline.py` 和原 `outputs` 哈希均未变化。

## 8. 与方案的差异和处理

1. 旧《任务定义》中的70/15/15已被最新冻结规格的60/15/10/15取代，本实现采用后者；
2. UCI原始表存在完全重复记录，去重后行数低于论文/官网原始行数，数据卡同时保留原始与清洗后数值；
3. 当前系统Python的全局/用户包目录不可写，因此`xlrd 2.0.1`安装到项目`.deps`；不影响数据和划分结果；
4. 工作区`.git`元数据无效，无法提供commit哈希，manifest已明确标记而非伪造；
5. B-small六表本轮只完成注册，尚未下载；符合C1边界，Track B准备应在后续独立验收；
6. 旧脚本通过Conda包装器运行时出现GBK进度输出错误，直接调用同一环境Python验证成功。

## 9. 风险与技术债

- C2生成器应使用Python 3.11的`tabpollution`环境；该环境已有SDV/SDMetrics，但没有pytest，后续应采用项目本地测试环境或补齐轻量测试依赖；
- 当前开发环境PyTorch为CPU版，即使系统可见RTX 4070也不能直接用于深度模型；C2的GaussianCopula不受影响，CTGAN/TVAE需先确认SDV实际CUDA后端和确定性；
- Credit的`SEX/EDUCATION/MARRIAGE/PAY_*`按语义固定为类别/序数代码，后续所有生成器和下游模型必须沿用registry，不能自动把它们全部当连续变量；
- split生成依赖scikit-learn 1.8.0的确定性实现；环境升级时必须先比较现有split哈希，不得静默重建；
- 原始数据许可为CC BY 4.0，论文和发布材料中需要保留UCI引用与署名。

## 10. 是否满足进入C2的条件

**满足。** 两表的来源、schema、稳定row_id、5-seed冻结分区、数据卡、配置和manifest均已验证，后续生成器可以被强制限制为只读取`R_source_train`。

下一阶段建议先执行C2的小范围入口：定义统一generator接口与访问审计，先用Adult、seed=42运行GaussianCopula/CTGAN/TVAE smoke，并证明生成器从未读取其他三个真实分区；通过后再启动正式5-seed训练。本报告到此停止，没有自行执行C2。
