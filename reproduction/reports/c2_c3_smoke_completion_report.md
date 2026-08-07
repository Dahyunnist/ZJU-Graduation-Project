# C2–C3 工程与冒烟验收完成报告

> 完成日期：2026-07-15；总 run_id：`c2-c3-smoke-20260715`；阶段状态：**完成 C2–C3 smoke，停止在 C3，未启动 C4 或正式实验**。

## 1. 结论

本轮完成了统一生成器接口、三种 SDV 生成器适配、训练访问审计、四个隔离合成池、质量与 TSTR/TRTR smoke 检查、四种污染条件、7 档比例 bag、manifest 重建，以及 P1–P5 协议验证。GaussianCopula、CTGAN、TVAE 三个 Adult smoke 均成功；没有 blocked 项。

所有结果仅用于工程验收。`split_seed=2026` 只负责读取已有真实分区，`generator_seed=42` 只用于 smoke 随机性。没有运行正式 seeds 2026–2030 的生成器训练，没有运行 300 epochs，没有训练 Credit，没有下载 B-small，也没有实现或运行 C4 检测/比例估计/估值算法。

## 2. 实际环境

|用途|Python|关键包|计算设备|
|---|---|---|---|
|基础测试|3.13.11|pandas 2.3.3、scikit-learn 1.8.0、pytest 9.0.2|CPU；系统可见 RTX 4070 Laptop GPU|
|生成器|3.11.15|SDV 1.37.3、SDMetrics 0.28.0、pandas 2.3.3、scikit-learn 1.9.0、torch 2.12.1+cpu|`torch.cuda.is_available()=False`，CUDA/cuDNN/GPU 均不可用|

生成器使用 `D:\Anaconda3\envs\tabpollution\python.exe` 直接运行。由于该环境中的 torch 是 CPU 版本，CTGAN/TVAE 按指令允许的降级方案使用 `R_source_train` 内确定性抽取的 3,000 行 smoke 子集，而非假定系统显卡可用。环境详情分别见 `reports/environment_c2_c3_base.txt` 和各 C2 run 的 `environment.json`。

## 3. C2 三生成器 smoke

|生成器|成功 run_id|拟合范围|epochs|拟合时间|四池采样总时间|模型大小|
|---|---|---|---:|---:|---:|---:|
|GaussianCopula|`c2-smoke-adult-gaussiancopula-s42-a2`|完整 `R_source_train`，29,273 行|不适用|66.207 s|12.821 s|218,541 B|
|CTGAN|`c2-smoke-adult-ctgan-s42-a2`|确定性 smoke 子集，3,000 行|20|79.115 s|6.584 s|1,776,759 B|
|TVAE|`c2-smoke-adult-tvae-s42-a2`|确定性 smoke 子集，3,000 行|20|45.444 s|2.675 s|586,526 B|

最终 resolved 参数：GaussianCopula 显式启用 min/max 与 rounding；CTGAN 使用 `batch_size=500, pac=10, epochs=20, enable_gpu=false`；TVAE 使用 `batch_size=500, embedding_dim=128, compress_dims=[128,128], decompress_dims=[128,128], epochs=20, enable_gpu=false`。三个模型均已从 `model.pkl` 重新加载，并用 seed 9090 再采样 10 行；schema 与列顺序检查全部通过。

### 3.1 访问审计

|生成器|实际读取分区|拟合行数|row_id 清单 SHA-256|结论|
|---|---|---:|---|---|
|GaussianCopula|仅 `R_source_train`|29,273|`9f9d4ef70fec7ecac45309b08d3e8292b74a8db9204297ff7e1b3a485a0de997`|通过|
|CTGAN|仅 `R_source_train` 内 smoke 子集|3,000|`cce3971693f3adabb8e29d84a4255e40a1132a9de7c3ec5de01589d222c43e95`|通过|
|TVAE|仅 `R_source_train` 内 smoke 子集|3,000|同 CTGAN|通过|

没有生成器读取 `R_detector_train`、`R_detector_val` 或 `R_final_test`。送入 SDV 的数据已去除 `row_id`、`split` 和 provenance 字段，目标列 `income` 被保留。

### 3.2 四个隔离合成池

三个生成器使用相同的池规格，但分别独立调用 `sample`；没有先生成大表再切分。

|池|行数|sample_seed|
|---|---:|---:|
|`S_detector_train`|8,051|143|
|`S_detector_val`|5,367|244|
|`S_final_test`|8,051|345|
|`S_downstream_mix`|32,201|446|

每个池的文件 SHA-256、内容 SHA-256、schema、来源 run_id 均在相应 `pools_manifest.json` 中。三组四池的 `synth_row_id` 均全局唯一且池间无 ID 交集。GaussianCopula 与 CTGAN 的池间内容精确重合数均为 0；TVAE 因模型退化出现少量内容重合（各池对为 0–5 条），但 ID 仍隔离，且该现象已如实记录。

|生成器|池|CSV SHA-256|
|---|---|---|
|GaussianCopula|`S_detector_train`|`f056435adbba58a417264b2b41e6e34712a2190b9f991c0545d049a499be436d`|
|GaussianCopula|`S_detector_val`|`cad2a6a33c42ce7e26e39542cf3efd62285e01b3dbaec7c257305e94de2011f8`|
|GaussianCopula|`S_final_test`|`bb24f79a2eb3644634892a4a3980f08bea5b49092855a1ca740f8308480eb209`|
|GaussianCopula|`S_downstream_mix`|`d8ab646c8a779f2c232755ec472cd42f83dbd9ef9dfe955453e90ab017de032d`|
|CTGAN|`S_detector_train`|`942e496865a0578b4e4f9a8d56c8ec7fd2544687c725e224f7e40e6235905b4a`|
|CTGAN|`S_detector_val`|`6a8c05cf73a98e609f3db1fc5440bc84e1454525a5842bbb38f52ebedd5e4135`|
|CTGAN|`S_final_test`|`83a25b06a7326fdc20361b508306de845a39d129fb1f52dc6037438b2fe2f105`|
|CTGAN|`S_downstream_mix`|`121d186d7844e55e7144141d009737b4031b96ecbf713ddbbdbcdce35c9a0220`|
|TVAE|`S_detector_train`|`6ea65088f972c1b47bdd47d3cdf5a1077f3415baac3913fe85cf0ecd5f8af747`|
|TVAE|`S_detector_val`|`5bb0b0d31920e351aa7bfec816ddf596208bd8fe09c25017212041e8a0afe5cb`|
|TVAE|`S_final_test`|`3a0dc27f074d2ac1f50c184c4f6b5ee2f4fd00c370bce30bcffbd1a9dd935445`|
|TVAE|`S_downstream_mix`|`43ce214c55284f43488b037e137a57aff064aa3b943cc55d610c653ecc8f8800`|

四池采样耗时（train/val/final/downstream）分别为：GaussianCopula 2.055/1.466/2.095/7.205 s，CTGAN 1.096/0.893/1.121/3.473 s，TVAE 0.520/0.400/0.494/1.261 s。

### 3.3 质量、伪影与 TSTR/TRTR

|生成器|SDMetrics overall|Column Shapes|Column Pair Trends|TRTR AUROC|TSTR AUROC|
|---|---:|---:|---:|---:|---:|
|GaussianCopula|0.8038|0.8459|0.7617|0.9081|0.8540|
|CTGAN|0.7686|0.8436|0.6935|0.9081|0.3775|
|TVAE|0.4983|0.6187|0.3779|0.9081|未得到：合成训练集目标单一|

三者 schema、列顺序、数值解析和字符串外侧空格检查均通过，格式合法率为 1.0。GaussianCopula 与 CTGAN 的池内完全重复率、对真实训练集精确重合率均为 0。TVAE 的 `S_downstream_mix` 有 14 条重复、1 条与真实训练记录精确重合；`S_final_test` 有 3 条重复。TVAE 只生成 `<=50K`，因此固定 LR 无法执行 TSTR，状态被明确记录为 `failed_single_class_train`，没有伪造数值。

这些结果表明 20-epoch CPU smoke 中 CTGAN/TVAE 的生成质量不足，尤其 TVAE 已发生目标坍缩；它们不能作为正式质量结论，也不能被解释为检测器效果。正式运行前必须先解决 GPU/训练预算并设置质量门槛。

## 4. C3 污染构造

每个生成器均生成 `4 conditions × 7 rates = 28` 个污染产物，共 84 个。比例为 0、0.05、0.10、0.25、0.50、0.75、1.00；计数统一采用 `floor(p*N+0.5)`，Adult 基底 `N=29,273` 时追加/替换数依次为 0、1,464、2,927、7,318、14,637、21,955、29,273。

|条件|最终规模|含义|
|---|---:|---|
|`real_only`|29,273|纯真实基底；p 只作为请求档位记录|
|`real_append_bootstrap_control`|29,273 + 追加数|从 `R_source_train` 显式有放回 bootstrap 的真实追加对照|
|`synthetic_append`|29,273 + 追加数|从 `S_downstream_mix` 追加合成记录|
|`synthetic_replace`|固定 29,273|真实移除数与合成加入数相同|

0% replacement 与 real-only 成员一致，100% replacement 全部来自合成池。每条混合记录均保存 `mix_row_id`、来源 ID、generator、pool、p、mix_seed 和 condition；特征接口显式排除这些元数据。

`real_append` 没有独立的新真实样本源，本轮按冻结方案要求实现为 `R_source_train` bootstrap control。这仍是需要导师确认的实验设计点，不能表述成新增独立真实数据。

## 5. 比例 bags 与重建

每个生成器在每个比例上生成 2 个 calibration bag 和 3 个 test bag，即 35 个；三生成器合计 105 个 bag、105,000 条成员记录。每个 bag 固定 1,000 条，bag 内无放回、无重复，不同 bag 之间允许复用。

- calibration 只使用 `R_detector_val + S_detector_val`；
- test 只使用 `R_final_test + S_final_test`；
- 三个生成器的 calibration/test 真实 ID 与合成 ID 均严格不相交；
- 0%/100% 边界与全部中间比例均通过精确计数检查；
- 每个 bag 保存成员 ID、来源、请求/实际比例、mix_seed 和成员顺序哈希。

实际检查示例 `bag:adult:GaussianCopula:s42:test:p050:b00`：重建 1,000 行，其中真实 500、合成 500，成员 SHA-256 为 `3fdf3497c3bd31cff5f07241f2e6e85ea76666333b7bcdd59245e5da1c5d477c`，与 manifest 完全一致。

## 6. P1–P5 协议验证

P1（同表同生成器但记录/用途隔离）、P2（同表跨生成器）、P3（跨表）、P4（跨表跨生成器）、P5（跨域）正例均通过。针对表、生成器、record ID 泄漏的反例均在单元测试中稳定失败并返回冲突原因。P5 明确标记 `include_in_p3_macro=false`，不会混入 P3 宏平均。

P3–P5 本轮只验证 manifest 规则，没有运行 B-small 或真实跨域实验。

## 7. 测试与回归

最终执行结果为：**59 passed，0 failed，0 skipped，6.46 s**。JUnit 报告为 `reports/pytest_c2_c3_smoke.xml`。

新增测试覆盖生成器访问审计、特征去 provenance、registry、seed 派生、四池 ID、save/load、重合统计、half-up 舍入、四种污染条件、0%/100% 边界、可重复构造、bag 隔离/比例/哈希、P1–P5 正反例，以及三个已保存 C2/C3 端到端 smoke 产物。昂贵生成器不会在每次 pytest 中重训。

C0–C1 与 legacy 回归哈希全部一致：Adult/Credit 处理文件、两个旧脚本、两个旧 baseline summary、week0 与 week0_smoke 报告均未变化。Adult/Credit 的五个正式 split 再验证通过，规模仍分别为 29,273/7,319/4,879/7,319 与 17,978/4,495/2,997/4,495。

## 8. 首次失败记录与修复

失败记录未删除：

1. GaussianCopula 首次在 TSTR 预处理时遇到 pandas `pd.NA` 与 sklearn imputer 兼容问题；改为明确转换 `numpy.nan` 后成功。
2. CTGAN 首次同时传入 SDV 1.37.3 的旧 `cuda` 和新 `enable_gpu` 参数；核对本机签名后只保留 `enable_gpu=false`。
3. TVAE 首次因目标列只含一个类别导致 LR 训练异常；质量检查改为如实返回 `failed_single_class_train`，不让诊断性失败破坏全部产物。

对应首次 run 目录保留 `failure.json`；成功重跑使用 `-a2`，聚合默认只读取含成功 summary 的 run。

## 9. 新增/修改文件概览

- `src/tabpollution/generators/`：统一接口、SDV 适配、四池、质量检查、Adult smoke；
- `src/tabpollution/mixing/`：污染构造、bag、协议、C3 smoke；
- `configs/smoke_c2_c3.yaml`：独立 smoke override，未修改正式参数；
- `src/tabpollution/cli.py`：generator smoke/validate、mixing smoke、bags inspect；
- `tests/unit/test_generators.py`、`test_mixing.py`、`test_protocols.py`；
- `tests/integration/test_c2_c3_smoke_artifacts.py`；
- 三个成功 C2 run、三个成功 C3 run 及失败尝试记录；
- `runs/c2-c3-smoke-20260715/run_manifest.json`；
- `reports/environment_c2_c3_base.txt`、`pytest_c2_c3_smoke.xml` 与本报告；
- `README.md` 的 C2–C3 实际命令。

## 10. 与冻结规格的差异、风险和技术债

1. CUDA 实测不可用，因此 CTGAN/TVAE 使用获准的 3,000 行 CPU smoke 子集；这不是正式拟合范围。
2. C3 CLI 采用 `mixing smoke` 复合命令一次构造 contamination、bags 和 protocol validation，而不是拆成三个独立 build 命令；底层模块已经分离，可单独复用。
3. SDV 1.37.3 的类级 `load` 发出未来弃用警告，但当前 reload 成功；后续应迁移到官方建议的 `utils.load_synthesizer`。
4. 实际训练 stdout/stderr 在桌面任务日志中可见，但成功 run 没有单独物化 `stdout.log`/`stderr.log`；关键环境、参数、耗时、失败异常和产物均已结构化保存。正式运行器应补日志重定向。
5. 当前 workspace 的 git 元数据无效，manifest 如实记录 `git_available=false`，没有伪造 commit。
6. 本轮 C2/C3 产物约 897.5 MB，其中 C3 物化的 84 个完整污染 CSV 约占 838.0 MB；正式多 seed 运行前应改用成员 manifest/按需重建或评估 Parquet，避免存储线性膨胀。

## 11. 是否可启动正式 C2

**工程入口已达到启动条件，但计算与质量条件尚未完全达到，因此不建议立即批量启动“两表 × 三生成器 × 五 seeds”。**

在正式启动前至少应完成：确认可用 GPU 环境及 CTGAN/TVAE 正式训练预算；用一个正式候选 seed 预跑质量门槛，确保目标不坍缩；让导师确认 `real_append_bootstrap_control` 的设计；补独立运行日志；估算两表五 seed 的磁盘和时间成本。完成这些后，当前访问审计、四池隔离、manifest、污染/bag 构造与回归测试可直接作为正式运行基础。

下一阶段建议先提交本报告供导师确认，再单独编写并审批正式 C2 执行指令。本轮到此停止，未自行进入 C4。
