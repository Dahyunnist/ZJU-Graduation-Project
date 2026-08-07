# C2 正式启动加固与单 seed 试运行完成报告

> 完成日期：2026-07-15；阶段：C2 formal readiness + Adult seed=2026 pilot；结论：**工程加固和 GaussianCopula pilot 完成，CTGAN/TVAE 被 GPU 硬闸门阻塞；未进入 C4。**

## 1. 本轮结论

本轮完成了 pilot/formal/smoke 配置隔离、GPU与磁盘预检、run状态和完成标志、stdout/stderr、timing、artifact manifest、正式聚合隔离、质量硬闸门、SDV新加载接口兼容、manifest-first污染/bag接口及Adult GaussianCopula正式配置单seed pilot。

最终状态：

|生成器|run_id|状态|是否调用fit|
|---|---|---|---:|
|GaussianCopula|`c2-pilot-adult-gaussiancopula-s2026-20260714T181119905473Z`|`pilot_passed`|是|
|CTGAN|`c2-pilot-adult-ctgan-s2026-20260714T180335714131Z`|`blocked_by_gpu`|否|
|TVAE|`c2-pilot-adult-tvae-s2026-20260714T180335715148Z`|`blocked_by_gpu`|否|

三者均为 `run_type=pilot`，正式聚合结果为 `included=[]`。没有运行Credit、2027–2030、正式50/100 bags、C4检测器、比例估计或估值算法。

## 2. 回归与既有产物保护

开始时原有测试为 `59 passed in 3.88s`。最终为 **85 passed，0 failed，0 skipped，22.34 s**，JUnit 位于`reports/pytest_c2_formal_readiness_pilot.xml`。

实际复核12个关键哈希无差异，包括：Adult/Credit处理数据、Adult seed=2026 split、两个legacy脚本、旧baseline，以及三个C2和三个C3成功smoke summary。结果保存于`reports/c0_c3_hash_regression_c2_pilot.json`。旧smoke模型再次加载和采样成功，数据分区规模仍为：

- Adult：29,273 / 7,319 / 4,879 / 7,319；
- Credit：17,978 / 4,495 / 2,997 / 4,495。

## 3. 实际环境与GPU预检

|项目|实际值|
|---|---|
|生成器Python|3.11.15，`D:\Anaconda3\envs\tabpollution\python.exe`|
|SDV / SDMetrics|1.37.3 / 0.28.0|
|scikit-learn|1.9.0|
|torch|2.12.1+cpu|
|`torch.version.cuda` / cuDNN|null / null|
|`torch.cuda.is_available()`|false|
|GPU张量测试|失败：CUDA不可用|
|系统显卡|RTX 4070 Laptop GPU，8,188 MiB，总空闲7,926 MiB，驱动576.40|
|磁盘|总514,690,379,776 B；预检时空闲106,489,966,592 B|
|SDV `enable_gpu`参数|CTGAN/TVAE 1.37.3构造函数均支持|

结论：操作系统能看到GPU不等于当前PyTorch可用CUDA。CTGAN/TVAE runner在调用fit前阻断，并保存`fit_called=false`。未静默安装CUDA、PyTorch或升级任何依赖。

基于3000行/20 epochs CPU smoke线性外推，Adult全量300 epochs的粗略CPU时间为CTGAN约11,580秒（3.22小时）、TVAE约6,651秒（1.85小时）；该估计不能替代GPU实测。隔离环境建议、安装/验证命令、数GB磁盘提示和回滚方案已写入`reports/c2_pilot_generator_preflight.json`，本轮未执行。

## 4. GaussianCopula seed=2026 pilot

### 4.1 运行配置和效率

- dataset：Adult；split_seed=2026；generator_seed=2026；
- fit scope：完整`R_source_train`，29,273行；
- 参数：`enforce_min_max_values=true`、`enforce_rounding=true`；
- fit：10.2034秒；总流程：18.7854秒；
- 模型：218,595 B，SHA-256 `73c3f7bb78799f2a3eb2b671c576ea01b811724be99cfcfc50955b721e0dbc47`；
- 访问审计：只读取`R_source_train`，通过；
- 模型保存后通过SDV `utils.load_synthesizer`重新加载，并再次采样10行；
- 25项最终产物的路径、大小、行数和SHA-256验证通过。

### 4.2 四个用途池

|池|行数|sample_seed|采样时间|CSV SHA-256|
|---|---:|---:|---:|---|
|`S_detector_train`|8,051|2,127|0.3306 s|`9ab35ce1855e61f60b6621d5752c06ce49aaa0d1393e51610bfd84c36ee0135e`|
|`S_detector_val`|5,367|2,228|0.2145 s|`5c12c8f91d4e4c151b24cb01094c3c11a3b28422247625e059b814ee9b48352f`|
|`S_final_test`|8,051|2,329|0.3309 s|`5d386a352c4f0f0504b54ce0e9ec5f63b56e44086fb26e4ce7fdb5f7773a7fba`|
|`S_downstream_mix`|32,201|2,430|1.0460 s|`3b6350febbd48c98221cf94e152309c8b500053ef91645f60ab0f00d0eed3fee`|

四池分别调用sample，`synth_row_id`全局唯一，ID无交集；池内重复、与真实训练集精确重合和池间内容重合均为0。

### 4.3 质量硬闸门和诊断

硬闸门零失败：访问、schema、数值解析、目标合法性与完整类别、ID隔离、模型重载、TSTR和关键产物均通过。

|指标|结果|
|---|---:|
|SDMetrics overall|0.8020|
|Column Shapes|0.8441|
|Column Pair Trends|0.7598|
|TRTR AUROC|0.9081|
|TSTR AUROC|0.8579|
|TSTR–TRTR差距|−0.0502|
|真实目标`>50K`比例|0.2394|
|合成目标`>50K`比例|0.2415|

这些仍是单seed pilot诊断值，不是正式五种子结果。

## 5. 采样随机性缺陷及修复

第一次完整pilot曾被初始质量门判为通过，但验收时发现其四池内容哈希与seed=42 smoke逐池完全相同。根因是适配器只调用了全局NumPy/Torch seed，没有把`sample_seed`传入SDV合成器内部随机状态。

处理结果：

1. 原run `c2-pilot-adult-gaussiancopula-s2026-20260714T180355316988Z`改为`quality_blocked`，删除完成标志并保留全部产物；
2. 增加每次sample前调用SDV内部`_set_random_state(sample_seed)`的版本兼容封装；
3. 增加fake模型单元测试，确认不同sample seed真实进入SDV内部状态；
4. 增加跨run内容哈希硬闸门；
5. 新pilot与seed=42四池内容哈希映射不同，`cross_run_seed_validation.json`通过；
6. 一次在fit前错误设置内部状态的尝试因SDV底层模型尚未创建而失败，run `c2-pilot-adult-gaussiancopula-s2026-20260714T181051915952Z`及异常日志被保留，随后修正为仅在fit完成后的sample阶段设置。

该修复非常关键，否则后续五个生成种子会名义不同但实际复用相同采样序列。

## 6. 日志、状态、产物和聚合隔离

成功pilot包含`config_resolved.yaml`、`environment.json`、`access_audit.json`、`stdout.log`、`stderr.log`、`timing.json`、模型、metadata、provenance、四池manifest、quality、quality_gate、reload validation、run manifest、status、artifact manifest和`COMPLETE`。

失败和blocked run也保存日志、环境、timing、状态和artifact manifest，但没有`COMPLETE`。聚合器只接受：

`run_type=formal AND status=formal_passed AND COMPLETE exists`

实际聚合排除了两个blocked、一个quality_blocked、一个failed和一个pilot_passed，正式included为空。结果见`reports/formal_aggregation_pilot_exclusion.json`。

## 7. manifest-first C3验证

新增污染recipe和bag成员manifest接口，默认不物化完整特征CSV；只有显式`--materialize`才会生成完整表。已有C3 smoke CSV未删除、未迁移、未改写。

实际fixture：

- GaussianCopula、`synthetic_replace`、p=25%、N=29,273；
- 合成数7,318，实际比例0.249991；
- recipe成员顺序哈希`4125005c8ca0684c68a2b7531d1198282c78729db29f65674b8b700b779d6dc1`；
- 与旧C3 smoke相同条件的来源成员和顺序逐项一致；
- 旧完整CSV为8,806,956 B，新recipe为711 B，单项减少99.9919%；
- 50% test bag重建为1,000行、真实/合成各500，成员哈希`3fdf3497c3bd31cff5f07241f2e6e85ea76666333b7bcdd59245e5da1c5d477c`与旧smoke一致；
- bag manifest 702 B，1,000个成员ID文件103,928 B，不保存特征副本。

新增CLI覆盖`mixing build`、`bags build`、`bags rebuild`、`protocol validate`和`runs aggregate`。本轮没有生成正式50/100 bags。

## 8. 新增测试覆盖

在原59项基础上新增26项，覆盖：pilot配置和300 epochs保护、run_type聚合隔离、成功/失败日志和artifact manifest、GPU失败不调用fit、单类目标坍缩、非法目标、TSTR、磁盘预检、SDV新旧load接口、sample seed内部传递、跨run采样种子失效、manifest-first四条件、按需物化、旧smoke成员一致性、bag重建、blocked run以及真实Gaussian pilot产物。

最终结果：**85 passed in 22.34s**，无警告。

## 9. 主要新增/修改文件

- `configs/pilot_c2.yaml`；
- `src/tabpollution/generators/preflight.py`、`pilot.py`、`quality_gate.py`；
- `src/tabpollution/generators/sdv_adapter.py`随机状态与load兼容修复；
- `src/tabpollution/runs.py`；
- `src/tabpollution/mixing/manifest_first.py`、`commands.py`；
- `src/tabpollution/cli.py`；
- `tests/unit/test_formal_readiness.py`及生成器随机状态测试；
- `tests/integration/test_c2_c3_smoke_artifacts.py`新增真实pilot、blocked、hash和manifest-first验收；
- 三个最终状态run、两个保留的Gaussian诊断/失败run；
- GPU预检、哈希回归、聚合隔离、fixture、状态表、JUnit和本报告；
- `README.md`第三轮真实命令。

## 10. 与冻结规格的差异、风险和技术债

1. CTGAN/TVAE正式配置pilot未运行，原因是当前torch为CPU版；没有降低300 epochs或缩小fit scope冒充成功。
2. TVAE先前20-epoch smoke的单类目标坍缩尚未通过正式配置复核；GPU环境修复后仍必须先过质量硬闸门。
3. SDV内部随机状态方法以下划线开头，属于当前1.37.3可用的私有接口；已用适配层和测试隔离，升级SDV时必须重新核对。
4. SDV仍提示旧`SingleTableMetadata`弃用；模型加载已迁移到`utils.load_synthesizer`，metadata API迁移可在依赖升级专项中处理，不能在正式矩阵中途升级。
5. manifest-first fixture证明能显著节省污染CSV空间，但正式bags尚未生成；正式C3开始前应再估算成员manifest总量。
6. git元数据仍无效，所有manifest如实记录，不伪造commit。

## 11. 是否满足启动完整正式C2

**尚不满足。** GaussianCopula工程与单seed pilot已经达到要求，但完整“两表×三生成器×5 seeds”仍受以下条件阻塞：

1. 创建并验收隔离GPU环境，确认PyTorch CUDA张量测试通过；
2. 在该环境完成Adult seed=2026 CTGAN与TVAE完整29,273行、300 epochs pilot；
3. 两者均通过目标类别、TSTR、重载、四池隔离和跨run采样随机性硬闸门；
4. 特别确认TVAE不再发生目标坍缩；
5. 依据实际GPU pilot重新评估五seed训练时间和磁盘预算。

建议下一阶段只做“隔离GPU环境建立与CTGAN/TVAE Adult seed=2026 pilot”，通过后再单独批准两表五seed正式C2。本轮已按停止线停在C2，未进入C4。
