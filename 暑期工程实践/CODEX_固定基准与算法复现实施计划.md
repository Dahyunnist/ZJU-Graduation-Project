# Codex实施计划：固定基准与算法复现

> 目标：把当前单点冒烟脚本升级为可复现基准，依次跑通经典、直接基线、比例估计、SOTA和估值分析。Codex执行时必须以本文件和两个规格文件为准，不得自行扩大范围或修改冻结口径。

## 0. Codex开始前必须阅读

按以下顺序阅读：

1. `暑期工程实践/阶段二_任务定义与评测规范.md`；
2. `暑期工程实践/第3步_固定基准规格_v1.md`；
3. `暑期工程实践/第5步_算法复现与公平比较方案_v1.md`；
4. `暑期工程实践/阶段二_直接相关文献与算法复现清单.md`；
5. D1、D2、D3三篇中文梳理和原论文；
6. `reproduction/README*.md`与现有两个Python脚本。

若文件之间发生冲突，优先级为：第3步冻结基准 > 第5步算法方案 > 任务定义 > 旧README/旧脚本。

## 1. 总体开发原则

- 保留用户现有代码与结果；不覆盖`outputs/`历史目录。
- 先写manifest、配置和测试，再跑昂贵训练。
- 所有新功能通过模块化接口实现，不继续把逻辑堆进单个脚本。
- 每完成一个阶段都运行测试、更新状态表和README。
- 任何外部仓库都固定commit，不直接依赖会变化的main分支。
- 不在代码中硬编码本机绝对路径、API密钥或数据下载凭据。
- 无法下载或无许可的数据集先记录为阻塞，不用来源不明镜像替代。
- 正式结果必须由聚合程序从`metrics.json`读取，禁止手工抄数。
- 不把smoke、debug、failed结果混入正式汇总。

## 2. 建议代码结构

Codex在`E:\毕设\reproduction`内渐进建立：

```text
configs/
  benchmark_v1.yaml
  datasets/{adult,credit,...}.yaml
  generators/{gaussian_copula,ctgan,tvae,tabddpm,tabsyn}.yaml
  detectors/{c2st_lr,c2st_xgb,char3gram_lr,flat_text,table_transformer,datum_wise,datum_wise_ta}.yaml
  quantifiers/{cc,pcc,acc,pacc,emq,hdy,dys,median_sweep,kdey,continuous_sweep}.yaml
  valuation/{knn_shapley,data_oob,tmc_shapley,data_banzhaf}.yaml
src/tabpollution/
  cli.py
  config.py
  data/{registry,loaders,cleaning,splits,cards}.py
  generators/{base,sdv_generators,tabddpm,tabsyn}.py
  mixing/{pools,bags,contamination}.py
  detectors/{base,c2st,char3gram,flat_text,table_transformer,datum_wise}.py
  quantification/{base,baselines,quapy_adapter,continuous_sweep}.py
  valuation/{base,utility_curves,knn_shapley,data_oob,tmc,banzhaf}.py
  evaluation/{metrics,calibration,timing,aggregate,plots}.py
  manifests/{schema,validate}.py
tests/
  unit/
  integration/
scripts/
  smoke.ps1
  benchmark_core.ps1
  benchmark_cross_table.ps1
manifests/
runs/
reports/
legacy/
```

如果现有目录存在同名用户文件，Codex应适配而不是覆盖。

## 3. 阶段C0：保护现状与环境审计

### 任务

1. 检查工作区状态和用户未提交修改；
2. 为现有脚本与输出建立只读清单；
3. 运行现有最小测试，确认环境仍能工作；
4. 记录Python、CUDA、GPU、CPU、内存、SDV和scikit-learn版本；
5. 将旧脚本在README中标为legacy smoke，不改变其历史结果；
6. 建立新包骨架、测试框架和`pyproject.toml`/锁定依赖方案。

### 验收

- `reports/current_state_audit.md`；
- `reports/environment_initial.txt`；
- 旧脚本仍能运行；
- 新包可导入，空测试通过。

### 禁止

- 不删除`outputs`；
- 不以格式化全仓库为由改写无关文件；
- 不立即下载14表或训练Transformer。

## 4. 阶段C1：配置、数据注册与manifest

### 任务

1. 实现YAML配置加载、schema校验与resolved config保存；
2. 建立Adult和Credit数据注册器；
3. 下载数据时保存原始文件、URL、许可、日期和SHA-256；
4. 统一缺失值、dtype、目标列和稳定`row_id`；
5. 按60/15/10/15生成冻结划分；
6. 输出数据卡Markdown与JSON；
7. 为B-small六表建立注册项，先不要求全部下载成功。

### 必须测试

- 相同配置和种子产生相同划分哈希；
- 四个真实分区交集为空、并集等于清洗数据；
- 目标类别比例偏差在允许范围；
- 重复行和`row_id`处理稳定；
- 未知列或schema变化会显式报错。

### 验收

```powershell
python -m tabpollution.cli data prepare --config configs/benchmark_v1.yaml
python -m tabpollution.cli data validate --benchmark benchmark_v1
```

生成两张Track A数据卡和冻结split manifest。

## 5. 阶段C2：生成器与合成池

### 任务

1. 定义统一generator接口；
2. 接入GaussianCopula、CTGAN、TVAE；
3. 每个生成器只读取`R_source_train`；
4. 生成四个用途隔离的合成池；
5. 保存模型、配置、训练时间、采样时间和provenance；
6. 实现SDMetrics、重复率、真实重合率、无效率和TSTR；
7. 先在Adult `seed=42`、20 epochs冒烟，再跑正式300 epochs和5种子。

### 必须测试

- 生成器从未读取禁用分区；
- 合成池ID互不重叠；
- 输出schema、列顺序和dtype可被统一加载；
- 真实/合成格式元数据一致；
- 完全相同输入配置可恢复模型并重新采样。

### 验收

三生成器均能在Adult产生合成池和数据卡；任何失败均进入状态表。

## 6. 阶段C3：污染构造与比例bags

### 任务

1. 实现`real_only/real_append/synthetic_append/synthetic_replace`；
2. 实现7档污染率；
3. 构造1000条/袋的校准和测试bags；
4. 保存bag manifest与来源真值；
5. 实现P1–P5协议验证器。

### 必须测试

- 每个替换式混合集合总行数固定；
- 实际污染数等于约定四舍五入规则；
- 0%和100%边界正确；
- calibration/test记录池隔离；
- P3表不重叠、P4表与生成器均不重叠；
- 给验证器注入泄漏后测试必须失败。

### 验收

能够只依赖manifest重建任意bag，并打印其真实污染率和来源组成。

## 7. 阶段C4：正式经典检测基线

### 任务顺序

1. 重构C2ST-LR；
2. 新增C2ST-XGBoost；
3. 新增Character 3-gram + LR；
4. 统一概率校准、阈值、指标、预测保存和效率统计；
5. 运行标签置换、格式伪影和数据泄漏sanity checks；
6. 跑Adult正式5种子；
7. 跑Track A两表×三生成器P1/P2；
8. 跑B-small的3-gram LR P3/P4。

### 首批应形成的结果

- `reports/detection_track_a.csv`；
- `reports/detection_bsmall.csv`；
- 每个run的逐样本预测；
- 同表、跨生成器、跨表、完全迁移四张基础结果表；
- 论文D1报告值与本地缩小版结果的差异说明。

### 阶段闸门

如果3-gram LR无法在B-small稳定运行，先修数据表示和GroupSplit，不进入Transformer阶段。

## 8. 阶段C5：比例估计

### 任务顺序

1. 手写CC/PCC/ACC/PACC并用玩具数据测试；
2. 固定QuaPy版本，接入EMQ、HDy、DyS、Median Sweep；
3. 所有方法先使用同一组C2ST-XGBoost分数；
4. 在Adult 7档污染率上运行；
5. 扩到两表×三生成器×5种子；
6. 再使用3-gram LR分数重复核心比较；
7. 加入KDEy；Continuous Sweep最后接入。

### 必须测试

- 预测范围、越界率和clip逻辑；
- 完美分类器时各算法应接近真实比例；
- 随机分类器时校正算法的失败/不稳定能够被记录；
- 反转标签会被测试捕获；
- calibration和evaluation bags严格隔离。

### 阶段闸门

必须先形成CC/PCC/ACC/PACC/EMQ/HDy/DyS/Median Sweep的同分数比较表，再投入Continuous Sweep。

## 9. 阶段C6：Transformer直接基线

### 任务顺序

1. 实现字符级tokenizer与Flat-text Transformer；
2. 实现Table/column-wise Transformer；
3. 在B-small冒烟并保存训练曲线；
4. 与3-gram LR使用完全相同的fold比较；
5. 记录最大序列长度、截断率、参数量、训练时间和显存。

### 阶段闸门

Flat-text和Table Transformer至少各在一个B-small fold完成训练和推理，且逐样本预测可读取，才进入Datum-wise。

## 10. 阶段C7：Datum-wise Transformer与table adaptation

### 任务顺序

1. 用单元测试实现datum序列化与CLS-Datum池化；
2. 实现无跨列位置编码的row transformer；
3. 通过列随机置换不变性测试；
4. 在单表Adult过拟合一个小batch，证明梯度和损失正确；
5. 跑B-small P3；
6. 实现gradient reversal与table分类头；
7. 实现cosine适配权重；
8. 比较D-DW和D-DWTA；
9. 跑P1–P4；
10. 资源允许时扩展B-full十四表和四生成器。

### 必须输出

- 模型结构和参数量；
- 列置换前后预测差异；
- 检测loss、table classification loss和适配权重曲线；
- D-DW/D-DWTA消融；
- 自实现细节与论文未说明细节清单；
- 官方代码如发布，单独建立官方实现运行，不覆盖自实现结果。

### 阶段闸门

不以“平均AUROC达到论文0.69”作为唯一通过条件；结构、协议、预测和差异解释完整即可认定最小复现。若结果异常高，优先排查表名、格式或生成器泄漏。

## 11. 阶段C8：下游效用与估值

### 任务顺序

1. 在Adult/Credit上跑LR和XGBoost的7档污染率效用曲线；
2. 同时跑真实追加、合成追加、合成替换三个对照；
3. 实现KNN-Shapley并用`N≤12`全子集暴力核验；
4. 在正式污染集上计算KNN-Shapley；
5. 实现Data-OOB并报告覆盖次数；
6. 输出价值分布、负价值比例、删除曲线和跨种子稳定性；
7. 有余力再做TMC-Shapley/Data Banzhaf；
8. Bike Sharing回归效用曲线只作为扩展，不阻塞核心验收。

### 阶段闸门

下游测试集始终为纯真实`R_final_test`；一旦发现调参使用测试集，相关run全部作废并重跑。

## 12. 阶段C9：聚合、图表与复现报告

### 自动生成的主表

1. 数据集与生成器情况表；
2. 样本检测P1–P4结果表；
3. 比例估计总体及5%/10%结果表；
4. 污染率—下游效用表；
5. 数据价值分布与删除结果表；
6. 效率表；
7. 论文值—本地值—实现差异表；
8. 算法复现状态表。

### 自动生成的主图

- 污染率—比例估计误差曲线；
- 污染率—下游效用曲线；
- 真实/合成数据价值分布；
- 跨协议AUROC对比；
- 生成质量与检测难度散点图；
- Datum-wise适配消融图。

### 验收

```powershell
python -m tabpollution.cli report build --benchmark benchmark_v1 --runs runs --output reports/benchmark_v1
```

报告中的每个数可以追溯到run_id和预测文件。

## 13. 建议执行节奏

|阶段|建议时间|主要交付|
|---|---:|---|
|C0–C1|1–2天|工程骨架、环境审计、两表数据卡和冻结划分|
|C2–C3|2–4天|三生成器、合成池、污染集和bags|
|C4|2–3天|LR、XGBoost、3-gram LR正式结果|
|C5|2–3天|八个经典量化器和低污染曲线|
|C6|2–4天|两个Transformer直接基线|
|C7|4–7天|Datum-wise、table adaptation、P1–P4|
|C8|3–5天|效用曲线、KNN-Shapley、Data-OOB|
|C9|1–2天|自动汇总、图表和复现报告|

这是工程估计，不是硬性截止；每阶段只在验收通过后继续。

## 14. 算力不足时的保底顺序

保底必须保留：

1. Track A两表×三生成器；
2. C2ST-LR/XGBoost；
3. B-small的3-gram LR；
4. 八个经典比例估计器；
5. Datum-wise核心版至少一个跨表fold；
6. LR/XGBoost效用曲线；
7. KNN-Shapley和Data-OOB至少在Adult完成。

可以后移：B-full四生成器、Table adaptation完整大实验、Continuous Sweep、Data Banzhaf、LLM检测、回归扩展。

## 15. Codex每次完成阶段后的汇报模板

```text
阶段：C?
完成内容：
新增/修改文件：
运行命令：
通过的测试：
产生的run_id与结果：
与论文/预期趋势是否一致：
发现的问题和风险：
下一阶段前是否需要用户决定：
```

## 16. 可直接交给Codex的首条指令

完整首轮指令见：`暑期工程实践/给Codex的首轮执行指令_C0-C1.md`。建议直接复制该文件中的完整代码块，以便同时约束范围、验收测试和汇报格式。

```text
请先阅读暑期工程实践/CODEX_固定基准与算法复现实施计划.md及其引用的第3步、第5步规格文件。现在只执行阶段C0和C1：审计并保护现有reproduction目录，建立模块化工程骨架、配置校验、Adult/Credit数据注册、稳定row_id、冻结60/15/10/15划分、数据卡和相关单元测试。不要开始训练新生成器或Transformer，不要覆盖历史outputs。完成后运行测试并按计划中的阶段汇报模板报告。
```

## 17. C0—C1通过后的第二轮指令

C0—C1实际完成报告见`reproduction/reports/c0_c1_completion_report.md`。第二轮完整指令见：`暑期工程实践/给Codex的第二轮执行指令_C2-C3-Smoke.md`。

第二轮先完成C2—C3工程和Adult三生成器smoke，不直接启动两表×三生成器×5 seeds正式训练。只有访问审计、四池隔离、污染构造、bags重建和协议泄漏测试通过后，才单独下达正式训练指令。
