# 给 Codex 的第二轮执行指令（C2—C3 工程与冒烟验收）

将下面代码块中的全部内容原样发送给 Codex。本轮实现生成器、合成池、污染构造和比例bags的工程闭环，并在Adult上完成三生成器冒烟；不启动正式5-seed大实验。

```text
你现在需要在 E:\毕设 工作区继续执行毕业设计“固定基准与算法复现”工程。C0—C1已经验收通过，本轮执行“C2—C3工程与冒烟验收”，完成后停止，不进入C4，也不启动两表×三生成器×5个正式种子的高成本训练。

一、开始前必须完整阅读

1. E:\毕设\reproduction\reports\c0_c1_completion_report.md
2. E:\毕设\reproduction\reports\current_state_audit.md
3. E:\毕设\暑期工程实践\第3步_固定基准规格_v1.md
4. E:\毕设\暑期工程实践\第5步_算法复现与公平比较方案_v1.md
5. E:\毕设\暑期工程实践\CODEX_固定基准与算法复现实施计划.md
6. E:\毕设\reproduction\README.md
7. E:\毕设\reproduction\configs\benchmark_v1.yaml及现有数据集、生成器配置
8. E:\毕设\reproduction\src\tabpollution和tests中的现有代码

约束优先级仍为：第3步冻结基准 > 第5步算法方案 > Codex实施计划 > 旧README/旧脚本。不要修改已冻结的数据集、正式种子、分区、污染率、bag规模或指标。

二、已验收的起点

以下内容已经通过验收，不要重新设计：

1. Adult清洗后48,790行；Credit清洗后29,965行；
2. 正式seed为2026、2027、2028、2029、2030，smoke seed为42；
3. 真实分区固定为R_source_train=60%、R_detector_train=15%、R_detector_val=10%、R_final_test=15%；
4. Adult每个正式seed的分区规模为29,273/7,319/4,879/7,319；
5. Credit每个正式seed的分区规模为17,978/4,495/2,997/4,495；
6. C0—C1最终测试为22 passed；
7. 原run_minimal_loop.py、run_adult_baseline.py和outputs均为legacy，只读保留；
8. run_id=c0-c1-20260715及其产物不得覆盖；
9. 工作区git元数据无效，继续在manifest中如实标记，不得伪造commit；
10. B-small只注册未准备，本轮不下载、不运行Track B。

三、本轮范围与明确停止线

本轮必须完成：

1. C2统一generator接口、SDV三生成器适配、访问审计、四类隔离合成池、模型/配置/日志/provenance、质量检查和Adult smoke；
2. C3四种污染构造、7档比例、smoke比例bags、bag manifest、按manifest重建和P1—P5协议验证器；
3. 单元测试、集成测试、可重复性检查和C2—C3 smoke报告。

本轮禁止：

1. 不运行正式seed=2026—2030的生成器训练；
2. 不运行CTGAN/TVAE正式300 epochs；
3. 不在Credit上训练生成器；
4. 不下载B-small或14表；
5. 不实现或运行C2ST、XGBoost、3-gram、Transformer、比例估计算法或估值算法；
6. 不开始C4；
7. 不把seed=42、20 epochs、缩小训练集或smoke bags写成正式结果；
8. 不覆盖C0—C1的data、manifests、reports、runs或legacy输出。

四、环境处理要求

1. 开始先重新运行现有22项测试，确认C0—C1未退化；
2. 生成器实际运行优先直接调用：
   D:\Anaconda3\envs\tabpollution\python.exe
   该环境已确认Python 3.11.15、SDV 1.37.3、SDMetrics 0.28.0；不要优先使用会触发GBK进度输出错误的conda run包装器；
3. 开始训练前实际检测torch.cuda.is_available()、torch版本、CUDA版本和GPU名称，不因系统存在RTX 4070就假定当前torch可用CUDA；
4. 若生成环境缺少本轮必要的轻量依赖，可安装到E:\毕设\reproduction\.deps_c2或为该环境补充明确版本，但不要无关升级SDV、pandas、scikit-learn、torch或CUDA；
5. SDV/SDMetrics导入必须采用延迟导入，使不含SDV的基础测试环境仍可运行纯数据和协议测试；
6. 分别保存基础测试环境和生成器环境报告；所有实际版本写进run manifest；
7. 如果GPU不可用，允许CPU完成smoke；若全量R_source_train的CTGAN/TVAE 20 epochs耗时明显不合理，可退回到从R_source_train确定性抽取的3,000行smoke子集，但必须在配置、run_id、报告中写明fit_scope=smoke_subset_3000，不能与全量结果混报。GaussianCopula优先使用完整R_source_train。

五、C2：统一生成器接口

在现有模块化结构中新增src/tabpollution/generators，不要把逻辑重新堆进单脚本。至少实现以下接口：

fit(real_records, metadata, seed, **config)
sample(n, sample_seed, pool_name) -> records
save(path)
load(path)
get_provenance() -> mapping

要求：

1. 定义BaseGenerator/Protocol和统一异常；
2. 接入SDV GaussianCopulaSynthesizer、CTGANSynthesizer、TVAESynthesizer；
3. 生成器名称、参数和SDV类之间有严格registry，未知生成器显式失败；
4. 模型只接收由冻结split manifest筛出的R_source_train；调用fit前生成输入row_id清单及SHA-256；
5. 访问审计必须记录允许分区、实际读取row_id、数量、哈希和检查结论；一旦混入R_detector_train、R_detector_val或R_final_test立即失败；
6. 送入SDV模型前删除row_id、split、provenance等元数据；target保留为表格字段，以便后续TSTR；
7. metadata必须由R_source_train及dataset registry构建，不能扫描其他分区；
8. save/load后能继续sample；保存SDV模型、resolved generator config、metadata、输入审计、环境、stdout/stderr、训练时间和采样时间；
9. 不假定不同SDV版本参数相同。先核对本机SDV 1.37.3的真实构造函数，再写适配，不要照旧API猜参数；
10. 对无法完全确定的CTGAN/TVAE记录torch、CUDA、cuDNN和deterministic设置；不能承诺第三方库做不到的逐字节模型复现。

六、C2：四个隔离合成池

每个“dataset×generator×generator_seed”必须通过四次独立sample调用生成：

1. S_detector_train；
2. S_detector_val；
3. S_final_test；
4. S_downstream_mix。

硬性要求：

1. 四个池使用显式、不同且可重建的sample_seed；sample seed派生规则写入配置和测试；
2. 禁止先生成一个大CSV再随意切分；
3. 每条记录保存synth_row_id、dataset_id、generator_name、generator_seed、sample_seed、pool_name；
4. synth_row_id在四池全局唯一且确定，不能只使用临时行号；
5. 模型特征加载接口必须排除所有上述provenance字段；
6. 每个池保存schema、行数、文件SHA-256、内容哈希和来源模型run_id；
7. smoke池最小规模：
   - S_detector_train不少于R_detector_train规模的110%；
   - S_detector_val不少于max(R_detector_val规模的110%, 1000)；
   - S_final_test不少于max(R_final_test规模的110%, 1000)；
   - S_downstream_mix不少于R_source_train规模的110%；
8. 若CTGAN/TVAE采用smoke_subset_3000拟合，仍可按上述要求采样，但必须将“拟合子集规模”和“采样规模”分开记录；
9. 默认优先保存CSV+JSON schema/manifest，除非现有环境已经稳定支持Parquet；不要只为Parquet引入大型依赖。

七、C2：Adult三生成器smoke

仅运行Adult、smoke seed=42：

为避免重新发明一套未冻结的真实划分，本轮固定读取Adult已经存在的`split_seed=2026`作为smoke数据分区；`generator_seed=42`和各pool的sample_seed负责模型与采样随机性。不要新建或覆盖“real split seed=42”，并在所有manifest中同时区分`split_seed=2026`与`generator_seed=42`。

1. GaussianCopula：完整R_source_train快速拟合，显式固定enforce_min_max和enforce_rounding；
2. CTGAN：20 epochs；batch_size、pac、cuda、verbose等最终实际值写入resolved config；
3. TVAE：20 epochs；batch_size、embedding_dim、compress_dims、decompress_dims、cuda等最终实际值写入resolved config；
4. 三者分别生成四个隔离池；
5. 每个生成器拥有独立run_id，不共享或覆盖模型目录；
6. 失败时先定位依赖、schema、batch/pac合法性和CUDA问题；不能偷偷换成另一生成器、减少到0 epochs或复用旧CTGAN CSV；
7. 某生成器在合理修复后仍无法完成时，将状态标记blocked，保存异常、已尝试方案和下一步；其余生成器继续完成，不伪造成功。

八、C2：生成质量、伪影和TSTR smoke检查

对每个Adult smoke生成器至少输出：

1. SDMetrics overall、Column Shapes、Column Pair Trends；
2. 合成池内部完全重复率；
3. 四池之间的内容重合率和synth_row_id交集；
4. 合成记录与R_source_train精确重合率；
5. schema、列顺序、数值/类别dtype和目标取值合法率；
6. 缺失标记、字符串前后空格、数值小数/整数格式等伪影摘要；
7. 训练时间、各池采样时间、模型大小、CPU/GPU状态；
8. TSTR smoke：用S_downstream_mix训练固定LR，在纯真实R_final_test上评测；
9. TRTR参照：用相同数量的R_source_train训练相同LR，在同一R_final_test上评测；
10. TSTR/TRTR只用于生成质量smoke，不能进入正式效用主表；预处理器分别只在各自训练集拟合，不能在R_final_test拟合或调参。

质量检查的主要任务是发现训练失败和格式泄漏。不能因为某生成器更容易被识别就称其“检测效果更好”。

九、C3：污染构造接口

在src/tabpollution/mixing中实现可复用函数和manifest，不在Notebook手工拼表。固定支持：

1. real_only：N条纯真实基底；
2. real_append：N条真实基底再追加round(pN)条真实记录；
3. synthetic_append：N条真实基底再追加round(pN)条S_downstream_mix记录；
4. synthetic_replace：从N条真实基底移除round(pN)条，再加入同量S_downstream_mix，最终仍为N条；
5. p固定为0、0.05、0.10、0.25、0.50、0.75、1.00；
6. round规则必须明确为一种固定实现并测试，不依赖Python银行家舍入的隐含行为；推荐half-up或floor(pN+0.5)，写入manifest；
7. 每条混合记录保存mix_row_id、source_type、source_row_id/synth_row_id、generator、pool、p、mix_seed和condition；
8. 任何检测/下游模型读取特征时自动删除全部来源真值和元数据；
9. 同一配置重建得到相同成员和顺序哈希；
10. synthetic_replace的0%等于real_only成员，100%全部来自S_downstream_mix；
11. real_append所需“额外真实池”在冻结方案中没有独立来源。不要静默把测试集加入训练。本轮API必须要求显式real_extra_pool；Adult smoke可从R_source_train进行确定性有放回bootstrap，并在condition中标记real_append_bootstrap_control。报告中把它列为待导师/研究方案明确的实验设计点，不能表述为新增独立真实样本。

十、C3：比例估计smoke bags

正式配置中的bag_size=1000、calibration_bags=50、test_bags=100保持不变。本轮为验证工程，只新增独立smoke配置：每个比例生成2个calibration bag和3个test bag，bag_size仍必须为1000。

对Adult×三smoke生成器×7比例执行：

1. calibration bag只能使用R_detector_val和S_detector_val；
2. test bag只能使用R_final_test和S_final_test；
3. calibration与test的真实row_id集合、synth_row_id集合和sample pool必须隔离；
4. 每个bag内部默认无放回；不同bag之间允许重用记录，但manifest必须标明；
5. 每个bag合成数量严格等于固定round规则得到的数量；
6. 保存bag_id、真实比例、实际比例、real_count、synthetic_count、记录ID、来源、dataset、generator、seed、mix_seed和构造配置；
7. 不必物化保存每个完整bag的重复表格副本。优先保存成员manifest，并提供按bag_id从冻结真实表和合成池重建的命令；
8. 重建后打印成员数量、来源组成、实际比例和内容SHA-256；
9. 0%和100%边界必须有真实集成测试；
10. 本轮不运行CC/PCC/ACC或其他量化器。

十一、C3：协议验证器

实现P1—P5 manifest验证器，即使P3—P5本轮没有真实B-small数据，也必须用测试夹具验证逻辑：

1. P1：训练/测试表相同、生成器相同，记录ID和用途池不重叠；
2. P2：表相同、训练生成器与测试生成器不重叠；
3. P3：训练表与测试表不重叠；
4. P4：训练/测试表不重叠且训练/测试生成器不重叠；
5. P5：训练/测试领域不相同，结果不得混入P3宏平均；
6. 对每条规则编写正例和故意注入泄漏的反例；反例必须稳定失败并指出冲突表、生成器或record ID。

十二、CLI、配置和目录要求

在现有CLI上增加清楚、可重复的命令，实际命名可调整，但至少覆盖：

1. generator smoke：训练/恢复一个生成器并生成四池；
2. generator validate：校验访问、四池、schema、质量和provenance；
3. mixing build：构造四类污染条件；
4. bags build：生成smoke bag manifests；
5. bags inspect/rebuild：按bag_id重建并显示真值；
6. protocol validate：校验P1—P5 manifest。

新增配置应区分：

- frozen formal defaults；
- smoke override；
- runtime resolved config。

不得修改benchmark_v1.yaml中的正式seed、正式比例或50/100正式bag数量来迁就smoke。所有新run必须唯一，不得覆盖；smoke/debug/failed/formal必须有显式run_type字段或等价机制，后续聚合默认排除smoke/debug/failed。

十三、必须实现并运行的测试

先保留并通过现有22项测试，再新增至少覆盖：

1. 生成器fit输入只含R_source_train，注入其他分区row_id时失败；
2. 送入模型的表不含row_id/provenance；
3. 三个SDV适配器registry和未知生成器错误；
4. sample seed派生稳定且四池不同；
5. 四池synth_row_id互斥、schema/列序一致；
6. save/load后可采样，且输出通过schema检查；
7. 合成内部重复、真实重合、池间重合统计在人工夹具上数值正确；
8. 污染数量round规则在奇数N和0/5/10/25/50/75/100%正确；
9. synthetic_replace总规模固定，synthetic_append规模正确；
10. 0%/100%边界正确；
11. 混合构造可重复，元数据不会进入特征；
12. calibration/test真实与合成成员严格隔离；
13. bag内部无重复、数量1000、实际比例准确；
14. 同一bag_id从manifest重建内容哈希一致；
15. P1—P5正例通过、注入表/生成器/记录泄漏的反例失败；
16. Adult GaussianCopula、CTGAN、TVAE各完成一次端到端smoke集成测试；
17. C0—C1数据、split和legacy哈希回归检查仍通过。

深度模型测试不得每次pytest都重新训练。昂贵smoke只运行一次并保存artifact；常规测试使用小型fake generator或已保存的最小artifact。不得把“测试通过”建立在网络实时下载或随机未固定行为上。

十四、阶段闸门

只有同时满足以下条件，才能把本轮标记complete：

1. 原22项测试继续通过；
2. 三个Adult smoke生成器均成功，或失败项被如实标记blocked且其余工程可验收；
3. 每个成功生成器都有模型、resolved config、环境、访问审计、四池manifest和质量报告；
4. 没有任何生成器读取R_detector_train、R_detector_val或R_final_test；
5. 四池ID互斥，schema和provenance检查通过；
6. 四种污染条件和7档比例通过精确数量测试；
7. 三生成器的smoke calibration/test bags可按manifest重建；
8. P1—P5泄漏验证器的正反例测试通过；
9. 原legacy和C0—C1产物哈希未变化；
10. 没有启动正式5-seed/300-epoch训练或C4算法。

十五、本轮必须交付

1. src/tabpollution/generators及统一SDV适配器；
2. src/tabpollution/mixing及污染/bag/协议模块；
3. generator、mixing、bags、protocol配置和CLI；
4. Adult×GaussianCopula/CTGAN/TVAE smoke模型及四池；
5. 访问审计、模型/池provenance和合成池总manifest；
6. 三生成器质量卡、TSTR/TRTR smoke结果；
7. 四种污染条件smoke manifest；
8. 3生成器×7比例×(2 calibration+3 test)的bag manifests；
9. P1—P5验证结果；
10. 新增测试及JUnit结果；
11. 唯一run manifest和C2—C3 smoke完成报告；
12. 更新README，给出从环境设置到重建任意bag的真实成功命令。

十六、结束汇报格式

完成后停止，不进入C4。最终报告必须基于实际文件和实际运行结果，包含：

- 阶段：C2—C3工程与冒烟验收；
- 完成、blocked和未执行内容；
- 新增/修改文件；
- 实际Python/SDV/SDMetrics/torch/CUDA/GPU环境；
- 每个生成器的fit scope、epochs、参数、训练时间、采样时间、模型大小和run_id；
- 每个生成器四池规模、sample_seed、SHA-256及互斥检查；
- 访问审计结果，明确实际读取的真实分区和row_id数量；
- SDMetrics、重复率、真实重合率、格式合法率、TRTR/TSTR；
- 四类污染构造的规模与比例检查；
- bags数量、隔离检查、任选bag重建示例和SHA-256；
- P1—P5正反例验证结果；
- 测试总数、通过/失败/跳过及耗时；
- C0—C1与legacy哈希回归结果；
- 与冻结规格的任何差异和原因；
- 风险、技术债和blocked项；
- 是否满足启动“C2正式两表×三生成器×5 seeds”的条件；
- 下一阶段建议，但不要自行执行。

不要用计划值代替实际值，不要把smoke结果写成正式结论，不要因为部分生成器失败而删除失败记录。
```
