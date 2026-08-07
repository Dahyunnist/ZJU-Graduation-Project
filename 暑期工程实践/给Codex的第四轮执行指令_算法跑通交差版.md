# 给 Codex 的第四轮执行指令（算法跑通交差版）

将下面代码块中的全部内容原样发送给 Codex。本轮目标不是形成正式五种子论文结果，而是在已有固定基准上，把三条任务线的经典方法、直接论文基线和SOTA核心真正跑通，形成导师可检查的代码、日志、结果与进展表。完成后停止，将工作重心转回文献综述与研究方案。

```text
你现在需要在 E:\毕设 工作区继续执行毕业设计“合成表格数据污染检测、比例估计与估值影响分析”工程。此前已经完成C0–C3基础设施、Adult三生成器smoke、C2正式运行加固和GaussianCopula seed=2026 pilot。本轮改变工作重心：不继续等待完整两表×三生成器×5 seeds正式基准，而是在现有数据和smoke产物上完成“算法跑通交差版”。

本轮完成后停止，不自动启动正式五种子实验，也不直接改写国内外研究综述与研究方案正文；只生成可供后续写作引用的算法事实材料与进展表。

一、开始前必须完整阅读

按顺序阅读：

1. E:\毕设\reproduction\reports\c2_formal_readiness_pilot_completion_report.md
2. E:\毕设\reproduction\reports\c2_c3_smoke_completion_report.md
3. E:\毕设\暑期工程实践\阶段二_直接相关文献与算法复现清单.md
4. E:\毕设\暑期工程实践\阶段二_文献完整性与直接相关性复核.md
5. E:\毕设\暑期工程实践\阶段二_任务定义与评测规范.md
6. E:\毕设\暑期工程实践\第3步_固定基准规格_v1.md
7. E:\毕设\暑期工程实践\第5步_算法复现与公平比较方案_v1.md
8. E:\毕设\暑期工程实践\CODEX_固定基准与算法复现实施计划.md
9. D1、D2、D3、D4、D5、Q1–Q5、V2、V4的本地原文与中文梳理
10. E:\毕设\reproduction\README.md、configs、src、tests、runs和reports现状

冲突优先级：第3步冻结基准 > 第5步算法方案 > 任务定义与评测规范 > 本指令 > 旧README/legacy脚本。

二、“算法跑通交差”的定义

本轮的目标状态是runthrough/smoke_passed，不是reproduced或formal。每个纳入算法至少满足：

1. 有明确论文出处；
2. 有代码来源结论：官方仓库、第三方实现或论文对齐自实现；
3. 若使用外部仓库，记录URL、commit、license、检索日期和本地目录；
4. 能由命令行完成训练、推理和主要指标计算；
5. 保存resolved config、环境、stdout、stderr、timing、模型或必要状态、预测/估计结果和metrics；
6. 至少在一个固定mini协议上端到端运行成功；
7. 有单元测试或小型精确核验；
8. 能说明与论文原设定的差异；
9. run_type固定为runthrough或smoke，不得进入formal汇总；
10. 失败也保留结构化记录，不能静默从进展表删除。

仅“包安装成功”“类可以import”“Notebook打开了”不算跑通。反过来，本轮也不要求五种子、完整14表或论文全部报告值复现。

三、本轮范围与最终必须跑通的算法矩阵

任务线一：样本级检测，必须跑通：

1. D-LR：C2ST Logistic Regression，同时输出pMSE诊断；
2. D-XGB：C2ST XGBoost；若xgboost不可用，可先排查安装，不能用RandomForest冒充；
3. D-3G：Character 3-gram + Logistic Regression；
4. D-FT：Flat-text character Transformer；
5. D-TT：Table/column-wise Transformer；
6. D-DW：Datum-wise Transformer核心版；
7. D-DWTA：Datum-wise + table adaptation最小版，与D-DW成对运行。

任务线二：语料级污染比例估计，必须跑通：

1. Q-CC；
2. Q-PCC；
3. Q-ACC；
4. Q-PACC；
5. Q-EMQ/SLD；
6. Q-HDy；
7. Q-DyS；
8. Q-Median Sweep。

任务线三：估值影响分析，必须跑通：

1. V-CURVE-LR：污染比例—LR下游效用曲线；
2. V-CURVE-XGB：污染比例—XGBoost下游效用曲线；
3. V-KNN：KNN-Shapley；
4. V-OOB：Data-OOB。

本轮暂不强制：KDEy、Continuous Sweep、TMC-Shapley、Data Banzhaf、XAI解释、LLM检测、TabDDPM、TabSyn、B-full十四表和正式五种子。可完成但不得阻塞上述必跑矩阵，也不得用可选项代替必跑项。

四、论文代码与外部仓库审计

在实现SOTA前必须重新联网检查，截至执行日核对：论文页面、扩展版、作者主页、GitHub、GitLab和论文引用页面。

重点检查：

1. D1 Cross-table Synthetic Tabular Data Detection；
2. D2 Synthetic Tabular Data Detection in the Wild；
3. D3 Robust Detection of Synthetic Tabular Data Under Schema Variability；
4. QuaPy官方仓库：https://github.com/HLT-ISTI/QuaPy；
5. Data-OOB论文与作者公开代码；
6. KNN-Shapley论文作者公开实现；
7. XGBoost、PyTorch等只使用官方包来源。

当前已知事实：D3的AAAI 2026正式页面写明“代码将在扩展版开放”，此前尚未找到公开官方仓库。执行时必须重新检查，但不得把同名非官方仓库误写成官方代码。

输出`reports/code_availability_audit.md`和JSON，字段至少包含：algorithm_id、paper、paper_url、searched_locations、official_code_url、commit、license、retrieved_at、conclusion、evidence。

处理规则：

- 找到官方代码：固定commit，记录license，在`external/`隔离保存；先跑官方最小命令，再接统一接口；
- 只有第三方实现：明确标记third_party，不覆盖自实现；
- 没有代码：标记paper_aligned_reimplementation，列出论文明确细节、缺失细节和本地假设；
- 不允许伪造官方仓库、commit、license或论文超参数；
- 外部仓库不得散落在毕设根目录，统一放`reproduction/external/<algorithm_or_repo>`；
- 不修改外部仓库原始代码来掩盖兼容问题；必要补丁单独保存patch和说明。

五、开始前回归、环境和保护要求

1. 重新运行当前完整测试，预期至少85项通过；
2. 运行data validate、旧generator validate、Gaussian pilot validate；
3. 复核C0–C3和第三轮关键哈希；
4. 不覆盖任何既有run、report、smoke模型或pilot；
5. 保存本轮基础CPU环境报告；
6. 检查xgboost、torch、CUDA、scipy、joblib、pyarrow、QuaPy等实际版本；
7. 深度检测器优先使用隔离环境，不能修改已验收的tabpollution生成器环境；
8. 若GPU环境仍不可用，允许D-FT/D-TT/D-DW/D-DWTA使用CPU tiny配置跑通，但必须标记`cpu_tiny_runthrough`，不得报告为论文性能复现；
9. 不以等待GPU为由跳过全部SOTA，至少要在CPU tiny配置完成前向、反向、保存、加载和推理闭环。

六、为跨表算法建立最小三表runthrough轨道

已有Adult和Credit。为了使D-3G、D-FT、D-TT、D-DW和D-DWTA不是只在单表上空转，本轮允许新增一个小型公开表，优先使用UCI Abalone，形成：

- Adult；
- Credit；
- Abalone。

要求：

1. 只从官方/权威来源下载Abalone，保存URL、license、日期、SHA-256和数据卡；
2. 生成稳定row_id和固定runthrough划分，不修改Adult/Credit冻结划分；
3. Abalone只服务跨表检测runthrough，不进入Track A正式主结果；
4. 为Credit和Abalone各用GaussianCopula快速生成一组runthrough合成池；生成器只能读取各自允许真实训练分区；
5. 不训练新的CTGAN/TVAE，不等待正式C2；
6. 数据和生成run均标记runthrough；
7. 三表每表只取足以跑通算法的确定性子集，默认建议每类训练1,000、验证300、测试500；若表规模不足，按可用规模调整并记录；
8. 构造三个leave-one-table-out mini folds，每次两表训练、一表测试；
9. D-DWTA的table分类头至少在两个训练表上有意义；
10. 该轨道只称`three-table mini cross-table runthrough`，不得写成D1/D3的14表正式复现。

如果Abalone官方数据暂时无法取得，不用来源不明镜像。先尝试另一个本地文献明确、官方来源可获得的小表；仍失败时，用三种不同schema的人工fixture只验证架构，并把真实跨表结果标记blocked。D-3G和D-DW至少仍应在Adult/Credit两表完成一个双向mini fold。

七、统一检测器接口和数据表示

在`src/tabpollution/detectors`建立统一接口：

fit(train_records, train_labels, val_records=None, val_labels=None, **context)
predict_score(records, **context) -> 一维数组，越大越像合成
save(path)
load(path)
get_provenance() -> mapping

统一要求：

1. 来源标签定义固定：real=0，synthetic=1；
2. row_id、synth_row_id、split、pool、generator、table_id真值、文件名和provenance不得进入模型特征；
3. table_id只允许在协议验证、分组和D-DWTA对抗表头真值中使用，不作为检测输入捷径；
4. 阈值和概率校准只用validation；
5. 输出原始score与校准概率；
6. 所有算法读取同一fold manifest和相同测试ID；
7. 指标至少包括AUROC、AUPRC、balanced accuracy、F1、TPR@FPR=5%、Brier、ECE、训练时间、推理时间和模型大小；
8. 保存逐样本prediction，包括record_id、source_label、raw_score、probability、prediction、table、generator、fold和run_id；
9. 测试集标签只用于最终指标，不可用于调参、早停或校准；
10. 每个run保存config、环境、日志、timing、模型、predictions、metrics、artifact manifest和status。

八、D-LR和D-XGB跑通要求

D-LR：

1. 数值中位数填补+StandardScaler；类别众数填补+OneHot；
2. C候选固定为0.01、0.1、1、10，只在val选择；
3. 真实/合成训练按1:1抽样；
4. 同时计算propensity score MSE/pMSE及其定义说明；
5. 在Adult GaussianCopula P1 smoke/runthrough完成；
6. 做标签置换sanity check，AUROC应接近0.5；
7. 做格式元数据伪影分类器，若明显高于随机，正式指标加warning。

D-XGB：

1. 使用XGBoost官方Python包；
2. 小型固定搜索空间，只用val AUROC选择；
3. 记录xgboost版本、tree_method、线程、early stopping和类别编码；
4. 在与D-LR完全相同的Adult记录ID上运行；
5. 做标签置换sanity check；
6. 特征重要性可输出gain，SHAP不是本轮必需；
7. 若CPU运行，限制线程避免占满系统；
8. D-LR/D-XGB的P1结果只作为算法跑通证据，不作为正式主表。

九、D1/D2三条直接基线跑通要求

D-3G：

1. 每个单元规范化为`<column>:<value>`；
2. 数值、缺失、布尔、字符串格式统一；
3. 每条记录用固定seed打乱列片段，测试同时报告原顺序与随机置换；
4. `TfidfVectorizer(analyzer='char', ngram_range=(3,3)) + LogisticRegression`；
5. vectorizer只能拟合训练文本；
6. 保存词表大小、稀疏矩阵内存和测试未知3-gram率；
7. 跑Adult P1和三表mini cross-table folds。

D-FT：

1. 原始字符级文本输入，不使用预训练LLM；
2. 保存字符词表/tokenizer；
3. 明确最大长度、截断率、padding、embedding维度、层数、head数；
4. tiny配置可用较小维度和1–2层，但必须显式标记与论文差异；
5. 跑至少一个三表mini fold，完成训练、保存、加载、推理和指标。

D-TT：

1. 按D1/D2描述实现table/column-wise直接基线；
2. 数值/类别预处理只能在训练表拟合；
3. 跨表共享表示不能依赖固定列数；
4. 若论文细节不足，将所有推断写入`reproduction_notes.md`；
5. 跑与D-FT相同mini fold和记录ID；
6. 输出参数量、截断/列数统计和列置换敏感性。

十、D3 Datum-wise与table adaptation跑通要求

D-DW最小论文对齐结构：

1. 单元表示为`<column>:<value>`字符序列；
2. datum encoder含字符embedding、datum内部局部位置编码和CLS-Datum；
3. 每行组合所有CLS-Datum和CLS-Target；
4. row transformer不加入跨列全局位置编码；
5. CLS-Target进入真实/合成二分类头；
6. BCE训练，validation AUROC早停；
7. 配置记录embedding维度、datum/row层数、heads、最大datum长度、最大列数、dropout和optimizer；
8. 对同一行随机打乱列10次，预测最大差异低于明确容差；
9. 先过拟合一个tiny batch，证明loss下降；
10. 跑至少一个三表mini cross-table fold并保存预测。

D-DWTA：

1. 在D-DW上增加gradient reversal和table分类头；
2. 适配权重采用论文描述的cosine schedule；
3. 保存检测loss、table loss和adaptation weight曲线；
4. 训练表至少两个，table标签不能进入检测推理输入；
5. 与D-DW在相同fold、相同初始化策略和记录ID上成对比较；
6. 输出D-DW与D-DWTA的mini消融表；
7. 若官方代码仍未开放，implementation必须写`paper_aligned_self_implementation`；
8. 不以达到论文AUC为通过条件；结构、训练闭环、列置换不变性、预测和差异说明完整即可认定runthrough。

十一、样本检测sanity checks

所有检测器至少验证：

1. 来源标签反转测试能被捕获；
2. 训练/验证/测试record ID无交集；
3. table和generator协议无泄漏；
4. provenance列无法进入特征；
5. 标签置换后性能接近随机；
6. 仅格式元数据模型结果被单独报告；
7. 真实/合成使用相同序列化、dtype和缺失值规范；
8. 原始列序和随机列序测试；
9. 深度模型save/load预测误差在容差内；
10. 极小数据上能过拟合，正常数据上无NaN loss。

十二、比例估计统一接口与运行数据

在`src/tabpollution/quantification`实现：

fit(calibration_scores, calibration_labels, **context)
predict_prevalence(test_scores, **context) -> raw与clipped估计
save/load

本轮统一使用同一个已跑通的D-XGB Adult GaussianCopula分数，不能为不同量化器更换检测器。若D-XGB分数异常，再使用D-LR做诊断，但主runthrough表仍保持同分数比较。

使用已有Adult GaussianCopula smoke比例bags：每比例2个calibration、3个test、bag_size=1000。它们足以算法跑通，但不是正式50/100 bags。

严格隔离：

- calibration只来自`R_detector_val + S_detector_val`；
- test只来自`R_final_test + S_final_test`；
- 量化器不得重新训练检测器；
- 真实比例只用于拟合允许的校准方法或最终评价，不能泄漏给测试预测。

十三、八个比例估计器跑通要求

手写并精确测试：

1. CC：验证集阈值后的硬计数；
2. PCC：校准概率平均；
3. ACC：基于validation TPR/FPR校正，分母接近0时返回结构化失败；
4. PACC：保存软TPR/FPR和校正前后值。

QuaPy/论文方法：

5. 固定QuaPy官方仓库commit或明确PyPI版本与BSD-3-Clause许可；
6. EMQ/SLD：保存迭代次数、收敛、初始/最终先验；
7. HDy：固定bins候选及选择规则；
8. DyS：固定距离、bins和搜索网格；
9. Median Sweep：明确使用的修复版本、阈值集合和有效阈值数量。

要求：

1. 手写CC/PCC/ACC/PACC与QuaPy等价实现做小fixture交叉核验；
2. 完美分类器fixture应接近真实比例；
3. 随机/常数分类器应触发不稳定或合理退化；
4. 输出raw estimate、clipped estimate和out-of-range标记；
5. 指标至少为MAE、RMSE、Bias、最大绝对误差和越界率；
6. 单列5%和10%结果；
7. 保存每个bag每个方法的估计；
8. 输出八方法同分数runthrough表和误差曲线；
9. 不把3个test bags/比例写成正式统计结论。

十四、下游效用曲线跑通要求

使用已有Adult GaussianCopula C3 smoke污染条件，运行：

1. 下游LR；
2. 下游XGBoost；
3. 7个比例；
4. `real_only`、`real_append_bootstrap_control`、`synthetic_append`、`synthetic_replace`四条件；
5. 测试始终为纯真实`R_final_test`；
6. 预处理与模型只能在对应训练集拟合；
7. 超参数不得在`R_final_test`选择；
8. 输出AUROC、AUPRC、F1、balanced accuracy、log-loss、训练时间和训练规模；
9. 输出相对real_only的Delta utility；
10. 绘制污染比例—效用曲线，并明确这些是单generator/smoke条件的runthrough趋势。

若直接读取旧84个CSV耗时或空间不合理，使用已经实现的manifest-first重建；不要复制新的完整污染CSV。

十五、KNN-Shapley跑通要求

1. 按Jia等论文的无权KNN-Shapley递推实现；
2. 先在N≤10或N≤12玩具分类数据上枚举全部子集，计算精确Shapley；
3. 与递推结果逐点比较，误差低于明确容差；
4. 再在Adult GaussianCopula `synthetic_replace` 的一个中等污染比例上运行确定性子集，建议总训练样本1,000–3,000；
5. validation/test必须纯真实；
6. 数值/类别编码和距离度量固定并记录；
7. 输出真实/合成价值均值、中位数、分位数、负价值比例；
8. 输出价值作为来源诊断分数的AUROC，但不把它冒充主检测器；
9. 运行删除最低价值5%、10%、20%后的下游效用变化；
10. 保存逐训练记录价值与来源。

十六、Data-OOB跑通要求

1. 先审计是否存在作者官方代码；有则先跑官方最小示例；
2. 无官方代码时按论文定义实现基于bagging弱学习器OOB预测的数据价值；
3. 在小fixture上检查每条记录只由未包含它的弱学习器评价；
4. 保存每条记录OOB覆盖次数，覆盖为0时显式失败/缺失，不填0冒充价值；
5. 在与KNN-Shapley相同Adult污染子集上运行；
6. 输出真实/合成价值分布、负价值比例、删除曲线；
7. 计算与KNN-Shapley价值排序的Spearman；
8. 记录弱学习器数量、bootstrap seed、基学习器、训练时间和覆盖分布；
9. 只称Data-OOB runthrough，不宣称完整12数据集论文复现。

十七、run、配置和结果目录

建议新增：

configs/detectors/
configs/quantifiers/
configs/valuation/
configs/runthrough/
src/tabpollution/detectors/
src/tabpollution/quantification/
src/tabpollution/valuation/
src/tabpollution/evaluation/
external/
runs/runthrough-*/
reports/algorithm_runthrough/

每个算法run使用唯一run_id，最低包含：

- run_manifest.json；
- config_resolved.yaml；
- environment.json/txt；
- stdout.log、stderr.log；
- timing.json；
- metrics.json；
- predictions.csv/parquet或estimates.csv/parquet或values.csv/parquet；
- model/state；
- artifacts_manifest.json；
- status.json；
- 成功run的COMPLETE标记。

status至少区分：planned、installing、running、smoke_passed、runthrough_passed、paper_code_unavailable、blocked、failed。runthrough聚合器与formal聚合器分离，formal汇总永远排除本轮结果。

十八、CLI最低覆盖

实际命名可以调整，但必须提供等价命令：

1. detector run --algorithm <id> --config <runthrough-config>；
2. detector validate --run-id <id>；
3. detector compare --runs ...；
4. quantifier run --algorithm <id或all-classic> --score-run <detector-run>；
5. quantifier validate；
6. utility run --model <lr|xgb> --generator GaussianCopula；
7. valuation run --algorithm <knn_shapley|data_oob>；
8. runthrough aggregate；
9. report algorithm-progress；
10. 所有README命令必须是实际成功命令，不写未验证示例冒充成功。

十九、必须新增并运行的测试

保留当前全部测试，并至少覆盖：

样本检测：

1. provenance和真值列不会进入特征；
2. real=0、synthetic=1方向固定；
3. split、table、generator泄漏会失败；
4. LR/XGB在可分toy数据上AUROC接近1；
5. 标签置换结果接近随机；
6. 3-gram序列化确定且对列顺序检查符合设计；
7. vectorizer不读取test；
8. Transformer tokenizer、padding、截断正确；
9. D-DW列置换不变性；
10. D-DW tiny batch loss可下降；
11. gradient reversal符号正确；
12. D-DWTA table标签不进入检测输入；
13. 深度模型save/load预测一致；

比例估计：

14. CC/PCC/ACC/PACC公式；
15. 标签方向反转会失败；
16. 完美、随机、常数score fixture；
17. 分母接近0的结构化失败；
18. raw/clipped和越界率；
19. calibration/test ID隔离；
20. 八方法复用同一score文件；

估值：

21. KNN-Shapley与全子集精确值一致；
22. Data-OOB只使用OOB模型；
23. OOB覆盖0被识别；
24. 下游测试始终为R_final_test；
25. 训练条件规模和污染数正确；
26. 删除曲线成员选择稳定；

工程：

27. runthrough不能进入formal聚合；
28. 成功/失败日志与artifact manifest；
29. 半成品run不进入runthrough成功汇总；
30. C0–C3及第三轮关键哈希不变；
31. 每个必跑算法至少有一个真实保存run的集成验收；
32. 测试不重新执行所有昂贵深度训练，读取已保存最小artifact。

二十、分阶段执行顺序和中途闸门

严格按顺序：

R0：回归、环境、代码可用性审计、三表mini数据准备。

R1：D-LR、D-XGB。两者不通过标签置换、泄漏和伪影检查时，不继续量化器。

R2：D-3G、D-FT、D-TT。至少D-3G完成三表mini folds，两个Transformer各完成至少一个fold。

R3：D-DW、D-DWTA。必须先过tiny batch和列置换测试，再跑mini fold。

R4：用冻结D-XGB分数跑八个经典量化器。检测器分数一旦冻结，不因某量化器效果差而重训检测器。

R5：LR/XGB效用曲线、KNN-Shapley、Data-OOB。

R6：统一聚合、导师进展表、证据索引和完成报告。

每阶段失败时先定位依赖、协议、数据、标签方向和资源问题。不得用另一算法偷换、减少到0 epoch、复用无关旧预测或手工编造结果。

二十一、“暂时交差”的阶段闸门

只有同时满足以下条件，才能宣布本轮算法跑通任务暂时完成：

1. 原有全部测试继续通过，新增测试通过；
2. D-LR、D-XGB、D-3G均有真实端到端run；
3. D-FT、D-TT至少各有一个真实mini fold run；
4. D-DW通过tiny batch、列置换不变性，并有一个跨表mini run；
5. D-DWTA有两个训练表的table adaptation真实训练run，并与D-DW成对；
6. CC、PCC、ACC、PACC、EMQ、HDy、DyS、Median Sweep使用同一D-XGB分数跑完7比例smoke bags；
7. LR和XGBoost完成四条件×7比例的效用曲线；
8. KNN-Shapley通过全子集精确核验并在Adult污染子集产生记录价值；
9. Data-OOB在相同污染子集产生价值、覆盖次数和删除曲线；
10. 每个算法都有状态、配置、日志、结果和来源说明；
11. D1–D3官方代码可用性结论有证据，未找到时没有伪造；
12. 所有run明确是runthrough/smoke，不进入formal结果；
13. C0–C3与第三轮产物哈希未变；
14. 没有启动正式五种子、完整14表、两表三生成器正式矩阵；
15. 形成导师可直接查看的算法进展表和完成报告。

允许blocked的范围非常有限：只有外部官方代码/数据确实不可获得可以标记blocked，但对应论文对齐自实现的核心算法不能因此全部跳过。特别是D-DW核心必须至少完成自实现runthrough；D-DWTA必须至少完成最小训练闭环。若CPU资源不足，缩小batch、维度、层数和样本数，但不得改变核心结构后仍宣称论文性能复现。

二十二、必须交付的导师材料

1. `reports/algorithm_runthrough_completion_report.md`：完整技术报告；
2. `reports/supervisor_algorithm_progress_table.md`：面向导师的简洁表格；
3. `reports/reproduction_status.csv`：更新所有必跑算法状态；
4. `reports/code_availability_audit.md/json`；
5. `reports/algorithm_runthrough/detection_summary.csv`；
6. `reports/algorithm_runthrough/cross_table_mini_summary.csv`；
7. `reports/algorithm_runthrough/quantification_summary.csv`；
8. `reports/algorithm_runthrough/quantification_bag_estimates.csv`；
9. `reports/algorithm_runthrough/utility_curve.csv`与图；
10. `reports/algorithm_runthrough/valuation_summary.csv`；
11. `reports/algorithm_runthrough/knn_shapley_values.*`；
12. `reports/algorithm_runthrough/data_oob_values.*`；
13. `reports/algorithm_runthrough/dw_dwta_ablation.csv`；
14. `reports/algorithm_runthrough/paper_vs_local_scope.md`；
15. 每个算法成功命令、配置、日志和run_id索引；
16. 最终JUnit报告；
17. 唯一总run manifest；
18. README更新。

导师进展表至少包含：算法、对应论文、角色（经典/直接基线/SOTA）、实现来源、运行数据、协议、当前状态、关键结果、与论文差异、run_id、下一步。状态表不得把“自实现tiny smoke”写成“完整复现”。

二十三、完成报告必须回答的问题

完成后停止，并明确回答：

1. 实际跑通了哪些经典算法、直接论文基线和SOTA？
2. 哪些是官方实现，哪些是第三方，哪些是论文对齐自实现？
3. 每个算法在哪些数据、生成器和协议上跑通？
4. D1/D2三条基线是否全部跑通？
5. D3 Datum-wise与table adaptation是否完成最小复现？
6. 八个比例估计器是否在同一检测分数上比较？
7. KNN-Shapley是否通过精确核验？Data-OOB覆盖是否有效？
8. 哪些结果只能称runthrough，哪些已达到更高复现状态？
9. 与论文设置和报告值有哪些差异？
10. 哪些问题留待正式五种子阶段解决？
11. 当前材料是否足够支撑回到文献综述和研究方案写作？

最终汇报应给出实际测试数、实际运行时间、run_id、主要smoke结果、blocked/failed记录、资源情况和可点击文件路径。不要用计划值代替实际值，不要手工编造论文对齐结果，不要把生成器当检测算法，不要把量化器与样本检测混为一谈。
```
