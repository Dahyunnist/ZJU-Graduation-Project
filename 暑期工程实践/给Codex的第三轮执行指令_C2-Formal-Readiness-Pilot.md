# 给 Codex 的第三轮执行指令（C2 正式启动加固与单 seed 试运行）

将下面代码块中的全部内容原样发送给 Codex。本轮不是直接启动“两表 × 三生成器 × 5 seeds”的大矩阵，而是先修复上一轮暴露的正式运行风险，并用 Adult、seed=2026 做一次受控试运行。完成后停止，不进入 C4。

```text
你现在需要在 E:\毕设 工作区继续执行毕业设计“固定基准与算法复现”工程。C0–C1 与 C2–C3 Adult smoke 已完成验收。本轮执行“C2 正式启动加固与单 seed 试运行”，完成后停止；不要进入 C4，不要批量启动两表×三生成器×5 seeds。

一、开始前必须完整阅读

按顺序阅读：

1. E:\毕设\reproduction\reports\c2_c3_smoke_completion_report.md
2. E:\毕设\reproduction\runs\c2-c3-smoke-20260715\run_manifest.json
3. E:\毕设\暑期工程实践\第3步_固定基准规格_v1.md
4. E:\毕设\暑期工程实践\第5步_算法复现与公平比较方案_v1.md
5. E:\毕设\暑期工程实践\CODEX_固定基准与算法复现实施计划.md
6. E:\毕设\reproduction\README.md
7. E:\毕设\reproduction\configs、src/tabpollution、tests 中现有文件
8. 三个成功 C2 smoke run、三个成功 C3 smoke run及其failure记录

冲突优先级仍为：第3步冻结规格 > 第5步算法方案 > 本指令 > 旧README/legacy脚本。不得覆盖或重写已经验收的C0–C3产物。

二、已经验收的起点

1. 最终测试为59 passed；
2. Adult/Credit清洗数据和2026–2030正式分区已冻结；
3. Adult三生成器smoke、四类隔离合成池、污染构造、35个smoke bags与P1–P5验证均已通过；
4. GaussianCopula smoke使用完整R_source_train；CTGAN/TVAE只使用3,000行、20 epochs、CPU；
5. 当前生成器环境为Python 3.11.15、SDV 1.37.3、SDMetrics 0.28.0、torch 2.12.1+cpu，torch.cuda.is_available()为False；
6. TVAE smoke发生目标坍缩，只生成<=50K；CTGAN smoke的TSTR AUROC较差；这些是风险，不是正式结论；
7. C3 smoke物化CSV约占838 MB，正式多seed不能照此线性复制；
8. 成功run尚未单独物化stdout.log/stderr.log；SDV类级load存在弃用警告；
9. real_append当前是R_source_train有放回bootstrap control，不是独立新增真实数据；
10. git元数据无效，继续如实记录，不得伪造commit。

三、本轮目标

本轮只完成以下两部分：

A. 正式运行前工程加固

1. 为正式/试运行增加完整的日志、artifact manifest、状态和防覆盖机制；
2. 建立与smoke配置严格分离的formal配置和pilot override；
3. 增加正式运行预检与资源预算，尤其是GPU、磁盘、预计训练时间；
4. 把C3正式污染数据改为“成员manifest + 按需重建”，避免默认物化84个大CSV；
5. 建立生成质量硬闸门与警告项，防止TVAE目标坍缩被带入正式实验；
6. 补齐独立CLI和测试，但不得改变冻结数据、正式seeds、污染率、bag数量或指标。

B. Adult、正式配置、单seed=2026试运行

1. GaussianCopula必须执行一次完整pilot；
2. CTGAN与TVAE只有在本机实际满足GPU预检且资源预算合理时，才允许各执行一次完整R_source_train、300 epochs pilot；
3. 如果torch仍为CPU版或CUDA不可用，CTGAN/TVAE本轮不得偷偷改成CPU全量、减少epochs或继续使用3,000行冒充正式pilot；将它们标记为blocked_by_gpu，并给出可执行但未擅自执行的环境修复方案；
4. 所有本轮试运行的run_type必须为pilot，不得写成formal，也不得进入正式5-seed汇总；
5. 完成一个生成器后立即验收，不因另外两个blocked而伪造或删除结果。

四、明确停止线

本轮禁止：

1. 不进入C4，不实现或运行C2ST-LR、XGBoost、3-gram、Transformer或Datum-wise；
2. 不实现或运行CC/PCC/ACC/PACC/EMQ等比例估计算法；
3. 不实现或运行效用曲线、Shapley、Data-OOB等估值算法；
4. 不训练Credit生成器；
5. 不下载或运行B-small/B-full；
6. 不启动2027–2030，不启动正式5-seed矩阵；
7. 不静默安装或升级CUDA、PyTorch、SDV、pandas、scikit-learn；
8. 不删除约897.5 MB已有smoke产物，不覆盖任何旧run；
9. 不把pilot、smoke、debug、failed、blocked结果写入formal汇总；
10. 不根据pilot结果修改冻结数据划分、污染率、正式epochs、bag规模或主指标。

五、开始时的回归与哈希检查

1. 重新运行当前完整测试，必须先确认59项继续通过；
2. 运行data validate和generator validate；
3. 校验C0–C1数据、split、legacy以及六个成功C2/C3 smoke run的关键哈希；
4. 如果任何已验收哈希变化，立即停止本轮试运行，只调查并报告，不重建旧产物；
5. 保存本轮开始时的基础环境报告与生成器环境报告。

六、配置分层与run状态

新增配置时必须区分：

1. frozen formal defaults：继续引用benchmark_v1.yaml和正式generator配置；
2. pilot override：只限定dataset=adult、split_seed=2026、generator_seed=2026、run_type=pilot；不得降低正式epochs；
3. runtime resolved config：保存本次构造函数最终实际参数、fit scope、设备、版本和seed。

正式默认仍为：

- GaussianCopula：完整R_source_train，固定enforce_min_max/enforce_rounding；
- CTGAN：完整R_source_train，300 epochs，batch/pac/enable_gpu等实际值显式记录；
- TVAE：完整R_source_train，300 epochs，batch/embedding/compress/decompress/enable_gpu等实际值显式记录。

run_id建议使用：

- c2-pilot-adult-gaussiancopula-s2026-<timestamp或attempt>
- c2-pilot-adult-ctgan-s2026-<timestamp或attempt>
- c2-pilot-adult-tvae-s2026-<timestamp或attempt>

每个run必须包含：config_resolved.yaml、environment.json/txt、access_audit.json、stdout.log、stderr.log、timing.json、model/checkpoint、metadata、generator_provenance、四池manifest、quality.json、artifacts_manifest.json、run_manifest.json和最终status。失败/blocked也必须有结构化记录，禁止只有聊天输出。

状态至少区分：planned、preflight_passed、running、pilot_passed、quality_blocked、blocked_by_gpu、failed。聚合器默认只接受formal且成功的run；pilot永远不能自动升级为formal。

七、GPU与资源预检

在任何CTGAN/TVAE训练前实际记录：

1. Python可执行文件；
2. torch版本、torch.version.cuda、cuDNN版本；
3. torch.cuda.is_available()；
4. GPU名称、显存总量和可用量；
5. SDV是否把enable_gpu正确传给当前版本后端；
6. 一个不写正式run的小型张量GPU运算是否成功；
7. Adult全量300 epochs的时间和磁盘预算估计，并写明估计依据；
8. 当前磁盘剩余空间。

如果任一GPU硬条件失败：

- 不训练CTGAN/TVAE pilot；
- 输出blocked_by_gpu记录；
- 给出建议的新隔离conda环境名、与本机驱动兼容的PyTorch安装命令、验证命令、预计下载/磁盘量和回退方案；
- 只提供方案，不在本轮擅自安装或修改现有tabpollution环境。

GaussianCopula不受GPU阻塞，应继续完成。

八、正式C2运行器加固

1. 所有SDV/SDMetrics继续延迟导入，基础测试环境不要求安装SDV；
2. 核对SDV 1.37.3官方可用加载接口，在不破坏兼容性的前提下优先迁移到utils.load_synthesizer；若保留旧接口，必须封装兼容分支并测试；
3. 捕获并保存stdout/stderr，训练失败时也必须落盘；
4. artifacts_manifest记录每个产物的相对路径、类型、行数/大小、SHA-256和来源run；
5. 启动前先估算四池行数、模型与中间产物大小；磁盘不足时在训练前失败；
6. 对run目录使用原子状态或完成标志，半成品不得被成功聚合器读取；
7. fit仍只能读取冻结split manifest筛出的R_source_train；
8. 每个池仍必须独立sample，sample seed由generator_seed加固定offset 101/202/303/404派生；
9. Adult pilot四池规模沿用已经验证的8,051/5,367/8,051/32,201；
10. 所有provenance字段在模型特征入口处自动删除，target保留。

九、生成质量闸门

把质量检查分成“硬失败”和“诊断警告”，不要为了让结果好看临时改阈值。

硬失败至少包括：

1. 访问审计失败或读取了非R_source_train分区；
2. schema/列顺序不一致、数值解析失败、目标出现非法取值；
3. 四池synth_row_id不唯一或用途池ID相交；
4. 保存模型无法重新加载和采样；
5. 合成目标类别集合缺少真实训练集中的任一目标类别，即TVAE smoke中出现的单类坍缩；
6. TSTR流程因单类、NaN或预处理泄漏无法执行；
7. 池文件/manifest哈希不一致；
8. run日志或关键manifest缺失。

诊断警告至少报告但不凭单一阈值擅自否决：

- SDMetrics overall、Column Shapes、Column Pair Trends；
- 每池重复率、与真实精确重合率、池间内容重合；
- 合成/真实目标比例偏差；
- TSTR相对TRTR差距；
- 缺失率、整数/小数格式和字符串格式偏差；
- 训练时间、采样时间、模型大小和峰值内存/显存。

如果出现硬失败，保留全部产物和异常，将status写成quality_blocked，不得进入后续正式矩阵。

十、C3正式存储加固（本轮只实现和测试，不生成正式bags）

上一轮84个污染CSV约838 MB。本轮将正式默认改为manifest-first：

1. contamination manifest保存真实/合成成员ID、顺序或可确定重建所需的seed与来源池，不默认保存完整特征CSV；
2. bags继续保存成员manifest，按bag_id从冻结真实表和合成池重建；
3. 提供显式--materialize选项，仅在用户要求时物化某个污染集或bag；
4. 实现mixing build、bags build、bags inspect/rebuild、protocol validate等清晰CLI；现有mixing smoke可以保留兼容；
5. 重建后必须输出行数、来源组成、实际比例、成员/内容SHA-256；
6. 对同一配置重复构建manifest，成员和顺序哈希必须一致；
7. 不删除、不迁移、不改写现有C3 smoke CSV。

本轮不要生成正式50 calibration+100 test bags；只用小fixture和已有smoke manifest证明新接口正确。

十一、Adult seed=2026 pilot执行

先完成全部加固和测试，再执行：

1. GaussianCopula：完整29,273行R_source_train，generator_seed=2026，独立生成四池，运行全部质量闸门并重载模型；
2. CTGAN：只有GPU预检通过时，完整29,273行、300 epochs、generator_seed=2026；
3. TVAE：只有GPU预检通过时，完整29,273行、300 epochs、generator_seed=2026；
4. 每个pilot独立run，不能共享模型或复用seed=42合成CSV；
5. 一个pilot成功后立即生成质量卡和验收记录；
6. 不构造正式比例bags，不跑任何检测算法；
7. 三个pilot无论成功、quality_blocked或blocked_by_gpu都要进入本轮状态表。

十二、必须新增并运行的测试

保留现有59项并至少新增覆盖：

1. smoke/pilot/formal配置与run_type不能混淆；
2. pilot seed必须为2026且不得进入formal聚合；
3. stdout/stderr、timing和artifact manifest在成功/失败时均落盘；
4. 半成品run不会被成功聚合；
5. artifacts manifest哈希与文件一致；
6. GPU预检失败时CTGAN/TVAE不会调用fit；
7. 单类目标合成数据稳定触发quality_blocked；
8. 合法双类目标可以完成TSTR；
9. 非法目标、schema漂移、池ID交叉、访问泄漏均失败；
10. SDV模型新旧load兼容分支（可用fake adapter测试，常规pytest不重训）；
11. manifest-first污染构造与旧smoke成员结果一致；
12. 未materialize时不产生完整污染CSV；
13. 指定单个污染集/bag可重建且哈希稳定；
14. 磁盘预检不足时在训练前失败；
15. C0–C3全部既有哈希继续不变；
16. GaussianCopula seed=2026 pilot真实端到端产物验收。

昂贵CTGAN/TVAE训练不能放进常规pytest。若GPU不可用，blocked路径必须有真实集成验收。

十三、CLI最低覆盖

实际命名可以合理调整，但至少提供等价能力：

1. generator preflight --config <pilot-config>；
2. generator pilot --generator <name> --dataset adult --seed 2026；
3. generator validate --run-id <run-id>；
4. mixing build --manifest-only；
5. bags build --manifest-only；
6. bags inspect/rebuild --bag-id <id>；
7. protocol validate；
8. runs validate/aggregate，并证明默认排除smoke、pilot、failed和blocked。

README必须给出本机真实成功命令，不要只写计划命令。

十四、阶段闸门

只有同时满足以下条件，本轮才可标记complete：

1. 原59项测试继续通过，新增测试全部通过；
2. C0–C3既有数据、split、legacy、smoke产物哈希不变；
3. formal/pilot配置分层与run聚合隔离通过；
4. 成功/失败日志与artifact manifest机制通过；
5. GPU预检给出真实结论，未假定RTX 4070等于PyTorch CUDA可用；
6. GaussianCopula Adult seed=2026完整pilot成功并通过质量硬闸门；
7. CTGAN/TVAE要么完成符合完整R_source_train+300 epochs的pilot，要么被如实标记blocked_by_gpu；
8. manifest-first C3接口可重建且默认不物化大CSV；
9. 没有启动Credit、2027–2030、正式bags、C4或其他任务线算法；
10. 唯一总run manifest和本轮完成报告已生成。

十五、本轮交付物

1. 正式/pilot配置、schema和resolved配置；
2. GPU/资源预检模块与报告；
3. 加固后的generator runner、日志和artifact manifest；
4. 生成质量硬闸门与质量卡；
5. manifest-first污染/bag构造及重建CLI；
6. Adult seed=2026 GaussianCopula pilot完整产物；
7. CTGAN/TVAE pilot产物或blocked_by_gpu结构化记录；
8. 新增单元/集成测试与JUnit报告；
9. reproduction_status或等价状态表；
10. 唯一总run manifest；
11. reports/c2_formal_readiness_pilot_completion_report.md；
12. 更新README。

十六、最终汇报格式

完成后停止，不进入C4。最终汇报必须基于实际文件和实际运行结果，至少包括：

- 阶段：C2正式启动加固与单seed试运行；
- 完成、blocked、failed和未执行内容；
- 新增/修改文件；
- 回归测试与最终测试数量、耗时；
- C0–C3哈希回归结果；
- 实际Python/SDV/SDMetrics/torch/CUDA/GPU/磁盘环境；
- GPU预检的每一项结果；
- 三生成器各自状态与run_id；
- 每个实际执行pilot的fit scope、epochs、参数、训练/采样时间、模型大小；
- 四池行数、sample seed、SHA-256、ID隔离和访问审计；
- 质量硬闸门、SDMetrics、重复/重合、目标比例、TRTR/TSTR；
- manifest-first存储验证、重建示例和节省空间估计；
- 日志、artifact manifest和聚合隔离验证；
- 与冻结规格的差异及原因；
- 风险、技术债和blocked项；
- 是否满足启动完整“两表×三生成器×5 seeds”正式C2的条件；
- 下一阶段建议，但不要自行执行。

不要用计划值替代实际值，不要把pilot写成formal，不要因CTGAN/TVAE受GPU阻塞而伪造训练，也不要删除失败记录。
```
