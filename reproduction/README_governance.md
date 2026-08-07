# 合成表格数据污染治理评测基准

本项目的中心问题是：**样本级检测误差如何传导为语料级污染率误差，并进一步影响下游治理决策？**

项目不再把“检测、比例估计、效用/估值”视为三组互不相关的实验，而是在相同的数据划分、检测得分和污染袋上形成连续证据链：

```text
Record-level detectability
        ↓ 冻结检测器得分、校准、FPR/TPR
Corpus-level prevalence
        ↓ 污染率估计误差、低污染率假阳性放大
Task-level consequence
        ↓ 保留/清理决策、纯真实测试效用与决策遗憾
Governance conclusion
```

## 快速开始

在 `reproduction` 目录执行：

```powershell
python -m pip install -e ".[test]"
python -m tabpollution governance preflight --config configs/governance_smoke.yaml
python -m tabpollution governance run --config configs/governance_smoke.yaml
```

本地smoke使用确定性的三表、三生成机制夹具，不下载数据、不调用GPU，覆盖P1--P4及5%/10%低污染率。结果写入：

```text
runs/governance-smoke-v1/
├── governance_evidence.csv
├── format_artifact_audit.csv
├── finding_1_transfer.csv
├── finding_2_format_artifacts.csv
├── finding_3_low_prevalence.csv
├── finding_4_quantifier_shift.csv
├── finding_5_utility_curve.csv
├── finding_6_detectability_vs_harm.csv
├── protocol_manifests.json
├── resolved_config.json
└── summary.json
```

`run_type: smoke` 的结果只能用于验证工程闭环，不进入论文正式汇总。

## 正式运行入口

正式配置为 `configs/governance_formal.yaml`。正式运行前必须准备：

1. Adult、Abalone和Credit的冻结真实数据；其中Abalone以冻结阈值 `rings > 9` 定义二分类任务，避免在评测结果产生后调整标签；
2. CTGAN和TVAE独立合成池；P3/P4使用Adult与Abalone作为训练表，使table adaptation具有可识别的表域信号；
3. `data/governance/pool_registry.csv`；
4. 服务器端CUDA环境和共享GPU使用确认。

服务器首次准备正式环境时安装项目扩展：

```bash
python -m pip install -e ".[formal,test]"
```

CUDA版PyTorch沿用课题组已经验证的独立安装，不由本项目重新覆盖。
生成器依赖冻结为既有pilot已验证的SDV 1.37.3与SDMetrics 0.28.0；正式矩阵中途不得升级。XGBoost及其他运行时版本会写入环境清单，单次正式矩阵内必须保持不变。

注册表格式参考 `configs/governance_pool_registry.example.csv`。路径相对于注册表文件解析，因此本地和服务器可使用相同配置，不应把绝对机器路径写入实验配置。正式合成池由以下入口构建：

```bash
python -m tabpollution governance source-prepare --config configs/governance_pool_build.yaml
python -m tabpollution governance pool-preflight --config configs/governance_pool_build.yaml
CONFIRM_SHARED_GPU=1 bash scripts/run_governance_pool_build.sh
```

`source-prepare`从UCI官方地址下载并清洗三张真实表，采用临时文件替换避免留下半成品，并记录原始文件与规范表的SHA-256。构建器先冻结互斥的 `source_train` 与评测真实池：CTGAN/TVAE只拟合前者，后者才进入检测、计量和下游纯真实测试，从而杜绝生成器见到评测记录。两部分及合成CSV、模型权重、目标变换的SHA-256均写入个人NFS；全部组合完成后才原子生成 `pool_registry.csv`。`--resume`只复用同时存在CSV和模型的完整组合，不覆盖已完成产物。

```bash
python -m tabpollution governance preflight --config configs/governance_formal.yaml
CONFIRM_SHARED_GPU=1 bash scripts/run_governance_pilot.sh
CONFIRM_SHARED_GPU=1 bash scripts/run_governance_formal.sh
```

必须先完成与正式矩阵使用相同协议、算法和数据池的单种子pilot；pilot成功后才启动正式脚本。正式配置采用5个冻结种子、7档污染率和每档100个比例估计bags；下游效用只在每档前5个bags上运行，以避免把量化所需重复数机械放大为不可承受的模型训练次数。两个脚本均保存实际Python与依赖环境清单。

## P1--P4

| 协议 | 训练与测试关系 | 解释 |
|---|---|---|
| P1 | 同表、同生成器、记录隔离 | 同分布上限，不代表可部署性 |
| P2 | 同表、生成器隔离 | 未知生成器迁移 |
| P3 | 表结构隔离、生成器可相同 | 未知schema迁移 |
| P4 | 表结构与生成器均隔离 | 最严格的完全迁移 |

固定schema的C2ST-LR/XGBoost只参加P1/P2；字符序列方法和DWT/DWTA可参加P1--P4。比例估计覆盖CC、PCC、ACC、PACC、EMQ、HDy、DyS、Median Sweep和连续密度KDEy。协议验证器会拒绝记录、表或生成器泄漏。

## 结果解释边界

- `artifact_gate_passed=false` 表示格式变量本身具备明显来源可分性，相应检测结果必须附带伪影警告。
- AUROC反映排序，不等价于低污染率下的数量准确性；必须同时报告FPR、校准误差、MAE和Bias。
- 合成来源标签不等价于负任务价值；下游实验使用纯真实测试集直接测量保留和清理后的效用。
- `formal_inclusion=true` 只由正式配置生成；smoke、pilot和runthrough不得写入正式统计表。

进一步说明见：

- `docs/研究定位与实验主线.md`
- `docs/统一治理基准实验契约.md`
- `docs/服务器运行手册.md`
