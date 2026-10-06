# Tabular Synthetic Data Governance

合成表格数据治理的研究代码、冻结实验及研究设计。Python 包名为 `tabpollution`。

当前重点：**合成数据的哪些偏差会影响下游学习，来源检测驱动的清理又在什么条件下改善或损害模型？** 检测与比例估计是解释治理决策的工具，不再各自作为独立的研究终点。

## 当前状态

- 2026-10-06：E0 历史数据核验及独立重分析已完成，见[执行报告](reproduction/reports/e0_20261006/E0_核验与重分析报告.md)。修正追加低比例漏选及效用去重，明确清理失败覆盖率；未开展新训练。
- 历史正式矩阵：`governance-formal-v2-calibration`，5 个种子、110 个分片；完成记录保留在 `reproduction/reports/formal_analysis/`。
- 2026-10-02 科研审计发现：工程完整性不等于结论有效性。旧汇总的低比例筛选、效用统计条件、对照设计及复现忠实度存在需要处理的问题；旧报告不再作为未经限定的最终论文结论。
- E1 配对筛查已冻结首批配置，采用 Adult/Credit/Abalone、CTGAN/TVAE、真实数据留出测试和单进程 CPU 后台运行。它是探索性试验，当前状态以服务器 `job-status.txt` 和各分片 `complete.json` 为准；不能将固定合成池的拆分种子称为独立生成器重复。

## 从这里阅读

1. [全面审视与证据边界](reproduction/docs/research/2026-10-02_全面审视与证据边界.md)：哪些成果可以保留，哪些说法应收回，问题定位及修复优先级。
2. [最新研究与 benchmark 对照](reproduction/docs/research/2026-10-02_研究检索与基准对照.md)：一手来源、发表状态、相邻工作、新颖性风险。
3. [聚焦探索计划](reproduction/docs/research/2026-10-02_聚焦探索计划.md)：研究问题、控制变量、实验分阶段设计和停止条件。
4. [仓库边界与同步说明](reproduction/docs/repository_layout.md)：本机、远程与服务器各保存什么。
5. [历史框架运行说明](reproduction/README_governance.md)：旧版 CLI、配置和运行约束。

## 代码布局

```text
reproduction/                 # 保留路径兼容历史配置和服务器软链接
  pyproject.toml              # 可安装的 tabpollution 包
  src/tabpollution/           # 数据、生成、检测、量化、估值、治理、分析
  configs/                   # smoke / pilot / frozen formal 配置
  tests/                     # 单元测试及集成测试
  scripts/                   # 数据准备、分片执行及分析入口
  docs/research/             # 当前审计和下一阶段研究设计
  manifests/                 # 可追溯的实验登记
  reports/                   # 小型历史审计及结果；不是独立样本数据库
  legacy/                    # 早期验证材料，不作为正式结果
  data/ runs/ outputs/ checkpoints/  # 本地或服务器生成，Git 忽略
```

## 最小验证（CPU，不下载数据）

使用独立环境，Python 3.11 或以上；不要覆盖共享服务器的系统环境或 CUDA 安装。

```bash
cd reproduction
python -m pip install -e ".[test]"
python -m pytest tests/unit/test_governance_mixtures.py tests/unit/test_governance_metrics.py tests/unit/test_research_semantics.py -q
python -m tabpollution governance preflight --config configs/governance_smoke.yaml
```

正式训练不是默认快速开始步骤。共享服务器必须先确认资源配额和设备，不自动占用空闲卡。迁移评测口径必须创建新版本，不可覆盖旧 run。

## 材料与开放边界

文献 PDF、学位论文草稿、实习记录和私人服务器交接文档不属于代码仓库发布内容；整理只取消其 Git 跟踪，本地已有材料保留。历史 Git 提交不改写，因此旧材料仍可能在 Git 历史中找到。大数据及逐样本结果通过独立存储管理。

目前这是**具有基准要素的研究实验框架**，尚不能仅凭目录整齐或分片完成宣称成为成熟公共 benchmark。发布要求见探索计划。第三方算法与数据各自许可仍须核验，不能把本仓库公开等同于取得全部再分发授权。
