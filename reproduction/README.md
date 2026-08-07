# 复现实验代码与结果

> 说明：根目录的 `run_minimal_loop.py`、`run_adult_baseline.py` 与原有 `outputs/` 是阶段一 legacy smoke，保留用于历史追踪，不属于 benchmark v1 正式结果。正式基准基础设施位于 `src/tabpollution`、`configs`、`data`、`manifests`、`tests` 和 `reports`。

这个文件夹集中存放已经完成的复现工作，避免把脚本、依赖和实验输出散落在毕设根目录。

## 内容

- `run_minimal_loop.py`：第 0 步最小闭环，Adult + CTGAN + 分布图 + Logistic Regression 检测器；
- `run_adult_baseline.py`：阶段一 Adult 基线复现，SDMetrics + Logistic Regression AUROC；
- `requirements-minimal.txt`：当前复现实验所需最小依赖；
- `requirements.txt`：说明书中的完整依赖清单；
- `README_第0步环境准备.md`：第 0 步运行说明；
- `README_阶段一基线复现.md`：Adult 基线复现说明；
- `outputs/`：已经跑出的实验结果。

## 使用方式

从毕设根目录进入本文件夹：

```powershell
cd reproduction
conda activate tabpollution
```

运行第 0 步最小闭环：

```powershell
python run_minimal_loop.py --train-rows 3000 --sample-rows 1000 --epochs 20
```

运行阶段一 Adult 基线复现：

```powershell
python run_adult_baseline.py --train-rows 3000 --sample-rows 1000 --epochs 20 --output-dir outputs/adult_baseline
```

复用已有第 0 步数据评估基线：

```powershell
python run_adult_baseline.py `
  --real-csv outputs/week0/adult_real_sample.csv `
  --synthetic-csv outputs/week0/adult_synthetic_ctgan.csv `
  --output-dir outputs/adult_baseline_reuse
```

## Benchmark v1：C0—C1 基础设施

在 `reproduction` 目录使用 Python 3.11+：

```powershell
python -m pip install -e .
python -m tabpollution.cli environment capture --output reports/environment_initial.txt
python -m tabpollution.cli data prepare --config configs/benchmark_v1.yaml
python -m tabpollution.cli data validate --benchmark benchmark_v1
python -m pytest
```

如果当前 Python 的全局/用户包目录不可写，可把轻量 XLS 读取依赖放在项目内：

```powershell
python -m pip install --target .deps xlrd==2.0.1
$env:PYTHONPATH=((Resolve-Path 'src').Path + ';' + (Resolve-Path '.deps').Path)
python -m tabpollution.cli data prepare --config configs/benchmark_v1.yaml
```

数据准备命令只下载并清洗 UCI Adult 和 Default of Credit Card Clients，生成稳定 `row_id`、5个正式随机种子的 60/15/10/15 冻结划分以及数据卡。本阶段不训练生成器或检测模型。

关键约束：

- `R_source_train` 仅供后续生成器拟合和下游真实训练基底使用；
- `R_detector_train`、`R_detector_val`、`R_final_test` 严格隔离；
- `row_id`、`split`、来源信息不得成为模型特征；
- 已有 run_id 和历史输出不得静默覆盖；
- seed=42 只用于 smoke，正式种子为 2026—2030。

## Benchmark v1：C2–C3 Adult smoke

本阶段只验证生成器、隔离合成池、污染构造和比例 bag 的工程闭环，不是正式实验结果。真实分区固定读取 `split_seed=2026`，生成器使用 `generator_seed=42`；GaussianCopula 拟合完整 `R_source_train`，CTGAN/TVAE 在 CPU 上各拟合确定性的 3,000 行 smoke 子集、20 epochs。

在 `reproduction` 目录运行：

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
$python='D:\Anaconda3\envs\tabpollution\python.exe'

# 单个生成器的一次性 C2 smoke（已有同名 run 时会拒绝覆盖）
& $python -m tabpollution.cli generator smoke --generator GaussianCopula --project-root .
& $python -m tabpollution.cli generator smoke --generator CTGAN --project-root .
& $python -m tabpollution.cli generator smoke --generator TVAE --project-root .

# 校验三个已保存模型可重新加载并采样
& $python -m tabpollution.cli generator validate --project-root .

# C3 复合命令：构造四类污染条件、smoke bags 和 P1–P5 验证结果
& $python -m tabpollution.cli mixing smoke --generator GaussianCopula --project-root .
& $python -m tabpollution.cli mixing smoke --generator CTGAN --project-root .
& $python -m tabpollution.cli mixing smoke --generator TVAE --project-root .

# 按 bag_id 检查可重建成员、来源组成和 SHA-256
python -m tabpollution.cli bags inspect --generator GaussianCopula `
  --bag-id 'bag:adult:GaussianCopula:s42:test:p050:b00' --project-root .

# 不重新训练生成器的完整回归测试
python -m pytest -q --junitxml=reports/pytest_c2_c3_smoke.xml
```

成功产物位于 `runs/c2-smoke-adult-*-s42-a2` 与 `runs/c3-smoke-adult-*-s42`。失败的首次尝试保留在原 run 目录并含 `failure.json`，不会参与结果聚合。正式 5-seed、300-epoch、Credit、B-small 和 C4 均未在本阶段执行。

## C2 正式启动加固与 seed=2026 pilot

本阶段仍不属于正式结果。它使用正式配置但明确标记 `run_type=pilot`，聚合器会排除所有 pilot、smoke、failed 和 blocked run。

```powershell
cd E:\毕设\reproduction
$env:PYTHONPATH=(Resolve-Path 'src').Path
$python='D:\Anaconda3\envs\tabpollution\python.exe'

# 实际GPU、SDV构造函数、磁盘和时间预算预检
& $python -m tabpollution.cli generator preflight `
  --config configs/pilot_c2.yaml --project-root . `
  --output reports/c2_pilot_generator_preflight.json

# 正式配置、Adult、单seed pilot；GPU不通过时CTGAN/TVAE只写blocked记录，不调用fit
& $python -m tabpollution.cli generator pilot --generator GaussianCopula `
  --dataset adult --seed 2026 --config configs/pilot_c2.yaml --project-root .
& $python -m tabpollution.cli generator pilot --generator CTGAN `
  --dataset adult --seed 2026 --config configs/pilot_c2.yaml --project-root .
& $python -m tabpollution.cli generator pilot --generator TVAE `
  --dataset adult --seed 2026 --config configs/pilot_c2.yaml --project-root .

# manifest-first污染fixture：默认不写完整特征CSV
python -m tabpollution.cli mixing build --generator GaussianCopula `
  --condition synthetic_replace --proportion 0.25 `
  --output reports/manifest_first_contamination_fixture.json --manifest-only --project-root .

# 构建成员manifest并重建一个bag
python -m tabpollution.cli bags build --generator GaussianCopula --stage test `
  --proportion 0.5 --bag-index 0 --output reports/manifest_first_bag_fixture.json `
  --manifest-only --project-root .
python -m tabpollution.cli bags rebuild --generator GaussianCopula `
  --manifest reports/manifest_first_bag_fixture.json --project-root .

# 证明pilot不会进入正式汇总
python -m tabpollution.cli runs aggregate --runs-dir runs

python -m pytest -q --junitxml=reports/pytest_c2_formal_readiness_pilot.xml
```

本机实际结果：GPU在操作系统中可见，但生成器环境的torch为CPU版，因此CTGAN/TVAE均为`blocked_by_gpu`且`fit_called=false`。GaussianCopula最终有效pilot为`c2-pilot-adult-gaussiancopula-s2026-20260714T181119905473Z`。适配器会在每次采样前把`sample_seed`传给SDV内部随机状态，并用跨run内容哈希检查防止不同seed产生相同池。

## 算法跑通交差版（第四轮）

本轨道只证明经典方法、直接论文基线和SOTA核心可以完成训练、推理、评测与保存闭环。所有run均为`run_type=runthrough`，不会进入formal聚合。

```powershell
cd E:\毕设\reproduction
$env:PYTHONPATH=(Resolve-Path 'src').Path
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$python='D:\Anaconda3\envs\tabpollution\python.exe'

# 完整算法runthrough（已有结果无需重复执行）
& $python -m tabpollution.cli runthrough all --project-root .

# 若只需从已保存成功run重建报告，不重新训练
& $python -m tabpollution.cli runthrough aggregate --project-root .

# 最终回归测试
python -m pytest -q --junitxml=reports/pytest_algorithm_runthrough_final.xml
```

实际完成矩阵包括D-LR、D-XGB、D-3G、D-FT、D-TT、D-DW、D-DWTA，八个经典比例估计器，LR/XGBoost效用曲线、KNN-Shapley和Data-OOB。三表mini跨表轨道使用Adult、Credit和官方UCI Abalone；深度模型是CPU tiny论文对齐自实现，不是论文性能复现。完成报告见`reports/algorithm_runthrough_completion_report.md`。
