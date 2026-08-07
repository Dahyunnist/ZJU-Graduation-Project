# 阶段一：Adult 基线复现

本基线对应说明书 4.2 的要求：

- 用 SDMetrics 计算真实表格与合成表格的统计相似性；
- 用一个 scikit-learn 分类器判断记录是真实还是合成；
- 在 Adult 数据集上输出 AUROC，作为后续实验的地基代码。

## 1. 环境

使用前面已经创建的 Conda 环境：

```powershell
conda activate tabpollution
python -m pip install -r requirements-minimal.txt
```

`requirements-minimal.txt` 已包含本阶段需要的 `sdv`、`sdmetrics`、`scikit-learn`、`pandas` 等依赖。

## 2. 一键复现基线

```powershell
python run_adult_baseline.py --train-rows 3000 --sample-rows 1000 --epochs 20 --output-dir outputs/adult_baseline
```

脚本会自动完成：

1. 加载 UCI Adult 数据集；
2. 训练 CTGAN；
3. 生成 1000 条合成数据；
4. 运行 SDMetrics `QualityReport`；
5. 训练 Logistic Regression 判别器并输出 AUROC。

## 3. 复用已有第 0 步数据

如果已经跑过 `outputs/week0`，可以直接评估已有真实/合成数据：

```powershell
python run_adult_baseline.py `
  --real-csv outputs/week0/adult_real_sample.csv `
  --synthetic-csv outputs/week0/adult_synthetic_ctgan.csv `
  --output-dir outputs/adult_baseline_reuse
```

## 4. 输出文件

`outputs/adult_baseline` 中会生成：

- `adult_real.csv`：参与评估的真实 Adult 数据；
- `adult_synthetic_ctgan.csv`：CTGAN 合成数据；
- `sdmetrics_quality_report.pkl`：SDMetrics 完整报告；
- `sdmetrics_properties.csv`：统计指标总览；
- `sdmetrics_details_column_shapes.csv`：单列分布相似性明细；
- `sdmetrics_details_column_pair_trends.csv`：列对关系相似性明细；
- `classifier_report.txt`：分类器 Accuracy、AUROC、precision/recall/F1；
- `baseline_summary.json`：机器可读汇总；
- `baseline_summary.md`：人工阅读版实验结果。

## 5. 本次已跑通结果

使用 `--train-rows 3000 --sample-rows 1000 --epochs 20` 的完整复现实验结果：

- SDMetrics overall quality score: `0.7734`
- Column Shapes: `0.8232`
- Column Pair Trends: `0.7236`
- Logistic Regression AUROC: `0.7808`

解释：SDMetrics 分数说明 CTGAN 数据在统计上与真实数据有一定相似性；AUROC 明显高于 `0.5`，说明简单分类器仍能区分真实/合成数据，这个结果可以作为后续更强检测方法的基础对照。
