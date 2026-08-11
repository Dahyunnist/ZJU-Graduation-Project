# Seed 2026 P1--P2 正式分片阶段验收报告

## 阶段结论

截至 2026-08-11，seed 2026 的 P1 与 P2 共 12 个正式分片全部完成。当前总进度为 12/110，剩余 98 个分片。P1 六个检测器分片和 P2 六个检测器分片均写入 `COMPLETE.json`，所有分片均为 `formal_inclusion=true`。

本轮连续执行了七个新分片：P1 Datum-wise；P2 Char 3-gram、C2ST-LR、C2ST-XGBoost；P2 Flat Transformer、Column-positional Ablation、Datum-wise Transformer。所有分片顺序执行，没有并行占用 GPU。

## 分片概览

| 分片 | 用时 | 目标 AUROC | 平均污染率 MAE | 非主方法失败 |
|---|---:|---:|---:|---:|
| P1 Datum-wise | 37.3 min | 0.8387 | 0.0548 | 0 |
| P2 Char 3-gram | 12.5 min | 0.7248 | 0.1209 | 0 |
| P2 C2ST-LR | 7.5 min | 0.3768（source）；0.6232（oracle） | 0.2638 | 1,300 |
| P2 C2ST-XGBoost | 7.4 min | 0.7403 | 0.1021 | 0 |
| P2 Flat Transformer | 29.9 min | 0.7146 | 0.1067 | 0 |
| P2 Column-positional Ablation | 40.3 min | 0.7907 | 0.0930 | 0 |
| P2 Datum-wise | 34.8 min | 0.7289 | 0.0945 | 0 |

每个分片均生成 35,100 行正式证据；重复 bag-量化器键均为 0，每个预注册组合均恰有 100 个 bag，误差分解残差处于 1e-16 浮点精度。除下述 Median Sweep 明确不适用外，污染率估计均为有限值。

## Median Sweep 不适用记录

P2 C2ST-LR 的 `oracle_target × median_sweep` 共 1,300 个 bag 返回 `failed:no_valid_median_sweep_threshold`。其余 26 个“校准策略×量化器”组合全部正常，主量化器 PACC 没有失败。

该现象源于 C2ST-LR 在未见生成器上发生检测方向反转：source 策略目标 AUROC 为 0.3768，带目标标签的 oracle 校准识别并翻转方向后 AUROC 为 0.6232；翻转后的 oracle 校准分布不存在满足 Median Sweep 稳定条件的合法阈值。这是预注册算法在特定分数分布下的可定义性失败，不是管线异常。正式契约允许非主量化器保留失败状态，故该分片 `formal_inclusion=true`。不得在观察正式结果后添加回退规则或删除该失败组合。

## 跨生成器初步现象

P2 相比 P1 引入未见生成器。多种检测器均表现出 oracle 校准显著降低污染率误差的现象：

- Char 3-gram：source MAE 0.1496，oracle MAE 0.0638；
- C2ST-XGBoost：source MAE 0.1259，oracle MAE 0.0541；
- Flat Transformer：source MAE 0.1296，oracle MAE 0.0610；
- Column Ablation：source MAE 0.1134，oracle MAE 0.0519；
- Datum-wise：source MAE 0.1075，oracle MAE 0.0687。

目标真实负例锚点可以迁移 FPR 阈值，但在 P2 中通常不能恢复完整概率校准或类条件分数分布，因此污染率 MAE 仍接近 source 策略。这与中心研究问题一致：阈值迁移、概率校准迁移和量化器校正迁移是三个不同层次。

上述数值仅来自 seed 2026，不能作为最终显著性结论。必须等待五个种子完成后进行同 bag 配对检验与 Holm 校正。

## 资源与后续批量策略

三个深度结构在 P1/P2 均完成正式规模训练和评价：显存约 1.8--2.3 GiB，无 OOM、得分退化或截断；训练结束后均释放 GPU。共享 GPU 监控未发现其他计算进程。

由此，已验证结构可以采用每批 2--4 个分片的单进程顺序运行，不进行 GPU 并行。下一阶段 P3 的建议批次为 Char 3-gram、Flat Transformer、Column-positional Ablation、Datum-wise Transformer 四个分片；首次出现的 Datum-wise+TA 应继续单独运行和验收。
