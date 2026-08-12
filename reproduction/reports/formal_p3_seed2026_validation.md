# Seed 2026 P3 正式分片验收报告

## 阶段结论

截至 2026-08-12，seed 2026 的 P3 五个合法检测器分片全部完成，正式矩阵进度由 12/110 推进至 17/110，剩余 93 个分片。五个分片均原子写入 `COMPLETE.json`，每个分片包含 35,100 行证据，重复“校准策略—量化器—bag”键为 0，每个“校准策略×量化器”组合均包含 1,300 行，误差分解最大绝对残差不超过 $2.23\times10^{-16}$。

P3 隔离了跨表迁移：训练表与测试表不相交，但生成器类型保持已见。固定 schema 的 C2ST 不参加 P3，因此本阶段包括 Char 3-gram、Flat Transformer、Column-positional Ablation、Datum-wise Transformer 和 Datum-wise+TA。

## 分片结果

| 检测器 | 用时/min | 目标 AUROC | source PACC MAE | anchor PACC MAE | oracle PACC MAE | 正式纳入 |
|---|---:|---:|---:|---:|---:|---|
| Char 3-gram | 12.8 | 0.6216 | 0.6692 | 0.6692 | 0.0509 | 是 |
| Flat Transformer | 29.9 | 0.6976 | 0.6692 | 0.6692 | 不可定义 | 否 |
| Column-positional Ablation | 40.0 | 0.5725 | 0.6692 | 0.6692 | 不可定义 | 否 |
| Datum-wise Transformer | 41.8 | 0.5474 | 0.6692 | 0.6692 | 0.0497 | 是 |
| Datum-wise+TA | 34.9 | 0.6386 | 0.2293 | 0.2293 | 0.0387 | 是 |

表中 MAE 仅用于单种子运行验收和提出后续检验假设，不构成最终效果或显著性结论。正式结论必须等待五个种子完成后，使用相同 bag 的配对检验、置信区间和 Holm 多重比较校正。

## 正式纳入边界

所有五个分片都在工程意义上完整运行并保存证据，但 Flat Transformer 与 Column-positional Ablation 为 `analysis_ready=false`、`formal_inclusion=false`。两者均在 `oracle_target` 校准下出现 1,300 行 PACC `failed:unstable_denominator`；同一策略下 ACC、KDEy 和 Median Sweep 也分别出现预注册的不可定义状态。其 source-only 与 target-real-anchor 策略均正常运行。

诊断表明，两种检测器在源验证集上的 AUROC 分别为 0.9163 和 0.9355，但跨表目标验证集的 oracle Platt 校准退化为常数 0.5，使 PACC 所需的类条件概率差为零。该结果符合预注册实现与数学定义，不是训练崩溃或数据管线错误。不得在观察正式结果后增加回退规则、改变主量化器或删除失败组合。后续汇总应将其报告为“跨表条件下主量化器不可定义”，并与仍可计算的部署策略结果并列展示。

其余三个正式纳入分片均仅在 `oracle_target × median_sweep` 出现 1,300 行 `failed:no_valid_median_sweep_threshold`，主量化器 PACC 正常，因此不影响正式纳入。

## 初步研究现象

普通 Char 3-gram、Flat、Column 和 Datum-wise 在 source-only PACC 下的 MAE 均约为 0.6692，目标真实负例锚点没有改变 PACC 的概率校正参数，故 anchor PACC MAE 与 source 相同。oracle 目标标签校准在可定义的方法上将 MAE 降至约 0.04--0.05，说明跨表条件下的主要误差不仅来自排序能力下降，还来自类别条件分数分布和概率尺度的迁移。

Datum-wise+TA 相比普通 Datum-wise 将目标 AUROC 从 0.5474 提升到 0.6386，并将 source PACC MAE 从 0.6692 降至 0.2293；其 oracle PACC MAE 为 0.0387。这个单种子结果支持一个待验证假设：表域对抗适配能够缓解表示层面的跨表偏移及其向污染率估计的传导，但不能完全替代目标域校准。该假设应在其余四个种子及 P4 跨表跨生成器条件下继续检验。

## 运行恢复与资源记录

首批四分片顺序运行期间，本地远程控制会话被中断。前三个已完成分片的原子标记和证据未受影响；当时正在运行的 Datum-wise 尝试目录为空且没有发布完成标记。随后使用 `--resume` 从完成边界单独重跑 Datum-wise，并成功完成。该事件属于控制会话中断，不属于算法失败，也没有产生重复正式证据。

五个分片始终采用单进程顺序运行，没有 GPU 并行。Datum-wise+TA 首次正式分片单独运行并通过验收；结束后 GPU 为 17 MiB、0% 利用率且无计算进程。后续可按相同策略推进 seed 2026 的 P4：先运行四个已验证结构，再单独运行 Datum-wise+TA。
