# Robust Detection of Synthetic Tabular Data Under Schema Variability（中文梳理）

## 文献信息

Kindji G. C. N., Fromont E., Rojas-Barahona L. M., Urvoy T. AAAI 2026，40(27)：22617–22625。原文：[PDF](../论文PDF/03_Kindji2026_Robust_Detection_Schema_Variability.pdf)；[AAAI官方页面](https://ojs.aaai.org/index.php/AAAI/article/view/39422)。

## 贡献

论文面向列数、类型和含义均可变化的未知schema，提出datum-wise Transformer。其关键是以“单元格/字段为单位”形成能够跨表共享的记录表示，而不是要求所有表拥有同一固定列空间；进一步加入table adaptation，在目标表上进行适配。

## 实验结论

论文报告datum-wise架构相对此前唯一公开跨表基线在AUC和Accuracy上约提升7个百分点；加入表适配后Accuracy再提高约7个百分点。结果说明跨表检测并非不可行，但目标表适配信息对稳健性很重要。

## 局限与风险

截至2026-07-14，官方页面仍只说明代码将在扩展版提供，尚未发现可直接运行的官方仓库。独立复现时最容易出错的是单元格编码、字段上下文、聚合方式和adaptation数据使用边界。若适配阶段接触了带真实/合成标签的目标表，必须与完全零样本迁移分开报告。

## 本课题安排

这是任务线一的SOTA必跟踪项。先按论文实现datum-wise主干，记录所有未明确细节；官方代码发布后进行逐模块核对和替换。基准必须同时报告无适配和有适配结果，不能把两种设定混在同一排行中。

## 精读检查点

- 重画模型结构和张量维度。
- 标出table adaptation使用哪些目标数据、是否使用标签。
- 逐表提取结果，关注宏平均、最差表与置信区间，而非只看总体平均。

