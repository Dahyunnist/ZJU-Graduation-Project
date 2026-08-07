# Cross-table Synthetic Tabular Data Detection（中文梳理）

## 文献信息

Kindji G. C. N., Rojas Barahona L. M., Fromont E., Urvoy T. GenAIDetect@COLING 2025，78–84。原文：[PDF](../论文PDF/01_Kindji2025_Cross_table_Detection.pdf)；[官方页面](https://aclanthology.org/2025.genaidetect-1.5/)。

## 研究问题与地位

论文研究“检测器能否在表结构、领域或生成器发生变化时识别单条合成表格记录”。它不是普通的同表C2ST，而是目前与本课题“开放环境下合成表格污染检测”最直接的工作，也是后续IDA 2025和AAAI 2026工作的起点。

## 方法

作者提出三条可跨表的基线：**字符3-gram特征加Logistic Regression、把一行序列化为文本后送入Transformer、面向表格的Transformer表示**。实验使用多个真实表，并由CTGAN、TVAE、TabDDPM和TabSyn等生成器制造合成记录。四类协议逐步增加难度：训练测试条件一致、跨生成器、跨表以及生成器与表同时变化。

## 主要结论

同表或见过表结构时，真实/合成往往容易区分；一旦测试表和生成器未见，性能明显下降。因而高同表AUROC不能证明检测器可部署，跨表协议才是课题价值所在。

## 局限与本课题用法

工作属于短论文，基线和实验规模仍初步，未解决任意schema的稳健编码。本课题应完整复现三条基线和四类协议，并扩充数据集、生成器、随机种子和校准指标。它们属于第一优先级对比算法。

## 精读检查点

- 核对一行记录如何序列化、缺失值和数值列如何处理。
- 画清训练表、测试表、训练生成器、测试生成器的组合，防止协议泄漏。
- 记录每条基线输入表示与参数量，而不能只报告模型名字。

