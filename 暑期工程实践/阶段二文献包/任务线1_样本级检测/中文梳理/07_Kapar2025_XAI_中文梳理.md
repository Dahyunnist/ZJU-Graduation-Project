# What’s Wrong with Your Synthetic Tabular Data?（中文梳理）

## 文献信息

Kapar J., Koenen N., Jullum M. World Conference on Explainable Artificial Intelligence（xAI 2025），CCIS 2578：19–43。原文：[PDF](../论文PDF/07_Kapar2025_XAI_Synthetic_Data_Detection.pdf)；[Springer正式版](https://link.springer.com/chapter/10.1007/978-3-032-08327-2_2)；[代码](https://github.com/bips-hb/XAI_syn_data_detection)。

## 研究问题

传统质量指标可能互相矛盾，而且只告诉研究者“有差异”，不解释差异来自哪里。论文在真实/合成检测器之上加入可解释AI，以诊断生成模型的具体缺陷。正式版于2025年10月在线发表，不再只是预印本。

## 方法

先训练真实/合成二分类器，再使用Permutation Feature Importance、Partial Dependence、SHAP和反事实解释，分析哪些字段、取值范围及依赖关系使记录被判断为合成。正式版还在11个真实数据集、6类生成器上比较LR、随机森林和调优XGBoost，XGBoost整体最强；Adult/TabSyn与Nursery/CTGAN用于深入解释。

## 主要价值

方法能发现边际指标漏掉的不合理依赖、缺失模式或局部区域，适合作为检测结果的解释层。对毕业设计而言，它能让“算法比较”超越一张分数表，回答检测器究竟抓住了真实生成缺陷还是无关格式伪影。

## 局限与复现定位

该工作已有正式会议版本，但主要目标仍是解释生成质量，而非解决未知schema迁移；XAI结论也会随分类器和特征相关性改变。它不应排在Kindji系列和C2ST-XGBoost之前；建议在主检测器跑通后，对Adult及另一个数据集做诊断性复现，并交叉比较多种解释方法。
