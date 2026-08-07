# AI Models Collapse When Trained on Recursively Generated Data（中文梳理）

## 文献信息

Shumailov I., Shumaylov Z., Zhao Y., et al. Nature，2024，631：755–759。原文：[PDF](../论文PDF/05_Shumailov2024_Model_Collapse.pdf)；[DOI](https://doi.org/10.1038/s41586-024-07566-y)。

## 研究问题与结论

论文研究模型反复使用前代模型生成数据训练时会发生什么。理论与GMM、VAE、语言模型实验表明，递归生成训练会优先丢失原分布尾部，误差累积并最终产生“model collapse”。真实人类数据因而在合成内容增多时更有价值。

## 与本课题关系

该文提供“合成数据污染可能影响下游模型和数据价值”的强动机，也提示不能只观察平均准确率：应加入少数类召回、尾部分组性能、校准和多代累积影响。

## 重要边界

它不是表格记录检测算法，也不是Data Shapley式估值方法；递归替换训练与一次性把少量CTGAN记录混入Adult并非同一情景。因此不能把它列入样本级对比算法表。

## 可选复现

若工期允许，做一个低成本GMM或表格生成器多代递归示意，比较replacement与保留真实数据的accumulation。主线仍应是单代不同污染率下的下游效用和数据估值。

