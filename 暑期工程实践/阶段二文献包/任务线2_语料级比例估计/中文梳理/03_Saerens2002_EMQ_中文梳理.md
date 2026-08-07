# Adjusting the Outputs of a Classifier to New a Priori Probabilities（中文梳理）

## 文献信息

Saerens M., Latinne P., Decaestecker C. Neural Computation，2002，14(1)：21–41。原文：[PDF](../论文PDF/03_Saerens2002_EM_Prior_Shift.pdf)；[DOI](https://doi.org/10.1162/089976602753284446)。

## 核心假设与算法

论文假设训练和测试之间类别先验P(y)改变，但类条件分布P(x|y)保持稳定。EM迭代在E步依据当前测试先验修正每条样本后验，在M步用修正后验的平均更新先验，直至收敛。量化文献常称其为SLD或EMQ。

## 优势

它无需给测试样本真实标签，能同时估计新先验并修正个体后验，是prior probability shift下的经典强方法。较CC/PCC，它显式利用训练先验和全体测试分数的自洽关系。

## 风险

跨生成器或跨表时，合成类的P(x|y)本身可能改变，违反核心假设；分类器概率未校准也会影响迭代。不能把EMQ在随机配比实验中的优势外推到所有开放环境。

## 本课题复现

EMQ/SLD列为必复现。先在固定真实表和固定生成器、仅改变污染率的纯prior shift近似条件下验证，再在跨生成器、跨表条件下测量退化。记录迭代次数、收敛阈值和边界先验处理。

