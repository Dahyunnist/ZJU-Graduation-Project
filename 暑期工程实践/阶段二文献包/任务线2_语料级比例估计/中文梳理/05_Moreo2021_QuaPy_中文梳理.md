# QuaPy: A Python-Based Framework for Quantification（中文梳理）

## 文献信息

Moreo A., Esuli A., Sebastiani F. CIKM 2021：4534–4543。原文：[PDF](../论文PDF/05_Moreo2021_QuaPy.pdf)；[论文DOI](https://doi.org/10.1145/3459637.3482015)；[代码](https://github.com/HLT-ISTI/QuaPy)。

## 框架作用

QuaPy提供量化算法、分类器封装、人工流行率采样协议、评价指标、模型选择和统一实验接口。它让同一底层分类器分数可被不同量化算法复用，从而减少自写实现造成的不公平。

## 对本课题的价值

任务线二可以把每个混合表视为一个sample/bag，真实污染率作为prevalence标签。通过人工流行率协议，在0%到100%合成比例上重复采样，统一输出MAE、MSE等结果。

## 注意事项

框架不是算法本身，论文中列出的所有方法也不必全部复现。必须固定QuaPy版本和底层分类器，并核对方法的训练/验证数据要求。2026年Continuous Sweep论文还指出旧版QuaPy的Median Sweep曾有bug，因此版本号和结果核验尤其重要。

## 实施建议

先在一个数据集上跑通CC、PCC、ACC、PACC、EMQ、HDy、DyS和Median Sweep，再封装本课题跨生成器/跨表采样协议。保存环境文件、随机种子、每个bag的真实与预测比例。

