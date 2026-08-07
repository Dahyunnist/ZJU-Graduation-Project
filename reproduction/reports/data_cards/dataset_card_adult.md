# 数据卡：UCI Adult

## 来源与许可

- 数据集ID：`adult`
- UCI ID：2；DOI：10.24432/C5XW20
- 来源页面：https://archive.ics.uci.edu/dataset/2/adult
- 许可：CC BY 4.0
- 获取日期：2026-07-15
- 原始文件 SHA-256：`7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb`
- 处理文件 SHA-256：`9cf6e79f08b62089624828b3b3c60c64950fea93948f5a5b2829d9194724f2d2`

## 数据概况

- 任务：binary_classification；目标列：`income`
- 原始行数：48842；去除完全重复后：48790
- 完全重复行移除数：52
- 特征数：14（数值 6，类别 8）
- 数值列：age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
- 类别列：workclass, education, marital-status, occupation, relationship, race, sex, native-country

### 目标分布

|取值|数量|比例|
|---|---:|---:|
|<=50K|37109|0.760586|
|>50K|11681|0.239414|

### 缺失情况

|列|缺失数|缺失率|
|---|---:|---:|
|age|0|0.000000|
|fnlwgt|0|0.000000|
|education-num|0|0.000000|
|capital-gain|0|0.000000|
|capital-loss|0|0.000000|
|hours-per-week|0|0.000000|
|workclass|2795|0.057286|
|education|0|0.000000|
|marital-status|0|0.000000|
|occupation|2805|0.057491|
|relationship|0|0.000000|
|race|0|0.000000|
|sex|0|0.000000|
|native-country|856|0.017545|

## 冻结分区

采用顺序分层划分：先留出15%最终测试，再从其余85%留出10%验证，最后将75%拆为60%来源训练和15%检测训练。

|seed|R_source_train|R_detector_train|R_detector_val|R_final_test|assignment SHA-256|
|---:|---:|---:|---:|---:|---|
|2026|29273|7319|4879|7319|`47d75a9fc88b09901a020f5eb57f62fadd6c55921aaf63e5fcdcb8a2b904eff1`|
|2027|29273|7319|4879|7319|`54e13902a34e68a9e03213f50eb7d354d3f7622c844d1a752286592c52bc2b45`|
|2028|29273|7319|4879|7319|`3b61014cba56bf0c173f4b6699fb32f4be7502c233759428a6d916ec38fe6810`|
|2029|29273|7319|4879|7319|`744bccdbd3d94f3e8de3401f24025a46cb6f80271b44ef3b7a9cec061a91f974`|
|2030|29273|7319|4879|7319|`ff44fedc90aa5099ea12fbfb2d8338f331627c9dc180bb9cd61feae0999c5c6f`|

### 各分区目标分布

- seed=2026
  - `R_source_train`：<=50K=0.760564, >50K=0.239436
  - `R_detector_train`：<=50K=0.760623, >50K=0.239377
  - `R_detector_val`：<=50K=0.760607, >50K=0.239393
  - `R_final_test`：<=50K=0.760623, >50K=0.239377
- seed=2027
  - `R_source_train`：<=50K=0.760564, >50K=0.239436
  - `R_detector_train`：<=50K=0.760623, >50K=0.239377
  - `R_detector_val`：<=50K=0.760607, >50K=0.239393
  - `R_final_test`：<=50K=0.760623, >50K=0.239377
- seed=2028
  - `R_source_train`：<=50K=0.760564, >50K=0.239436
  - `R_detector_train`：<=50K=0.760623, >50K=0.239377
  - `R_detector_val`：<=50K=0.760607, >50K=0.239393
  - `R_final_test`：<=50K=0.760623, >50K=0.239377
- seed=2029
  - `R_source_train`：<=50K=0.760564, >50K=0.239436
  - `R_detector_train`：<=50K=0.760623, >50K=0.239377
  - `R_detector_val`：<=50K=0.760607, >50K=0.239393
  - `R_final_test`：<=50K=0.760623, >50K=0.239377
- seed=2030
  - `R_source_train`：<=50K=0.760564, >50K=0.239436
  - `R_detector_train`：<=50K=0.760623, >50K=0.239377
  - `R_detector_val`：<=50K=0.760607, >50K=0.239393
  - `R_final_test`：<=50K=0.760623, >50K=0.239377

## 已知风险

- Preprocessors must be fit only on the permitted training partition.
- row_id, split and provenance fields are metadata and must never become model features.
- Categorical codes in Credit carry semantics and are not continuous measurements.
