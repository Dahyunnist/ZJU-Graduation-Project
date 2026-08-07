# 数据卡：Default of Credit Card Clients

## 来源与许可

- 数据集ID：`credit`
- UCI ID：350；DOI：10.24432/C55S3H
- 来源页面：https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients
- 许可：CC BY 4.0
- 获取日期：2026-07-15
- 原始文件 SHA-256：`56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`
- 处理文件 SHA-256：`ee2fa731865931ec6c4c54d7c16e00f1051b9163a0bfeb7aa318e795302a4d29`

## 数据概况

- 任务：binary_classification；目标列：`default_payment_next_month`
- 原始行数：30000；去除完全重复后：29965
- 完全重复行移除数：35
- 特征数：23（数值 14，类别 9）
- 数值列：LIMIT_BAL, AGE, BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6
- 类别列：SEX, EDUCATION, MARRIAGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6

### 目标分布

|取值|数量|比例|
|---|---:|---:|
|0|23335|0.778742|
|1|6630|0.221258|

### 缺失情况

|列|缺失数|缺失率|
|---|---:|---:|
|LIMIT_BAL|0|0.000000|
|AGE|0|0.000000|
|BILL_AMT1|0|0.000000|
|BILL_AMT2|0|0.000000|
|BILL_AMT3|0|0.000000|
|BILL_AMT4|0|0.000000|
|BILL_AMT5|0|0.000000|
|BILL_AMT6|0|0.000000|
|PAY_AMT1|0|0.000000|
|PAY_AMT2|0|0.000000|
|PAY_AMT3|0|0.000000|
|PAY_AMT4|0|0.000000|
|PAY_AMT5|0|0.000000|
|PAY_AMT6|0|0.000000|
|SEX|0|0.000000|
|EDUCATION|0|0.000000|
|MARRIAGE|0|0.000000|
|PAY_0|0|0.000000|
|PAY_2|0|0.000000|
|PAY_3|0|0.000000|
|PAY_4|0|0.000000|
|PAY_5|0|0.000000|
|PAY_6|0|0.000000|

## 冻结分区

采用顺序分层划分：先留出15%最终测试，再从其余85%留出10%验证，最后将75%拆为60%来源训练和15%检测训练。

|seed|R_source_train|R_detector_train|R_detector_val|R_final_test|assignment SHA-256|
|---:|---:|---:|---:|---:|---|
|2026|17978|4495|2997|4495|`9febe610a191d7008f5394f731596b383c877b31451703b96b6cf0f9839d6971`|
|2027|17978|4495|2997|4495|`b5f0378bb3b3d52b1873062dc6ee3f737f96bfe473eb92c1a317f37bd173d6d9`|
|2028|17978|4495|2997|4495|`988939610d07e7de573687dc6db366d662b7c084c8f024af7aa254eb086e2688`|
|2029|17978|4495|2997|4495|`746659abbbb320bd25e6ea71dcf0c34a3e650fb96d482f550e6b2be17fae73a7`|
|2030|17978|4495|2997|4495|`bf5e2413f2ad9fdbae744472ccced208388dd92c9e577048020ad84092e2a47d`|

### 各分区目标分布

- seed=2026
  - `R_source_train`：0=0.778730, 1=0.221270
  - `R_detector_train`：0=0.778865, 1=0.221135
  - `R_detector_val`：0=0.778779, 1=0.221221
  - `R_final_test`：0=0.778643, 1=0.221357
- seed=2027
  - `R_source_train`：0=0.778730, 1=0.221270
  - `R_detector_train`：0=0.778865, 1=0.221135
  - `R_detector_val`：0=0.778779, 1=0.221221
  - `R_final_test`：0=0.778643, 1=0.221357
- seed=2028
  - `R_source_train`：0=0.778730, 1=0.221270
  - `R_detector_train`：0=0.778865, 1=0.221135
  - `R_detector_val`：0=0.778779, 1=0.221221
  - `R_final_test`：0=0.778643, 1=0.221357
- seed=2029
  - `R_source_train`：0=0.778730, 1=0.221270
  - `R_detector_train`：0=0.778865, 1=0.221135
  - `R_detector_val`：0=0.778779, 1=0.221221
  - `R_final_test`：0=0.778643, 1=0.221357
- seed=2030
  - `R_source_train`：0=0.778730, 1=0.221270
  - `R_detector_train`：0=0.778865, 1=0.221135
  - `R_detector_val`：0=0.778779, 1=0.221221
  - `R_final_test`：0=0.778643, 1=0.221357

## 已知风险

- Preprocessors must be fit only on the permitted training partition.
- row_id, split and provenance fields are metadata and must never become model features.
- Categorical codes in Credit carry semantics and are not continuous measurements.
