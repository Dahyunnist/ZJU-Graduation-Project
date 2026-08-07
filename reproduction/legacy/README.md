# Legacy smoke validation

这里保存对阶段一旧脚本的非破坏性复核结果，不属于 benchmark v1 正式实验。

- `validation_20260715`：通过 `conda run` 复用 week0 CSV；脚本完成后，Conda 包装器在输出进度字符时触发GBK编码错误。
- `validation_20260715_direct`：直接调用 `D:\Anaconda3\envs\tabpollution\python.exe` 重复相同评估，成功得到与历史 reuse 完全一致的 AUROC `0.848992`。

不得把本目录结果加入正式结果聚合。

