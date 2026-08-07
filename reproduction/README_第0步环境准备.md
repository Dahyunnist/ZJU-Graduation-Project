# 第 0 步：前置基础与环境准备

本目录已经放好最小闭环脚本：

- `requirements.txt`：对应说明书 3.3 的建库清单；
- `requirements-minimal.txt`：第 0 周最小闭环必需依赖；
- `run_minimal_loop.py`：对应说明书 3.4 的最小闭环实验。

## 1. 创建 Python 环境

建议使用 Python 3.11。当前机器检测到的 `python` 是 3.13，部分科研库可能还没有完全兼容。

如果使用 Conda：

```powershell
conda create -n tabpollution python=3.11 -y
conda activate tabpollution
python -m pip install --upgrade pip
python -m pip install -r requirements-minimal.txt
```

如果不用 Conda，也可以用 Python 3.11 创建 venv：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-minimal.txt
```

`requirements.txt` 保留了说明书中的完整建库清单，后续做 LLM、XGBoost、Streamlit 演示时再安装即可。

## 2. 跑通最小闭环

先用较小训练量快速验证流程：

```powershell
python run_minimal_loop.py --train-rows 3000 --sample-rows 1000 --epochs 20
```

我已经在本机创建并验证了 Conda 环境 `tabpollution`，并用上面的参数跑通了一次正式实验。

如果机器有可用 NVIDIA GPU，并且 PyTorch CUDA 版本安装正确，可以加：

```powershell
python run_minimal_loop.py --train-rows 3000 --sample-rows 1000 --epochs 20 --cuda
```

## 3. 输出结果

运行完成后会生成：

- `outputs/week0/adult_real_sample.csv`：Adult 真实样本；
- `outputs/week0/adult_synthetic_ctgan.csv`：CTGAN 生成的 1000 条合成样本；
- `outputs/week0/distribution_*.png`：真实/合成数据特征分布对比图；
- `outputs/week0/detector_report.txt`：Logistic Regression 区分真实/合成数据的结果。

`detector_report.txt` 中的 ROC-AUC 越接近 0.5，说明合成数据越难被简单分类器区分；越接近 1.0，说明真实数据和合成数据差异越明显。

## 4. 后续建议

确认脚本能跑通后，再逐步增大 `--epochs`，例如 100、300、500，并观察分布图和检测器指标是否变化。
