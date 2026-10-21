# 项目迁移与研究交接

更新日期：2026-10-21。目标是在另一台电脑保留现有实验、重新建立可复现环境，并开展 BLIP 研究。

## 当前状态

已完成 EfficientNetB0 和 DenseNet121 两种 CNN＋Transformer 的全量训练。全部本地数据共 8,091 张图片、40,455 条描述；按图片随机划分 6,472 张训练图、1,619 张验证图。两个模型均在第 7 轮达到最低验证损失、第 10 轮早停，最佳权重已独立加载并生成描述。

| 模型 | 最低验证损失 | 对应词级准确率 |
| --- | ---: | ---: |
| EfficientNetB0＋Transformer | 3.0951 | 40.02% |
| DenseNet121＋Transformer | 3.0735 | 39.90% |

词级准确率不是整句正确率，也不是独立测试结果。详细参数和限制见 [基础训练结果](BASELINE_RESULTS.md)。当前未安装或实现 BLIP，没有独立测试集上的生成质量指标，也未完成论文或最终 PDF 报告。

## 文档与代码导航

| 文件 | 作用 |
| --- | --- |
| README.md | 项目入口、架构与命令 |
| BASELINE_RESULTS.md | 全量训练实测结果及限制 |
| VERIFICATION.md | 早期小样本流程验证记录，不能代替全量实验结论 |
| RESEARCH_PLAN.md | BLIP 与统一评价的下一阶段计划 |
| requirements.txt | 基础 TensorFlow 依赖范围 |
| requirements-baseline-lock.txt | 当前 Windows CPU 环境实际依赖快照 |
| caption_data.py | 本地解压、描述读取和数据划分 |
| 案例5-自动图像描述/案例5-代码和数据/fullname_lab.py | 基础模型训练和预测 |
| run_full_training.py | 顺序执行两种基础模型的全量实验 |
| test_caption_data.py | 数据处理单元测试 |
| verify_models.py | 早期 verify-* 小样本权重检查，默认不检查 full-* |

## 需要复制的内容

推荐复制整个项目文件夹，并排除 `.venv/`、`__pycache__/` 和 `runs/*/features.npy`。虚拟环境不能跨电脑直接复用；特征缓存可重新计算，两个全量缓存合计约 3.65 GB。

必须保留：

- 所有源码、文档、依赖文件及案例正文 DOCX。
- 原始 `Flickr8k_Dataset.zip`、`Flickr8k_text.zip`、`captions.txt`。`data.txt` 仅包含原下载说明，不是图片数据本体。保留压缩包后可以省略已解压图片目录，目标机通过 prepare 重新解压，无需重新下载数据。
- `runs/full-efficientnet/` 和 `runs/full-densenet/` 中除 `features.npy` 外的全部文件。尤其是 `best.weights.h5`、`config.json`、`vocabulary.json`、`split.json`、`history.csv`、`results.json`、`predictions.json`、`training.log`。
- `runs/full-training-status.json`。

注意 `.gitignore` 排除了 `runs/`、`models/` 和解压图片。仅 git clone 或复制代码不能带走这些实验产物；未跟踪的原始数据也不能假定已上传。

### 预训练权重

基础模型当前配置为 `weights=imagenet`。预测构建模型时也会查找对应 ImageNet 权重，因此离线迁移需要复制本机 `%USERPROFILE%\.keras\models\` 的以下文件到目标用户的 `.keras/models/`：

| 文件 | MD5 |
| --- | --- |
| efficientnetb0_notop.h5 | 50bc09e76180e00e4465e1a485ddc09d |
| densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5 | 30ee3e1110167f948a6b9946edeeb738 |

项目 `models/` 不能视为完整备份；其中不一定包含两种权重。不复制 Keras 缓存时，联网环境会尝试下载权重，图片数据仍无需下载。对已有 ImageNet 基线，保留原配置和默认权重加载方式，避免改变 EfficientNet 的固定归一化结构。

## 目标电脑环境

建议有 NVIDIA GPU 的 Linux 或 WSL2，用于下一阶段 BLIP 微调。GPU 显存需求取决于模型、精度、batch size、冻结策略和序列长度，先实测小批次，不承诺固定显存即可完成全部实验。

基础模型使用 Python 3.12、TensorFlow 2.17.1、Keras 3.15.1。Windows 原生 TensorFlow 2.17 不使用 CUDA GPU。Windows CPU 环境可尝试按快照重建；Linux/WSL2 不要直接安装包含 tensorflow-intel 的 Windows 快照，应使用 requirements.txt，并按目标平台配置 TensorFlow GPU 环境。

BLIP 建议单独创建 PyTorch 虚拟环境，根据目标机驱动与 CUDA 兼容情况安装 PyTorch，再安装并锁定 transformers、Pillow 和评价依赖。不要将当前 TensorFlow 环境快照直接当成 BLIP 环境。

## 在新电脑验证基础项目

以下为 Windows PowerShell 命令，在目标项目根目录运行；先确保 `python --version` 为兼容版本（推荐 3.12）。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-baseline-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest -v test_caption_data
$script = '.\案例5-自动图像描述\案例5-代码和数据\fullname_lab.py'
.\.venv\Scripts\python.exe $script prepare
.\.venv\Scripts\python.exe $script predict --output runs/full-efficientnet --image '本地JPEG路径'
.\.venv\Scripts\python.exe $script predict --output runs/full-densenet --image '本地JPEG路径'
```

Linux/WSL2 的虚拟环境解释器为 `.venv/bin/python`，依赖安装使用 requirements.txt。先进行 TensorFlow 导入和设备识别检查，再运行上述数据与预测验证。

`prepare` 预期输出 8,091 张图片、40,455 条描述。复制文件后应检查关键文件的 SHA256 在两台电脑上一致，例如 PowerShell 的 `Get-FileHash 文件路径 -Algorithm SHA256`。

### 路径与恢复限制

历史 `split.json`、`predictions.json` 保存的是原电脑绝对路径，迁移后不能直接拿这些路径读取图片。它们的文件名和成员关系仍是有效实验记录。若需复现历史划分，应取图片 basename，重新拼接目标图片目录并逐一检查存在；保留原始记录，另存重定位后的版本。独立 predict 只需传入新机器上的图片路径，不依赖历史 split 路径。

现有程序支持从头训练和权重加载预测，不提供完整优化器状态断点续训。迁移已完成模型用于预测是可行的；新实验应从预训练初始化重新训练。不要直接运行 `run_full_training.py` 覆盖已完成的 full-* 目录。

## 迁移后的启动顺序

1. 验证文件、数据数量和两个基础模型的预测。
2. 读取 RESEARCH_PLAN.md，落实统一数据划分与评价协议。
3. 在新目录先验证 BLIP 零样本推理，再进行小批次微调试验。
4. 确认显存、训练日志、保存加载、评估流程均正常后，启动正式实验。

交接给下一位开发者的任务描述可使用：

> 请先阅读 MIGRATION.md、BASELINE_RESULTS.md 和 RESEARCH_PLAN.md。保留现有基础模型实验。检查当前电脑 GPU 与运行环境，建立独立 BLIP 环境，先实现统一数据划分和评价，再实现 BLIP 零样本、微调及公平对比；本地 Flickr8k 数据已存在，不要重复下载。所有新实验使用新输出目录，不覆盖历史权重，不虚构未运行的指标。
