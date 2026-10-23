# Windows BLIP 实验环境

本项目的 BLIP 使用 Windows 原生 PyTorch CUDA 环境，独立目录为 `.venv-blip`。已有 TensorFlow 基础模型仍使用原来的环境；这里的依赖文件不适用于 TensorFlow。

## 本机验证结果

验证日期为 2026-10-22。Windows 11、Python 3.12.10、RTX 4070 Ti（12 GB）、驱动 610.47；PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、Transformers 4.57.6，CUDA 运行库为 12.8。原有系统 Python 的包未修改。

| 检查 | 实测结果 |
| --- | --- |
| 依赖检查 | `pip check` 通过 |
| 数据处理测试 | 两项单元测试通过 |
| CUDA 运算 | 矩阵乘法、反向传播和 torchvision CUDA NMS 通过 |
| 本地数据 | 8,091 张图片、40,455 条描述；官方 6000/1000/1000 名单无交叉、无缺失 |
| BLIP 源权重 | 固定 revision，原始权重 SHA256 与官方 LFS 元数据一致 |
| 批量零样本推理 | 三张官方验证图片均生成非空文本 |
| 训练检查 | 两张官方训练图片、十条描述、batch size 2，完成五次 AdamW 更新 |
| 损失与梯度 | 五次损失均有限（3.2063、3.0531、2.4405、2.5107、2.9705），梯度有限且非零，文本参数发生更新 |
| 冻结与 masking | 视觉编码器无梯度，padding 标签为 `-100` |
| 保存加载 | 模型和 processor 保存成功，独立离线进程的三张图片生成结果与保存前一致 |
| 显存 | PyTorch 峰值分配 3088 MiB，峰值保留 3482 MiB；不包含桌面及其他程序占用 |

原始报告为 `runs/blip-env-20261022-verified/report.json`，终端日志为 `runs/blip-env-20261022-verified.log`。逐图生成及五条参考描述保存在该目录的 `zero_shot_predictions.json` 和 `after_smoke_predictions.json`，检查点位于 `checkpoint/`。本次未使用测试集图片，不报告 BLEU、METEOR、ROUGE-L 或 CIDEr。

三张零样本生成分别为 `a body of water`、`a boy wearing a red jacket` 和 `a woman and two dogs`。其中最后一张参考描述包含三只狗，说明推理流程通过不能代替质量评价。五次更新对应不同描述，不能将损失变化解释为收敛或性能提升。当前显存数字只适用于本次冻结策略、序列长度与小批次，正式微调仍需单独测量。

## 重建环境

在项目根目录运行以下 PowerShell 命令。无需激活虚拟环境，也不需要另外安装完整 CUDA Toolkit；需有兼容的 NVIDIA 驱动。

```powershell
python -m venv .venv-blip
.\.venv-blip\Scripts\python.exe -m pip install --upgrade pip
.\.venv-blip\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv-blip\Scripts\python.exe -m pip install -r requirements-blip-lock.txt
.\.venv-blip\Scripts\python.exe -m pip check
```

`requirements-blip.txt` 记录直接依赖，`requirements-blip-lock.txt` 记录本次安装的全部包版本。必须先安装 CUDA wheels，因为默认 PyPI 不提供这里锁定的 `+cu128` 包。当前采用 Transformers 4.57.6 的 BLIP 接口。

## 小范围验证

```powershell
$env:HF_HOME = Join-Path (Get-Location) 'models\huggingface'
$env:HF_HUB_DISABLE_XET = '1'
$env:HF_HUB_DOWNLOAD_TIMEOUT = '30'
$env:PYTHONIOENCODING = 'utf-8'
.\.venv-blip\Scripts\python.exe -m unittest -v test_caption_data
.\.venv-blip\Scripts\python.exe -u verify_blip_environment.py
```

第一次运行下载 BLIP 权重和 processor，权重约 990 MB。模型及缓存位于 `models/huggingface/`，本地 Flickr8k 数据不重新下载。每次验证自动使用新输出目录；指定 `--output` 时目录必须不存在。

检查点为 `Salesforce/blip-image-captioning-base`，固定 revision 为 `82a37760796d32b1411fe092ab5d4e227313294b`，原始 `pytorch_model.bin` 的 SHA256 为 `d6638651a5526cc2ede56f2b5104d6851b0755816d220e5e046870430180c767`。模型卡标注 BSD-3-Clause，描述为使用 COCO 的 ViT base 图像描述模型。尚未确认其训练数据与 Flickr8k 是否重叠，因此不能宣称评测数据已去重。

验证包含 CUDA 矩阵乘法及反向传播、torchvision CUDA 扩展、官方数据名单核验、三张验证图片的批量零样本生成，以及两张训练图片全部十条原始描述的五次优化器更新。训练采用冻结视觉编码器、FP32 参数、FP16 autocast 和 GradScaler，batch size 为 2；padding 标签置为 `-100`，BLIP 解码起始 token 按模型配置处理。

之后保存模型与 processor 为本地检查点，并在独立进程中只读取本地文件，核对加载后生成结果一致。验证同时保存优化器、GradScaler 和随机状态，但脚本未实现断点续训功能。

仅检查 GPU 环境时可运行：

```powershell
.\.venv-blip\Scripts\python.exe verify_blip_environment.py --gpu-only
```

已有模型缓存后可加 `--local-files-only` 重复验证。完整输出包含 `report.json`、`sample_manifest.json`、零样本及训练后逐图预测、`checkpoint/` 和独立加载日志。这里的训练损失、生成样例和运行时间用于检查环境，不代表完整训练或生成质量评价。

此后已实现 `train_blip.py` 的完整验证评价与断点恢复，并按用户指定的随机 80/20 划分完成四组全量微调、独立重载和模型导出，最终验证 CIDEr 1.0351，见 [BLIP 优化结果](BLIP_OPTIMIZATION_RESULTS.md)。首组配置见 [BLIP 全量训练](BLIP_FULL_TRAINING.md)。基础模型重训及统一句级评价也已完成，见 [CNN/BLIP 统一对比](CNN_BLIP_COMPARISON_RESULTS.md)；独立测试集评价仍待开展。

本机第一次下载时 Hugging Face 整文件传输停滞，已改用官方地址的分段下载并验证原始权重哈希。本机完整源检查点保存于 `models/blip-base-source-82a3776/`，包含 processor 配置、模型卡及记录 revision 的 `source.json`。可直接使用这个目录离线运行，省去再次访问模型服务：

```powershell
.\.venv-blip\Scripts\python.exe -u verify_blip_environment.py --model-dir models/blip-base-source-82a3776
```
