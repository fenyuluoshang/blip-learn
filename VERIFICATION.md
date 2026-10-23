# 自动图像描述案例验证记录

验证日期：2026-10-21。环境：Windows、Python 3.12.14、TensorFlow 2.17.1、Keras 3.15.1，CPU。

## 验证范围与结果

- 数据准备成功：8,091 张图片、40,455 条描述；默认训练集 6,472 张、验证集 1,619 张。使用已有本地数据，没有重新下载。
- `python -m unittest -v test_caption_data`：两项测试通过，覆盖 CSV/tab 描述读取、标点处理、结束标记保留、可复现划分与图片集合隔离。
- EfficientNetB0 和 DenseNet121 各完成一次实际训练、验证集评估，成功写入最佳权重、配置、词表、划分、CSV 日志和 JSON 结果。
- 两种模型均通过 `verify_models.py` 检查：输出形状正确、数值有限、概率归一化、未来词不影响先前预测、补零不影响损失、独立重建模型加载权重后预测一致。
- 两种模型的独立 CLI 预测进程均正常退出，能够读取本地 JPEG、加载保存的模型并生成文本。

## 小样本训练参数与结果

两种模型均使用 `--limit 4 --epochs 1 --batch-size 2 --weights none --train-backbone`，即随机初始化，3 张训练图片（15 条描述）、1 张验证图片（5 条描述），224×224 图像，256 维嵌入，序列长度 25，Adam 学习率 0.0001，随机种子 111。

| 模型 | 参数量 | 训练耗时（秒） | 训练损失 | 验证损失 |
| --- | ---: | ---: | ---: | ---: |
| EfficientNetB0 + Transformer | 5,742,579 | 56.67 | 4.44484 | 4.09473 |
| DenseNet121 + Transformer | 8,664,976 | 93.32 | 4.39834 | 4.59000 |

原始记录位于 `runs/verify-efficientnet/` 与 `runs/verify-densenet/`。两个训练进程存在时间重叠，耗时包含首次图编译，不适合用来比较架构速度。

在同一张验证图片上，两个模型的示例输出分别为 `a` 和 `a a a a a a a a a a a a`。这证明推理链路可运行，也说明随机初始化、小样本、单轮训练尚不具备有效描述能力。上述数字不能用于模型质量排序，亦不能代替正文要求的正式对比实验。未验证 ImageNet 预训练权重下载、完整数据集收敛、GPU 执行或最终描述质量。

运行时 Keras 提示跳过 Adam 优化器状态加载：预测只需要模型参数，模型参数恢复一致性检查已通过。当前程序提供训练和独立预测，不提供恢复优化器状态后的断点续训。

## 复现

在项目根目录执行：

```powershell
$python = '.\.venv\Scripts\python.exe'
$script = '.\案例5-自动图像描述\案例5-代码和数据\fullname_lab.py'
& $python -m unittest -v test_caption_data
& $python $script prepare
& $python $script train --backbone efficientnet --limit 4 --epochs 1 --batch-size 2 --weights none --train-backbone --output runs/verify-efficientnet
& $python $script train --backbone densenet --limit 4 --epochs 1 --batch-size 2 --weights none --train-backbone --output runs/verify-densenet
& $python verify_models.py
```

独立预测可将任一本地 JPEG 路径传给 `predict --output runs/verify-efficientnet --image '图片路径'`；换成 DenseNet 的输出目录即可验证另一个模型。

## 2026-10-23 提交前验证

Windows CUDA 环境中，BLIP 与数据处理测试 14 项、CNN 对比契约测试 2 项通过。两套环境分别执行：

```powershell
.\.venv-blip\Scripts\python.exe -m unittest -v test_caption_data test_blip_amp test_blip_loss test_blip_training test_blip_tuning
.\.venv-cnn-gpu\Scripts\python.exe -m unittest -v test_cnn_comparison
```

BLIP 测试覆盖图片隔离划分、padding、生成指标、损失与梯度累积等价、AMP 溢出恢复及调参流程。CNN 测试检查因果掩码、冻结主干和 padding 损失。CNN 环境的建立方法见 [统一实验](CNN_BLIP_COMPARISON.md)，不能在未安装 Keras 的 BLIP 环境中运行该组测试。

当前部分测试是本地集成检查：`test_blip_training` 需要 `models/blip-base-source-82a3776` 的 tokenizer 和 `models/nltk_data` 的 WordNet；`test_blip_tuning` 读取 `runs/blip-full-random80-20261022/config.json` 与 `runs/blip-training-preflight-20261022` 的审计产物。它们均未提交，刚克隆仓库时须按迁移文档恢复后再运行完整检查。仅数据处理、AMP 与损失测试可在装好依赖后独立运行。
