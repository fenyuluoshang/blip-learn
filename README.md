# Flickr8k 自动图像描述实践

根据案例正文实现 CNN + Transformer，并支持 EfficientNetB0 与 DenseNet121 两种视觉模型。原始教学代码保存在同目录的 `fullname_lab_original.py`。

两组基础模型全量训练已完成。迁移到另一台电脑请先阅读 [迁移与交接](MIGRATION.md)，实测结果见 [基础训练结果](BASELINE_RESULTS.md)，下一阶段见 [BLIP 研究计划](RESEARCH_PLAN.md)。

## 使用方法

在项目根目录使用 Python 3.10–3.12。Windows 原生 TensorFlow 默认使用 CPU；GPU 训练建议使用 WSL2。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$script = '.\案例5-自动图像描述\案例5-代码和数据\fullname_lab.py'
python $script prepare
python $script train --backbone efficientnet --output runs/efficientnet
python $script train --backbone densenet --output runs/densenet
python $script predict --output runs/efficientnet --image '本地图片.jpg'
```

`prepare` 不需要 TensorFlow。直接解压已有的 Flickr8k_Dataset.zip，并读取已有的 captions.txt；不会下载数据，也不会删除原始压缩包。路径以脚本位置为基准，不依赖运行目录。可使用 `--data-dir` 指定其他本地数据目录。

默认 `--weights imagenet` 在首次训练时可能由 Keras 下载**预训练权重**，这与 Flickr8k 数据不同。完全离线复现已有模型时，应复制 Keras 预训练权重缓存并保留 `imagenet` 配置，具体位置见迁移文档。自定义权重路径仅用于明确掌握相应架构与预处理的新实验。也可使用 `--weights none --train-backbone` 从头训练，但数据量小，效果通常较差。

小规模流程验证：

```powershell
python $script train --limit 20 --epochs 1 --batch-size 2 --weights none --train-backbone --output runs/smoke
```

该命令只验证训练流程，不代表模型已经具有可靠描述能力。正式对比两种模型时，使用相同 limit、随机种子、batch-size 和 epochs，并使用不同输出目录。

以下全量脚本会使用固定 full-* 输出目录；已完成的实验应先备份，新实验请修改为独立目录，避免覆盖。完整的两个模型实验可执行 `.\.venv\Scripts\python.exe run_full_training.py`，顺序使用全部数据训练，每个模型最多 30 轮，验证损失连续 3 轮未改善时早停。结果写入 `runs/full-efficientnet/` 和 `runs/full-densenet/`，总进度写入 `runs/full-training-status.json`。每个目录的 `training.log` 记录当前轮数和指标，结束后生成 `results.json` 及三张验证图片的 `predictions.json`。

正式训练启用 `--cache-features`，冻结 CNN 并将每张图片的视觉特征缓存一次，减少 CPU 重复计算。缓存使用磁盘映射，不要求将全部特征复制进 Python 内存；不能与 `--train-backbone` 同时使用。保存的仍是完整图片模型权重，独立预测无需读取缓存。

## 设计思路

1. **数据准备**：只提取本地压缩包中的 JPEG，忽略 macOS 元数据；兼容 CSV 描述文件和原始 tab 分隔文件，检查每张图是否存在。将描述转小写、清理标点，加上 `<start>` 和 `<end>`，最多保留 23 个正文词，防止结束标记被截断。
2. **划分与编码**：固定种子 111，按图片划分 80% 训练、20% 验证，防止同一图片的不同描述跨集合泄漏。词表仅在训练集拟合；每张图片的所有描述都参与学习，不假定每张图必须恰有五条描述。
3. **视觉特征**：224×224 图像输入 EfficientNetB0 或 DenseNet121，将二维特征图转换为区域序列。默认冻结预训练 CNN，减少训练开销。分别采用对应模型的预处理，EfficientNet 保持 0–255 输入。
4. **Transformer**：视觉编码器通过自注意力整合图像区域；文本解码器将词嵌入与位置嵌入相加，使用因果自注意力读取已经出现的词，再通过交叉注意力关注图像区域，预测下一个词。
5. **训练**：采用 teacher forcing，输入句子去掉最后一个词，目标句子去掉第一个词。损失与词级准确率忽略补零位置。使用 Adam、验证损失早停及最佳权重保存。
6. **预测**：加载配置、词表和最佳权重，从 `<start>` 开始逐词贪心生成，遇到 `<end>` 或达到长度上限结束，屏蔽补零、未知词和重复起始标记。

实现采用 Keras 函数式模型，使用权重文件加 JSON 配置恢复，避免自定义层序列化依赖。默认每条描述分别输入对应图片；开启特征缓存时，训练头与完整模型共享参数，因此仍可保存完整模型并独立预测。ImageNet EfficientNet 的固定归一化结构在训练与加载时保持一致，预测时需要对应预训练权重已在本机 Keras 缓存中（否则 Keras 会尝试下载）。

## 输出与实验说明

每个输出目录包含 `best.weights.h5`、`config.json`、`vocabulary.json`、`split.json`、`history.csv` 和 `results.json`。最后一个文件记录模型参数量、训练耗时及每轮训练和验证指标。

案例最终要求两种视觉模型的实验对比与 PDF 报告。已完成两种模型的全量训练及独立加载预测，见 [基础训练结果](BASELINE_RESULTS.md)；[验证记录](VERIFICATION.md) 保留的是早期小样本检查。尚未完成独立测试集上的生成质量对比和 PDF 报告。词级准确率不是整句描述质量，需要结合验证图片的生成结果判断；更完整的评估可后续增加 BLEU 等指标及独立测试集。
