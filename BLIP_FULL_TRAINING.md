# BLIP 全量随机划分实验

2026-10-22 在 Windows 主系统启动。按用户要求，本轮使用全部本地 Flickr8k 图片，固定随机种子 111，按图片随机划分 80% 训练与 20% 验证。输出为 `runs/blip-full-random80-20261022/`，运行期间的实际状态以 `status.json` 为准。

完成记录：首组于第 7 轮早停，最佳为第 4 轮，验证 CIDEr 0.9977；随后另外三组学习率实验也已完成。最终选中学习率 `5e-6`、第 6 轮检查点，beam=3 验证 CIDEr 1.0351，独立预测复核与模型导出通过，详见 [BLIP 优化结果](BLIP_OPTIMIZATION_RESULTS.md)。下列物理批次和日志命令对应首组。

启动检查：所有 8,091 张图片的视觉缓存已完成，完整验证集的 epoch 0 评价已保存，第 1 轮已产生真实优化器更新。训练前零样本验证 BLEU-1 为 0.6270、BLEU-4 为 0.2486、METEOR 为 0.4173、ROUGE-L 为 0.5017、CIDEr 为 0.6506，验证损失为 3.2386。这些是本轮固定随机验证集及下述分词协议下的预训练模型结果，微调结果以持续更新的 `history.json` 为准。

## 数据与配置

8,091 张图片中，6,472 张作为训练集，共 32,360 条描述；1,619 张作为验证集，共 8,095 条描述。非整数分界按训练集向下取整，剩余图片归验证集。图片的全部描述留在同一集合。`split.json` 只记录文件名，`config.json` 记录清单哈希、描述文件哈希和源权重版本。

本轮不采用官方 6000/1000/1000 划分，也未另留独立测试集。生成指标属于用于模型选择的验证集指标。历史 TensorFlow 基础模型使用同一随机种子和划分算法；其权重及实际 `split.json` 仍未迁入，因此历史实验成员尚未做产物级比对。

| 配置 | 本轮设置 |
| --- | --- |
| 初始模型 | 原始 `Salesforce/blip-image-captioning-base`，固定 revision `82a37760796d32b1411fe092ab5d4e227313294b` |
| 冻结范围 | 视觉编码器；训练文本生成部分 |
| 输入图像 | 官方 processor，384×384 |
| 视觉缓存 | 按图片提取 FP16 特征一次，约 6.7 GiB，磁盘映射读取 |
| 精度 | FP32 模型参数、FP16 autocast、GradScaler |
| 物理 batch size | 2 |
| 梯度累积 | 8，有效 batch size 16；每轮最后一组按实际批次数归一化 |
| 优化器 | AdamW，学习率峰值 `1e-5`，weight decay 0.01，梯度范数裁剪 1.0 |
| 学习率 | 前 5% 优化步骤预热，之后余弦下降 |
| 训练文本 | 每张图的全部原始描述，使用 BLIP tokenizer，不套用旧基线的清洗词表 |
| 文本长度 | 最多 32 tokens，保留 tokenizer 特殊标记；padding 标签为 `-100` |
| 验证生成 | 每轮全部 1,619 张验证图，贪心生成，最多 30 个新 tokens |
| 模型选择 | 最大验证 CIDEr；原始零样本模型也参与最佳模型保留 |
| 训练轮数 | 最多 10 轮，CIDEr 连续 3 轮未改善时早停 |
| 保存频率 | 每 250 次优化器更新和每轮结束保存 `last/`，更新最佳结果时保存 `best/` |

先记录 epoch 0 的完整验证集零样本结果，然后开始训练。首组每个完整训练轮含 16,180 个微批次、2,023 个梯度累积窗口；AMP 溢出时跳过该窗口的优化器更新并降低缩放值，实际更新次数扣除跳过次数。

## 查看进度

```powershell
Get-Content runs/blip-full-random80-20261022/status.json -Encoding UTF8
Get-Content runs/blip-full-random80-20261022.20261022-045157.log -Tail 20 -Wait
```

任务通过隐藏后台进程运行，不依赖当前终端窗口。脚本存活期间阻止 Windows 自动睡眠，退出时恢复；系统关机、手动睡眠或结束进程仍会中断训练。

`phase` 依次显示模型加载、视觉特征缓存、验证损失、验证生成、指标计算和训练。`batch`、`global_step` 和 `epoch` 记录训练进度，`last_checkpoint_step` 表示已安全落盘的最近优化步骤。错误时 `status` 为 `failed`，详情保存在状态文件与 stderr 日志。

## 输出与恢复

- `config.json`、`environment.json`、`split.json`、`source/`：配置、版本、数据清单与运行源码快照。
- `vision-features.npy` 和 JSON 元数据：冻结视觉特征缓存，可在恢复时复用。
- `history.json`：零样本和每轮验证指标。
- `predictions/epoch-00.json` 等：每张验证图片的原始生成文本及全部参考描述。
- `best/`：按验证 CIDEr 选出的模型、processor 和选择记录。
- `last/`：模型、processor、优化器、调度器、GradScaler、随机状态、当前轮数、批次游标和累计指标。
- `results.json`：训练完成后生成的汇总。

训练意外中断后，在项目根目录运行以下命令恢复。启动器会检查原训练进程，避免重复启动：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\start_blip_training.ps1 -Resume
```

恢复使用最近完整写入的检查点，其后尚未落盘的步骤会重新执行。保存状态只在完成梯度累积后进行，不丢失未完成的累积梯度。恢复时检查配置和数据清单哈希，防止切换数据后继续训练。保存采用临时目录替换，并短暂保留上一版目录。恢复后继续使用相同输出和特征缓存。

## 评价协议与验证

使用 NLTK BLEU-1/4、METEOR、rouge-score ROUGE-L、pycocoevalcap CIDEr。所有指标共用小写 ASCII 字母数字分词，BLEU 不使用平滑，METEOR 使用默认词干及 WordNet 同义词匹配，ROUGE-L 取最佳参考 F1 后按图片平均。CIDEr 使用库原生尺度（内部含乘 10），其他指标为 0–1。未使用 Java PTBTokenizer，因此与其他实现的数字比较前需统一分词协议。

本地 WordNet 位于 `models/nltk_data/corpora/wordnet.zip`，从 NLTK 官方数据仓库下载并核对官方清单校验值，来源记录位于 `models/nltk_data/wordnet-source.json`。所有评价依赖已加入 `requirements-blip-lock.txt`，训练期间无需联网。

全量启动前，五项数据/评价测试通过；14 图短训练完成第一轮，保存后独立启动进程，成功恢复并完成第二轮，总计 8 次优化器更新。训练状态、学习率调度和完整验证流程均正常，详情保存在 `runs/blip-training-preflight-20261022/`。这些小样本数字只用于检查流程，不作为正式模型质量结论。
