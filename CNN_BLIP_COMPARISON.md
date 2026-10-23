# 原案例与 BLIP 的统一对比实验

用户要求补齐引入 BLIP 之前与之后的实际差距，不能用词级准确率与 CIDEr 并排代替实验。因此本轮在 Windows CUDA 重训两种原案例模型，并重新运行 BLIP 统一评价。旧实验和旧 PPT 均保留。

## 固定协议

- 直接读取 `runs/blip-full-random80-20261022/split.json`，6,472 张训练图、1,619 张验证图，种子 111，按图片随机 80/20。原始描述及划分均核对 SHA256。
- 两种 CNN 复用 `fullname_lab.py` 中未经修改的 `build_model`，保持官方 Keras EfficientNetB0 / DenseNet121 ImageNet 权重和主干预处理、Transformer 架构、训练集词表、学习率和早停规则。
- 使用 Keras 3.15.1 的 PyTorch CUDA 后端。原 TensorFlow 操作通过同义 Keras ops 接口执行。这是新后端复现实验，不是恢复原电脑权重；JPEG 解码使用 Pillow，图像缩放使用 float32 半像素双线性插值。
- CNN 冻结；图像 224×224、嵌入 256、长度 25、词表 7,940、batch 32、Adam 1e-4、最多 30 轮、验证损失 patience 3。FP32，关闭 TF32。
- 前四项使用贪心解码：两种 CNN、BLIP 预训练、BLIP 微调。另列微调 BLIP 的 beam=3。
- 所有模型生成上限均为 24 个本模型 token，但各模型词表不同，不能解释为相同词数。
- 全部使用同一份原始参考描述和 `caption_metrics.py` 的 BLEU-1/4、METEOR、ROUGE-L、CIDEr。保存逐图预测，汇总时再次核对 CIDEr/BLEU-4。
- CNN 重新加载完整权重，从原图推理并与缓存预测逐字对照。BLIP 独立重载检查点，另记录原图预测与缓存预测的一致性。

## 运行与复现

`.venv-cnn-gpu` 通过 `Lib/site-packages/blip-runtime.pth` 读取本机 `.venv-blip/Lib/site-packages` 的 PyTorch/CUDA 和指标依赖；新增 Keras 依赖锁定在 `requirements-cnn-gpu-extra.txt`。BLIP 环境不修改。迁移时重新创建虚拟环境并改写 `.pth`，不能直接复制环境。

```powershell
.\.venv-cnn-gpu\Scripts\python.exe -m unittest -v test_cnn_comparison
.\.venv-cnn-gpu\Scripts\python.exe -u run_cnn_blip_comparison.py --output runs/cnn-blip-comparison-20261022
.\.venv-cnn-gpu\Scripts\python.exe analyze_cnn_blip_comparison.py --experiment runs/cnn-blip-comparison-20261022
```

输出目录不允许覆盖。复现实验时更换 `--output`。总状态在实验目录的 `status.json`；各模型日志、完整图像特征缓存、配置、词表、最佳权重、训练历史、生成结果和指标均在同一实验目录。

## 解释范围

本轮是统一验证协议的系统对比。BLIP 同时改变架构并使用额外图文预训练，其此前调参预算也较大，因此不能解释为单独替换 CNN 的因果增益。该验证集已参与早停与调参，只有一个训练种子，没有独立测试集。配对图片 bootstrap 区间只描述当前验证样本上的差值不确定性，不覆盖训练随机性和模型选择偏差。

实测结果完成后写入 `CNN_BLIP_COMPARISON_RESULTS.md` 和实验目录的 `comparison.json`。PPT 只采用完整实验的真实记录。
