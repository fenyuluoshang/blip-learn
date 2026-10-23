# CNN 与 BLIP 统一验证对比结果

本轮重训原案例两种 CNN + Transformer，全部采用同一 6,472/1,619 张图像划分和原始参考描述。前四项统一使用贪心解码，最后一项单列 beam=3。所有模型生成上限均为 24 个本模型 token。

| 模型 | CIDEr | BLEU-4 | METEOR | ROUGE-L |
| --- | ---: | ---: | ---: | ---: |
| EfficientNetB0 + Transformer | 0.4728 | 0.1783 | 0.3947 | 0.4352 |
| DenseNet121 + Transformer | 0.4685 | 0.1772 | 0.3949 | 0.4284 |
| BLIP 预训练 | 0.6506 | 0.2486 | 0.4173 | 0.5018 |
| BLIP 微调 | 1.0135 | 0.3477 | 0.5546 | 0.5650 |
| BLIP 微调 + beam=3 | 1.0353 | 0.3565 | 0.5652 | 0.5698 |

相对本轮 CIDEr 较高的 CNN（EfficientNetB0 + Transformer），微调 BLIP 在相同贪心解码下 CIDEr 增加 0.5407（相对 114.4%），BLEU-4 增加 0.1694。图片级配对 bootstrap 的 CIDEr 差值 95% 区间为 [0.5054, 0.5774]，仅反映当前验证样本的不确定性。

## 实验条件

- 复用原 `fullname_lab.py` 模型构建函数，保留两种官方 Keras ImageNet 主干及其预处理。
- Keras 3 的 PyTorch CUDA 后端在 Windows GPU 重训，FP32，随机种子 111。原历史权重未迁入，不能称为恢复历史预测。
- CNN 冻结，Transformer 维度 256，长度 25，训练集拟合词表 7,940，Adam 1e-4、batch 32，最多 30 轮，验证损失 patience 3。
- CNN 检查点按验证损失选择；BLIP 复用此前验证 CIDEr 选出的学习率 5e-6 最佳检查点。
- 各模型均独立重载权重并核验原图预测。CIDEr 与 BLEU-4 从逐图预测重新计算并核对报告。

## 解释边界

- 这是同一验证集上的系统效果对比，BLIP 同时改变模型结构并引入额外图文预训练数据。
- 该验证集参与过早停和调参，不是独立测试。仅一个训练种子，BLIP 与 CNN 的参数搜索预算不同。
- 统一生成步数仍使用各自 tokenizer，不能解释为相同词数。
- 配对区间不覆盖训练随机性、模型选择偏差和预训练数据重叠风险。

逐图对照、模型报告、训练曲线与检查点保存在 `runs/cnn-blip-comparison-20261022`。划分 SHA256：`c955cdd0e38a1dfb3cdd322796345db95ad33f77c64f52fd5265ef3f002fdaa9`。
