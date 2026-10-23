"""Validate matched predictions and quantify image-paired captioning gains."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.cider.cider import Cider

from blip_training_data import manifest_hash
from caption_metrics import PROTOCOL, tokens

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    base = args.experiment
    if read(base / 'status.json')['status'] != 'complete':
        raise ValueError('Comparison jobs have not all completed')
    split = read(ROOT / 'runs/blip-full-random80-20261022/split.json')
    split_hash = manifest_hash(split)
    names = split['validation']
    models = []
    baseline_rows = None
    per_image = {}
    labels = {'efficientnet': 'EfficientNetB0 + Transformer', 'densenet': 'DenseNet121 + Transformer',
              'blip-pretrained-greedy': 'BLIP 预训练', 'blip-finetuned-greedy': 'BLIP 微调',
              'blip-finetuned-beam3': 'BLIP 微调 + beam=3'}
    for key, label in labels.items():
        report = read(base / key / ('results.json' if key in ['efficientnet', 'densenet'] else 'report.json'))
        rows = read(base / key / 'predictions.json')
        if report['split_sha256'] != split_hash or [row['image'] for row in rows] != names:
            raise ValueError(f'Different validation manifest: {key}')
        if baseline_rows is None:
            baseline_rows = rows
        elif [row['references'] for row in rows] != [row['references'] for row in baseline_rows]:
            raise ValueError(f'Different reference captions: {key}')
        if report['metric_protocol'] != PROTOCOL or report['generation']['max_new_tokens'] != 24:
            raise ValueError(f'Different evaluation protocol: {key}')
        expected_beams = 3 if key.endswith('beam3') else 1
        if report['generation']['num_beams'] != expected_beams:
            raise ValueError(f'Different decoding: {key}')
        ground_truth = {i: [' '.join(tokens(ref)) for ref in row['references']] for i, row in enumerate(rows)}
        hypotheses = {i: [' '.join(tokens(row['caption']))] for i, row in enumerate(rows)}
        score, values = Cider().compute_score(ground_truth, hypotheses)
        bleu = corpus_bleu([[tokens(ref) for ref in row['references']] for row in rows],
                           [tokens(row['caption']) for row in rows], weights=(.25,) * 4)
        if not np.isclose(score, report['metrics']['cider'], rtol=0, atol=1e-10) or not np.isclose(bleu, report['metrics']['bleu4'], rtol=0, atol=1e-10):
            raise ValueError(f'Cannot reproduce recorded scores: {key}')
        per_image[key] = np.asarray(values)
        models.append({'key': key, 'label': label, 'metrics': report['metrics'],
                       'generation': report['generation'], 'validation_images': len(rows),
                       'checkpoint_sha256': report.get('weights_sha256', report.get('model_sha256')),
                       'best_epoch': report.get('best_epoch'), 'completed_epochs': report.get('completed_epochs'),
                       'report': str((base / key / ('results.json' if key in ['efficientnet', 'densenet'] else 'report.json')).resolve())})
    by_key = {model['key']: model for model in models}
    best_cnn = max(['efficientnet', 'densenet'], key=lambda key: by_key[key]['metrics']['cider'])
    gains = {}
    for old, new in [(best_cnn, 'blip-pretrained-greedy'), (best_cnn, 'blip-finetuned-greedy'),
                     ('blip-pretrained-greedy', 'blip-finetuned-greedy'),
                     ('blip-finetuned-greedy', 'blip-finetuned-beam3')]:
        difference = per_image[new] - per_image[old]
        generator = np.random.default_rng(111)
        replicates = np.array([difference[generator.integers(0, len(names), len(names))].mean() for _ in range(2000)])
        old_metrics, new_metrics = by_key[old]['metrics'], by_key[new]['metrics']
        gains[f'{new}_vs_{old}'] = {
            'before': old, 'after': new,
            'cider_absolute': new_metrics['cider'] - old_metrics['cider'],
            'cider_relative_percent': 100 * (new_metrics['cider'] / old_metrics['cider'] - 1),
            'cider_paired_bootstrap_95ci': np.quantile(replicates, [.025, .975]).tolist(),
            'bleu4_absolute': new_metrics['bleu4'] - old_metrics['bleu4'],
            'image_cider_improved_fraction': float((difference > 0).mean()),
            'image_cider_tied_fraction': float((difference == 0).mean()),
        }
    predictions = {key: read(base / key / 'predictions.json') for key in labels}
    paired = [{'image': name, 'references': baseline_rows[i]['references'],
               'captions': {key: predictions[key][i]['caption'] for key in labels},
               'cider': {key: float(per_image[key][i]) for key in labels}}
              for i, name in enumerate(names)]
    examples = [row for row in paired if row['image'] in ['2600386812_8790879d9a.jpg', '476740978_45b65ebe0c.jpg',
                                                        '2486364531_b482d7f521.jpg', '2089555297_95cf001fa7.jpg']]
    report = {'status': 'complete', 'experiment': str(base.resolve()), 'split_sha256': split_hash,
              'train_images': len(split['train']), 'validation_images': len(names),
              'references_per_image': sorted(set(len(row['references']) for row in paired)),
              'metric_protocol': PROTOCOL, 'models': models, 'best_cnn_by_cider': best_cnn,
              'gains': gains, 'examples': examples,
              'bootstrap': {'unit': 'paired validation images', 'resamples': 2000, 'seed': 111,
                            'method': 'percentile CI of per-image CIDEr differences; corpus IDF fixed',
                            'limits': 'Conditional validation-sample uncertainty; excludes training-seed and model-selection variability'},
              'limitations': ['Random 80/20 validation split used for model selection; no independent test set.',
                             'Single training seed. CNNs are new Keras/PyTorch backend replications, not historical checkpoints.',
                             'BLIP changes the full model and adds image/text pretraining; gain is a system comparison, not a CNN-only ablation.',
                             'Original CNN hyperparameters retained; BLIP had four learning-rate trials. Search budgets are unequal.',
                             'Same 24-step generation cap uses native model tokens; CNN word vocabulary and BLIP subword vocabulary differ.'],
              'finished_at': datetime.now(timezone.utc).isoformat()}
    (base / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (base / 'paired-predictions.json').write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding='utf-8')
    title = '# CNN 与 BLIP 统一验证对比结果\n\n'
    intro = f'本轮重训原案例两种 CNN + Transformer，全部采用同一 {len(split["train"]):,}/{len(names):,} 张图像划分和原始参考描述。前四项统一使用贪心解码，最后一项单列 beam=3。所有模型生成上限均为 24 个本模型 token。\n\n'
    table = '| 模型 | CIDEr | BLEU-4 | METEOR | ROUGE-L |\n| --- | ---: | ---: | ---: | ---: |\n'
    for model in models:
        metric = model['metrics']
        table += f'| {model["label"]} | {metric["cider"]:.4f} | {metric["bleu4"]:.4f} | {metric["meteor"]:.4f} | {metric["rouge_l"]:.4f} |\n'
    main_gain = gains[f'blip-finetuned-greedy_vs_{best_cnn}']
    lo, hi = main_gain['cider_paired_bootstrap_95ci']
    finding = f'\n相对本轮 CIDEr 较高的 CNN（{labels[best_cnn]}），微调 BLIP 在相同贪心解码下 CIDEr 增加 {main_gain["cider_absolute"]:.4f}（相对 {main_gain["cider_relative_percent"]:.1f}%），BLEU-4 增加 {main_gain["bleu4_absolute"]:.4f}。图片级配对 bootstrap 的 CIDEr 差值 95% 区间为 [{lo:.4f}, {hi:.4f}]，仅反映当前验证样本的不确定性。\n'
    details = '\n## 实验条件\n\n- 复用原 `fullname_lab.py` 模型构建函数，保留两种官方 Keras ImageNet 主干及其预处理。\n- Keras 3 的 PyTorch CUDA 后端在 Windows GPU 重训，FP32，随机种子 111。原历史权重未迁入，不能称为恢复历史预测。\n- CNN 冻结，Transformer 维度 256，长度 25，训练集拟合词表 7,940，Adam 1e-4、batch 32，最多 30 轮，验证损失 patience 3。\n- CNN 检查点按验证损失选择；BLIP 复用此前验证 CIDEr 选出的学习率 5e-6 最佳检查点。\n- 各模型均独立重载权重并核验原图预测。CIDEr 与 BLEU-4 从逐图预测重新计算并核对报告。\n'
    limitations = '\n## 解释边界\n\n- 这是同一验证集上的系统效果对比，BLIP 同时改变模型结构并引入额外图文预训练数据。\n- 该验证集参与过早停和调参，不是独立测试。仅一个训练种子，BLIP 与 CNN 的参数搜索预算不同。\n- 统一生成步数仍使用各自 tokenizer，不能解释为相同词数。\n- 配对区间不覆盖训练随机性、模型选择偏差和预训练数据重叠风险。\n'
    paths = f'\n逐图对照、模型报告、训练曲线与检查点保存在 `{base.as_posix()}`。划分 SHA256：`{split_hash}`。\n'
    (ROOT / 'CNN_BLIP_COMPARISON_RESULTS.md').write_text(title + intro + table + finding + details + limitations + paths, encoding='utf-8')
    print(table + finding, flush=True)


if __name__ == '__main__':
    main()
