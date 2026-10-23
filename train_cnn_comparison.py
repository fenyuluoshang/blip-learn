"""Retrain the original Keras CNN/Transformer on CUDA and the saved BLIP split.

The original build_model is reused unchanged with a Keras-ops namespace instead
of TensorFlow operations. This is a new backend replication, not recovered
historical weights. ImageNet applications, head architecture, vocabulary rules,
batch size, Adam learning rate, and validation-loss early stopping are retained.
"""
import os
os.environ['KERAS_BACKEND'] = 'torch'

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import keras
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from blip_training_data import manifest_hash, read_references, validate_split
from caption_data import DATA_DIR, load_captions
from caption_metrics import compute_metrics, PROTOCOL

ROOT = Path(__file__).resolve().parent


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def namespace():
    ops = keras.ops
    return SimpleNamespace(
        keras=keras, tile=ops.tile, range=ops.arange, shape=ops.shape,
        not_equal=ops.not_equal, cast=ops.cast, reduce_sum=ops.sum,
        equal=ops.equal, argmax=ops.argmax, float32='float32', int64='int64',
        math=SimpleNamespace(divide_no_nan=ops.divide_no_nan))


def load_builder():
    path = ROOT / '案例5-自动图像描述/案例5-代码和数据/fullname_lab.py'
    spec = importlib.util.spec_from_file_location('original_caption_case', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_model


def original_vocabulary(mapping, names, images):
    counts = Counter(word for name in names for text in mapping[str(images / name)] for word in text.split())
    # TextVectorization resolves equal-frequency tokens in reverse lexical order.
    return ['', '[UNK]'] + sorted(counts, key=lambda word: (counts[word], word), reverse=True)[:9998]


def encode(mapping, names, images, vocabulary, indices):
    lookup = {word: i for i, word in enumerate(vocabulary)}
    rows, ids = [], []
    for name in names:
        for text in mapping[str(images / name)]:
            sequence = [lookup.get(word, 1) for word in text.split()][:25]
            rows.append(sequence + [0] * (25 - len(sequence)))
            ids.append(indices[name])
    return torch.tensor(ids, dtype=torch.long), torch.tensor(rows, dtype=torch.long)


def image_batch(images, names):
    arrays = []
    for name in names:
        with Image.open(images / name) as image:
            # Resize float pixels with the half-pixel bilinear convention used
            # by tf.image.resize, preserving the original [0,255] input range.
            pixels = torch.from_numpy(np.array(image.convert('RGB'), copy=True)).permute(2, 0, 1).float()[None]
        arrays.append(F.interpolate(pixels, size=(224, 224), mode='bilinear', align_corners=False)[0].permute(1, 2, 0))
    return torch.stack(arrays).cuda()


def cache_features(model, images, names, output, batch_size):
    extractor = keras.Model(model.inputs[0], model.get_layer('visual_features').output)
    extractor.eval()
    shape = tuple(extractor.output.shape[1:])
    cache = np.lib.format.open_memmap(output / 'features.npy', mode='w+', dtype='float32', shape=(len(names), *shape))
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(names), batch_size):
            current = names[offset:offset + batch_size]
            cache[offset:offset + len(current)] = extractor(image_batch(images, current), training=False).cpu().numpy()
            if offset % (batch_size * 20) == 0 or offset + len(current) == len(names):
                print(f'CACHE {offset + len(current)}/{len(names)} elapsed={time.perf_counter()-started:.1f}s', flush=True)
                dump(output / 'status.json', {'phase': 'caching', 'cached_images': offset + len(current), 'total_images': len(names)})
    cache.flush()
    dump(output / 'features.json', {'image_names': names, 'dtype': 'float32', 'shape': list(cache.shape)})
    return cache, time.perf_counter() - started


def batch_tensors(cache, image_ids, sequences, rows):
    features = torch.from_numpy(np.array(cache[image_ids[rows].numpy()], copy=True)).cuda()
    sequence = sequences[rows].cuda()
    return features, sequence[:, :-1], sequence[:, 1:]


def masked_loss_accuracy(probabilities, target):
    # Keras sparse CE on probabilities clips to backend epsilon and ignores pad.
    prediction = probabilities / probabilities.sum(-1, keepdim=True)
    loss = F.nll_loss(prediction.clamp(keras.backend.epsilon(), 1 - keras.backend.epsilon()).log().flatten(0, 1), target.flatten(), reduction='none').reshape_as(target)
    mask = target != 0
    count = mask.sum()
    return (loss * mask).sum() / count, ((prediction.argmax(-1) == target) & mask).sum(), count


def epoch_pass(head, cache, encoded, batch_size, optimizer=None, seed=111):
    ids, sequences = encoded
    head.train(optimizer is not None)
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed)) if optimizer is not None else torch.arange(len(ids))
    loss_sum = correct_sum = tokens = 0.
    with torch.set_grad_enabled(optimizer is not None):
        for offset in range(0, len(ids), batch_size):
            features, inputs, targets = batch_tensors(cache, ids, sequences, order[offset:offset + batch_size])
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            probabilities = head([features, inputs], training=optimizer is not None)
            loss, correct, count = masked_loss_accuracy(probabilities, targets)
            if not torch.isfinite(loss).item():
                raise RuntimeError('Nonfinite training loss')
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * count.item()
            correct_sum += correct.item()
            tokens += count.item()
    return {'loss': loss_sum / tokens, 'word_accuracy': correct_sum / tokens}


def generate(head, cache, names, indices, vocabulary, references, batch_size):
    head.eval()
    start, end = vocabulary.index('<start>'), vocabulary.index('<end>')
    result = []
    with torch.inference_mode():
        for offset in range(0, len(names), batch_size):
            current = names[offset:offset + batch_size]
            visual = torch.from_numpy(np.array(cache[[indices[name] for name in current]], copy=True)).cuda()
            inputs = torch.zeros((len(current), 24), dtype=torch.long, device='cuda')
            inputs[:, 0] = start
            finished = torch.zeros(len(current), dtype=torch.bool, device='cuda')
            generated = [[] for _ in current]
            for step in range(24):
                scores = head([visual, inputs], training=False)[:, step, :].clone()
                scores[:, [0, 1, start]] = -1
                next_ids = scores.argmax(-1)
                for i, token in enumerate(next_ids.tolist()):
                    if not finished[i] and token != end:
                        generated[i].append(vocabulary[token])
                finished |= next_ids == end
                if finished.all().item() or step == 23:
                    break
                inputs[:, step + 1] = torch.where(finished, torch.zeros_like(next_ids), next_ids)
            result.extend({'image': name, 'caption': ' '.join(words), 'references': references[name]}
                          for name, words in zip(current, generated))
            if offset % (batch_size * 10) == 0 or len(result) == len(names):
                print(f'GENERATE {len(result)}/{len(names)}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', choices=['efficientnet', 'densenet'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', type=Path, default=ROOT / 'runs/blip-full-random80-20261022/split.json')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--cache-batch-size', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--limit', type=int, help='Preflight only: take this many images from each saved subset')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if min(args.epochs, args.patience, args.batch_size, args.cache_batch_size, args.eval_batch_size) < 1:
        parser.error('Counts must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(111)
    np.random.seed(111)
    torch.manual_seed(111)
    keras.utils.set_random_seed(111)
    images, references, captions_hash = read_references(args.data_dir)
    split = json.loads(args.split.read_text(encoding='utf-8'))
    validate_split(split, references)
    source_split_hash = manifest_hash(split)
    if args.limit:
        split = {key: names[:args.limit] for key, names in split.items()}
    validate_split(split, references)
    mapping = load_captions(images, args.data_dir / 'captions.txt')
    vocabulary = original_vocabulary(mapping, split['train'], images)
    all_names = sorted(split['train'] + split['validation'])
    indices = {name: i for i, name in enumerate(all_names)}
    config = dict(backbone=args.backbone, image_size=224, sequence_length=25, embed_dim=256,
                  train_backbone=False, seed=111, batch_size=args.batch_size, epochs=args.epochs,
                  weights='imagenet', backend='keras-torch-cuda', keras_version=keras.__version__,
                  torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  source_split_sha256=source_split_hash, split_sha256=manifest_hash(split),
                  captions_sha256=captions_hash, learning_rate=1e-4, patience=args.patience,
                  model_selection='minimum validation teacher-forced masked cross entropy',
                  precision='FP32 parameters, features, and arithmetic; TF32 disabled',
                  preflight=bool(args.limit), vocabulary_size=len(vocabulary),
                  optimizer='torch Adam(lr=1e-4, betas=(.9,.999), eps=1e-7)',
                  original_builder_sha256=hashlib.sha256((ROOT / '案例5-自动图像描述/案例5-代码和数据/fullname_lab.py').read_bytes()).hexdigest(),
                  preprocessing='Pillow JPEG RGB decode, float32 half-pixel bilinear resize to 224x224; original Keras application preprocessing',
                  metric_protocol=PROTOCOL)
    dump(args.output / 'config.json', config)
    dump(args.output / 'split.json', split)
    dump(args.output / 'vocabulary.json', vocabulary)
    print(json.dumps(config, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    model = load_builder()(namespace(), config, len(vocabulary), 'imagenet')
    model.cuda()
    print(f'MODEL parameters={model.count_params()} vocabulary={len(vocabulary)}', flush=True)
    if len(vocabulary) == 7940:
        expected_parameters = {'efficientnet': 9774759, 'densenet': 12697156}[args.backbone]
        if model.count_params() != expected_parameters:
            raise RuntimeError('Architecture differs from the original recorded case')
    features = model.get_layer('visual_features').output
    head = keras.Model([features, model.inputs[1]], model.output)
    head.cuda()
    cache, cache_seconds = cache_features(model, images, all_names, args.output, args.cache_batch_size)
    encoded_train = encode(mapping, split['train'], images, vocabulary, indices)
    encoded_valid = encode(mapping, split['validation'], images, vocabulary, indices)
    parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=1e-4, eps=1e-7)
    history = []
    best_loss, best_epoch, wait = float('inf'), None, 0
    for epoch in range(1, args.epochs + 1):
        dump(args.output / 'status.json', {'phase': 'training', 'epoch': epoch, 'max_epochs': args.epochs})
        epoch_started = time.perf_counter()
        train = epoch_pass(head, cache, encoded_train, args.batch_size, optimizer, seed=111 + epoch)
        valid = epoch_pass(head, cache, encoded_valid, args.batch_size)
        row = {'epoch': epoch, 'train_loss': train['loss'], 'train_word_accuracy': train['word_accuracy'],
               'validation_loss': valid['loss'], 'validation_word_accuracy': valid['word_accuracy'],
               'seconds': time.perf_counter() - epoch_started}
        history.append(row)
        dump(args.output / 'history.json', history)
        if valid['loss'] < best_loss:
            best_loss, best_epoch, wait = valid['loss'], epoch, 0
            model.save_weights(args.output / 'best.weights.h5')
        else:
            wait += 1
        print('EPOCH ' + json.dumps({**row, 'best_epoch': best_epoch, 'patience_used': wait}), flush=True)
        if wait >= args.patience:
            break
    model.load_weights(args.output / 'best.weights.h5')
    selection = {'best_epoch': best_epoch, 'completed_epochs': len(history), 'history': history,
                 'parameters': model.count_params(), 'train_images': len(split['train']),
                 'validation_images': len(split['validation']), 'cache_seconds': cache_seconds,
                 'split_sha256': manifest_hash(split), 'status': 'trained'}
    dump(args.output / 'training-results.json', selection)
    dump(args.output / 'status.json', {'phase': 'validation_generation', 'best_epoch': best_epoch})
    rows = generate(head, cache, split['validation'], indices, vocabulary, references, args.eval_batch_size)
    dump(args.output / 'predictions.json', rows)
    metrics = compute_metrics(rows)
    # Restore into a freshly built complete image model; verify saved parameters
    # and raw-image inference, independently of the cached training head.
    restored = load_builder()(namespace(), config, len(vocabulary), 'imagenet')
    restored.cuda()
    restored.load_weights(args.output / 'best.weights.h5')
    restored.eval()
    raw_names = split['validation'][:min(8, len(rows))]
    restored_extractor = keras.Model(restored.inputs[0], restored.get_layer('visual_features').output)
    restored_head = keras.Model([restored.get_layer('visual_features').output, restored.inputs[1]], restored.output)
    with torch.inference_mode():
        raw_features = restored_extractor(image_batch(images, raw_names), training=False).cpu().numpy()
    raw_indices = {name: i for i, name in enumerate(raw_names)}
    raw_rows = generate(restored_head, raw_features, raw_names, raw_indices, vocabulary, references, args.eval_batch_size)
    if [row['caption'] for row in raw_rows] != [row['caption'] for row in rows[:len(raw_rows)]]:
        raise RuntimeError('Independent raw-image reload predictions differ')
    dump(args.output / 'raw-image-predictions.json', raw_rows)
    result = {**selection, 'status': 'complete', 'metrics': metrics, 'metric_protocol': PROTOCOL,
              'generation': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 24,
                             'suppressed_tokens': ['', '[UNK]', '<start>']},
              'independent_raw_image_reload': 'passed', 'raw_image_check_count': len(raw_rows),
              'weights_sha256': hashlib.sha256((args.output / 'best.weights.h5').read_bytes()).hexdigest(),
              'seconds': time.perf_counter() - started, 'finished_at': datetime.now(timezone.utc).isoformat()}
    dump(args.output / 'results.json', result)
    dump(args.output / 'status.json', {'phase': 'complete', 'metrics': metrics, 'best_epoch': best_epoch})
    print('COMPLETE ' + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
