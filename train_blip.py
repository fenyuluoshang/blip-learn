"""BLIP frozen-vision fine-tuning with cached features and resumable state."""
import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BlipForConditionalGeneration, BlipProcessor, get_cosine_schedule_with_warmup

from blip_training_data import make_split, manifest_hash, read_references, validate_split
from blip_loss import grouped_caption_loss, accumulation_weight
from blip_amp import amp_optimizer_step
from caption_data import DATA_DIR
from caption_metrics import compute_metrics, PROTOCOL
from verify_blip_environment import MODEL_ID, REVISION, WEIGHTS_SHA256

ROOT = Path(__file__).resolve().parent
TZ = timezone(timedelta(hours=8))


def now():
    return datetime.now(TZ).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def status(output, **values):
    path = output / 'status.json'
    previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    previous.update(values, updated_at=now(), pid=os.getpid())
    atomic_json(path, previous)


class ImageDataset(Dataset):
    def __init__(self, images, names, processor):
        self.images, self.names, self.processor = images, names, processor

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        with Image.open(self.images / self.names[index]) as image:
            return self.processor(images=image.convert('RGB'), return_tensors='pt')['pixel_values'][0]


class CaptionDataset(Dataset):
    def __init__(self, features_path, pairs, image_indices, tokenizer, max_length, bos_id):
        self.features_path, self.features = str(features_path), None
        self.image_indices = [image_indices[name] for name, _ in pairs]
        encoded = tokenizer([text for _, text in pairs], padding='max_length', truncation=True,
                            max_length=max_length, return_tensors='pt')
        self.ids, self.masks = encoded['input_ids'], encoded['attention_mask']
        self.ids[:, 0] = bos_id
        self.labels = self.ids.clone().masked_fill(self.masks == 0, -100)

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, index):
        if self.features is None:
            self.features = np.load(self.features_path, mmap_mode='r')
        feature = torch.from_numpy(self.features[self.image_indices[index]].copy())
        return {'encoder_hidden_states': feature, 'input_ids': self.ids[index],
                'attention_mask': self.masks[index], 'labels': self.labels[index]}


def loader(dataset, batch_size, workers, sampler=None):
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers, sampler=sampler,
                      pin_memory=True, persistent_workers=False, drop_last=False)


def ensure_cache(model, processor, images, names, output, args):
    cache_root = args.feature_cache_dir.resolve() if args.feature_cache_dir else output
    path, meta_path = cache_root / 'vision-features.npy', cache_root / 'vision-features.json'
    signature = {'image_names': names, 'model_revision': REVISION, 'dtype': 'float16',
                 'extraction_precision': 'float16 autocast; final features cast to float16',
                 'image_size': model.config.vision_config.image_size}
    if path.exists() and meta_path.exists():
        recorded = json.loads(meta_path.read_text(encoding='utf-8'))
        existing = np.load(path, mmap_mode='r')
        if (recorded['signature'] != signature or list(existing.shape) != recorded['shape']
                or existing.dtype != np.dtype('float16')):
            raise ValueError('Cached visual features differ from the requested data/model')
        status(output, feature_cache=str(path), reused_visual_cache=True)
        return path
    if args.feature_cache_dir:
        raise ValueError('Shared visual cache must already be complete; original run is still its owner')
    model.eval()
    dataset = ImageDataset(images, names, processor)
    cache = None
    offset = 0
    status(output, phase='caching_visual_features', cached_images=0, cache_total=len(names))
    with torch.inference_mode():
        for pixels in loader(dataset, args.cache_batch_size, args.workers):
            with torch.autocast('cuda', dtype=torch.float16):
                features = model.vision_model(pixel_values=pixels.to('cuda', non_blocking=True)).last_hidden_state
            values = features.to(torch.float16).cpu().numpy()
            if cache is None:
                cache = np.lib.format.open_memmap(path, mode='w+', dtype='float16',
                                                  shape=(len(names), *values.shape[1:]))
            cache[offset:offset + len(values)] = values
            offset += len(values)
            if offset % (args.cache_batch_size * 20) == 0 or offset == len(names):
                print(f'Visual features {offset}/{len(names)} images', flush=True)
                status(output, cached_images=offset)
    cache.flush()
    atomic_json(meta_path, {'signature': signature, 'shape': list(cache.shape)})
    del cache
    return path


def to_cuda(batch):
    return {key: value.to('cuda', non_blocking=True) for key, value in batch.items()}


def decode_forward(model, batch):
    return model.text_decoder(**batch, encoder_attention_mask=torch.ones(
        batch['encoder_hidden_states'].shape[:2], dtype=torch.long, device='cuda'), return_dict=True)


def validate(model, processor, valid_data, features_path, names, image_indices, references,
             output, args, epoch):
    model.eval()
    status(output, phase='validation_loss', evaluation_epoch=epoch)
    loss_sum, token_count = 0., 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader(valid_data, args.eval_batch_size, args.workers):
            batch = to_cuda(batch)
            count = (batch['labels'][:, 1:] != -100).sum().item()
            with torch.autocast('cuda', dtype=torch.float16):
                loss = decode_forward(model, batch).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError('Nonfinite validation loss')
            loss_sum += loss.item() * count
            token_count += count
    rows = []
    cache = np.load(features_path, mmap_mode='r')
    status(output, phase='validation_generation', generated_images=0, generation_total=len(names))
    with torch.inference_mode():
        for offset in range(0, len(names), args.eval_batch_size):
            batch_names = names[offset:offset + args.eval_batch_size]
            features = torch.from_numpy(np.stack([cache[image_indices[name]] for name in batch_names])).to('cuda')
            input_ids = torch.full((len(batch_names), 1), model.config.text_config.bos_token_id,
                                   dtype=torch.long, device='cuda')
            with torch.autocast('cuda', dtype=torch.float16):
                ids = model.text_decoder.generate(
                    input_ids=input_ids, encoder_hidden_states=features,
                    encoder_attention_mask=torch.ones(features.shape[:2], dtype=torch.long, device='cuda'),
                    eos_token_id=model.config.text_config.sep_token_id,
                    pad_token_id=model.config.text_config.pad_token_id,
                    do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens)
            texts = processor.batch_decode(ids, skip_special_tokens=True)
            rows.extend({'image': name, 'caption': caption, 'references': references[name]}
                        for name, caption in zip(batch_names, texts))
            if len(rows) % (args.eval_batch_size * 25) == 0 or len(rows) == len(names):
                status(output, generated_images=len(rows))
    prediction_dir = output / 'predictions'
    prediction_dir.mkdir(exist_ok=True)
    atomic_json(prediction_dir / f'epoch-{epoch:02d}.json', rows)
    status(output, phase='validation_metrics')
    metrics = compute_metrics(rows)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('Nonfinite generation metric')
    metrics.update(validation_loss=loss_sum / token_count, validation_images=len(names),
                   validation_seconds=time.perf_counter() - started)
    return metrics


def rotate_checkpoint(output, name, write):
    destination, temporary, previous = output / name, output / (name + '.new'), output / (name + '.previous')
    for path in [temporary, previous]:
        if path.resolve().parent != output.resolve():
            raise ValueError('Checkpoint path must remain within experiment directory')
        if path.exists():
            shutil.rmtree(path)
    temporary.mkdir()
    write(temporary)
    if destination.exists():
        destination.rename(previous)
    temporary.rename(destination)
    if previous.exists():
        shutil.rmtree(previous)


def save_last(output, model, processor, optimizer, scheduler, scaler, state, config):
    def write(path):
        model.save_pretrained(path, safe_serialization=True)
        processor.save_pretrained(path)
        atomic_json(path / 'progress.json', state)
        torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(), 'state': state, 'config': config,
                    'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()},
                   path / 'training_state.pt')
    rotate_checkpoint(output, 'last', write)
    status(output, last_checkpoint_at=now(), last_checkpoint_step=state['global_step'])


def save_best(output, model, processor, metrics, epoch):
    def write(path):
        model.save_pretrained(path, safe_serialization=True)
        processor.save_pretrained(path)
        atomic_json(path / 'selection.json', {'epoch': epoch, 'metrics': metrics, 'selection': 'max validation CIDEr'})
    rotate_checkpoint(output, 'best', write)


def training(args):
    output = args.output.resolve()
    if args.resume:
        if not (output / 'last/training_state.pt').exists():
            raise ValueError('Resume requires last/training_state.pt')
    else:
        output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(8)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    images, references, captions_hash = read_references(args.data_dir)
    split_path = output / 'split.json'
    if args.resume:
        split = json.loads(split_path.read_text(encoding='utf-8'))
        validate_split(split, references)
    else:
        split = make_split(references, args.seed, args.validation_fraction, args.limit)
        atomic_json(split_path, split)
    source = json.loads((args.model_dir / 'source.json').read_text(encoding='utf-8'))
    if source['model_id'] != MODEL_ID or source['revision'] != REVISION:
        raise ValueError('Unexpected source checkpoint revision')
    with (args.model_dir / 'pytorch_model.bin').open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != WEIGHTS_SHA256:
            raise ValueError('Source checkpoint checksum mismatch')
    config = {key: value for key, value in vars(args).items()
              if key not in ['output', 'resume', 'stop_after_epoch', 'data_dir', 'model_dir', 'feature_cache_dir']
              and not (key == 'loss_group_size' and value is None)}
    config.update(model_id=MODEL_ID, model_revision=REVISION, source_weights_sha256=WEIGHTS_SHA256,
                  captions_sha256=captions_hash, split_sha256=manifest_hash(split),
                  train_images=len(split['train']), validation_images=len(split['validation']),
                  train_pairs=sum(len(references[name]) for name in split['train']),
                  validation_pairs=sum(len(references[name]) for name in split['validation']),
                  frozen_module='vision_model', precision='FP32 parameters, FP16 autocast and cached visual features',
                  effective_batch_size=args.batch_size * args.gradient_accumulation,
                  model_selection='max validation CIDEr',
                  split_protocol='image-level random 80/20 per user; no independent test set',
                  metric_protocol=PROTOCOL)
    if args.resume:
        if config != json.loads((output / 'config.json').read_text(encoding='utf-8')):
            raise ValueError('Resume configuration or dataset differs; use the original CLI options')
    else:
        atomic_json(output / 'config.json', config)
        snapshot = output / 'source'
        snapshot.mkdir()
        for filename in ['train_blip.py', 'blip_training_data.py', 'blip_loss.py', 'blip_amp.py', 'caption_metrics.py',
                         'caption_data.py', 'verify_blip_environment.py', 'requirements-blip-lock.txt']:
            shutil.copy2(ROOT / filename, snapshot / filename)
        atomic_json(output / 'environment.json', {
            'started_at': now(), 'python': sys.version, 'executable': sys.executable,
            'system': platform.platform(), 'gpu': torch.cuda.get_device_name(0),
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'packages': {name: importlib.metadata.version(name) for name in
                         ['torch', 'torchvision', 'transformers', 'numpy', 'Pillow', 'nltk', 'rouge-score', 'pycocoevalcap']}})
    print(f'Train: {config["train_images"]} images / {config["train_pairs"]} captions; '
          f'validation: {config["validation_images"]} images / {config["validation_pairs"]} captions', flush=True)
    status(output, status='running', phase='loading_model', error=None, train_images=config['train_images'],
           validation_images=config['validation_images'], config=str(output / 'config.json'))
    model_path = output / 'last' if args.resume else args.model_dir
    processor = BlipProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
    model = BlipForConditionalGeneration.from_pretrained(model_path, local_files_only=True).to('cuda')
    model.vision_model.requires_grad_(False)
    names = sorted(split['train'] + split['validation'])
    image_indices = {name: index for index, name in enumerate(names)}
    features_path = ensure_cache(model, processor, images, names, output, args)
    torch.cuda.empty_cache()
    train_pairs = [(name, text) for name in split['train'] for text in references[name]]
    valid_pairs = [(name, text) for name in split['validation'] for text in references[name]]
    train_data = CaptionDataset(features_path, train_pairs, image_indices, processor.tokenizer,
                                args.max_length, model.config.text_config.bos_token_id)
    valid_data = CaptionDataset(features_path, valid_pairs, image_indices, processor.tokenizer,
                                args.max_length, model.config.text_config.bos_token_id)
    micro_batches = math.ceil(len(train_data) / args.batch_size)
    updates_per_epoch = math.ceil(micro_batches / args.gradient_accumulation)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate,
                                 weight_decay=.01)
    total_steps = updates_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * .05), total_steps)
    scaler = torch.amp.GradScaler('cuda')
    state = {'epoch': 0, 'next_batch': 0, 'global_step': 0, 'best_cider': -1., 'best_epoch': None,
             'bad_epochs': 0, 'history': [], 'train_loss_sum': 0., 'train_tokens': 0,
             'split_sha256': config['split_sha256']}
    if args.resume:
        saved = torch.load(output / 'last/training_state.pt', map_location='cpu', weights_only=False)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        state = saved['state']
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved
        atomic_json(output / 'history.json', state['history'])
        print(f'Resumed epoch {state["epoch"] + 1}, batch {state["next_batch"]}, step {state["global_step"]}', flush=True)
    else:
        initial = validate(model, processor, valid_data, features_path, split['validation'], image_indices,
                           references, output, args, 0)
        initial.update(epoch=0, kind='pretrained_zero_shot', recorded_at=now())
        state['history'].append(initial)
        state['best_cider'], state['best_epoch'] = initial['cider'], 0
        atomic_json(output / 'history.json', state['history'])
        save_best(output, model, processor, initial, 0)
        save_last(output, model, processor, optimizer, scheduler, scaler, state, config)
        print(f'Zero-shot validation CIDEr={initial["cider"]:.4f}', flush=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    all_started = time.perf_counter()
    for epoch in range(state['epoch'], args.epochs):
        if state['bad_epochs'] >= args.patience:
            break
        permutation = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        next_batch = state['next_batch']
        remaining = permutation[next_batch * args.batch_size:]
        # Constructing a DataLoader must not advance the model's dropout RNG on resume.
        rng = torch.get_rng_state()
        batches = iter(loader(train_data, args.batch_size, args.workers, remaining))
        torch.set_rng_state(rng)
        model.train()
        model.vision_model.eval()
        optimizer.zero_grad(set_to_none=True)
        status(output, phase='training', epoch=epoch + 1, epochs=args.epochs,
               batch=next_batch, batches_per_epoch=micro_batches, global_step=state['global_step'])
        started = time.perf_counter()
        for batch_index, batch in enumerate(batches, start=next_batch):
            batch = to_cuda(batch)
            count = (batch['labels'][:, 1:] != -100).sum().item()
            window_start = (batch_index // args.gradient_accumulation) * args.gradient_accumulation
            divisor = min(args.gradient_accumulation, micro_batches - window_start)
            with torch.autocast('cuda', dtype=torch.float16):
                if args.loss_group_size:
                    decoded = decode_forward(model, dict(batch, labels=None, use_cache=False))
                    loss, token_loss_sum = grouped_caption_loss(
                        decoded.logits, batch['labels'], args.loss_group_size, model.text_decoder.label_smoothing)
                    weight = accumulation_weight(len(train_data), batch_index, args.batch_size,
                                                 args.gradient_accumulation, args.loss_group_size)
                else:
                    loss = decode_forward(model, batch).loss
                    token_loss_sum = loss.detach() * count
                    weight = 1 / divisor
            if not torch.isfinite(loss).item():
                raise RuntimeError(f'Nonfinite loss at epoch {epoch + 1} batch {batch_index}')
            scaler.scale(loss * weight).backward()
            state['train_loss_sum'] += token_loss_sum.detach().item()
            state['train_tokens'] += count
            if args.loss_group_size:
                del decoded, token_loss_sum
            state['next_batch'] = batch_index + 1
            if (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == micro_batches:
                updated, norm = amp_optimizer_step(optimizer, scaler,
                                                  (p for p in model.parameters() if p.requires_grad))
                if not updated:
                    state['overflow_skipped_updates'] = state.get('overflow_skipped_updates', 0) + 1
                    state['consecutive_gradient_overflows'] = state.get('consecutive_gradient_overflows', 0) + 1
                    status(output, overflow_skipped_updates=state['overflow_skipped_updates'],
                           loss_scale=scaler.get_scale())
                    print(f'AMP overflow: skipped update at epoch {epoch + 1}, batch {batch_index + 1}; '
                          f'loss scale reduced to {scaler.get_scale()}', flush=True)
                    if state['consecutive_gradient_overflows'] >= 8:
                        raise RuntimeError('Eight consecutive nonfinite gradient windows after AMP scale backoff')
                    continue
                state['consecutive_gradient_overflows'] = 0
                scheduler.step()
                state['global_step'] += 1
                if state['global_step'] % args.log_every == 0 or batch_index + 1 == micro_batches:
                    average = state['train_loss_sum'] / state['train_tokens']
                    current = {'epoch': epoch + 1, 'batch': batch_index + 1, 'global_step': state['global_step'],
                               'train_loss': average, 'learning_rate': scheduler.get_last_lr()[0],
                               'peak_allocated_memory_mib': torch.cuda.max_memory_allocated() / 2**20,
                               'elapsed_segment_seconds': time.perf_counter() - started}
                    status(output, phase='training', **current)
                    print(f'Epoch {epoch + 1}/{args.epochs} batch {batch_index + 1}/{micro_batches} '
                          f'step {state["global_step"]} loss={average:.4f}', flush=True)
                if state['global_step'] % args.checkpoint_every == 0:
                    save_last(output, model, processor, optimizer, scheduler, scaler, state, config)
        epoch_seconds = time.perf_counter() - started
        metrics = validate(model, processor, valid_data, features_path, split['validation'], image_indices,
                           references, output, args, epoch + 1)
        metrics.update(epoch=epoch + 1, train_loss=state['train_loss_sum'] / state['train_tokens'],
                       train_seconds=epoch_seconds, global_step=state['global_step'], recorded_at=now())
        state['history'].append(metrics)
        improved = metrics['cider'] > state['best_cider']
        if improved:
            state['best_cider'], state['best_epoch'], state['bad_epochs'] = metrics['cider'], epoch + 1, 0
            save_best(output, model, processor, metrics, epoch + 1)
        else:
            state['bad_epochs'] += 1
        state.update(epoch=epoch + 1, next_batch=0, train_loss_sum=0., train_tokens=0)
        atomic_json(output / 'history.json', state['history'])
        save_last(output, model, processor, optimizer, scheduler, scaler, state, config)
        status(output, epoch=epoch + 1, best_epoch=state['best_epoch'], best_cider=state['best_cider'],
               last_metrics=metrics, bad_epochs=state['bad_epochs'])
        print(f'Epoch {epoch + 1} complete: validation loss={metrics["validation_loss"]:.4f} '
              f'CIDEr={metrics["cider"]:.4f} best_epoch={state["best_epoch"]}', flush=True)
        if args.stop_after_epoch and epoch + 1 >= args.stop_after_epoch:
            status(output, status='stopped_at_requested_epoch', phase='checkpoint_saved')
            return
    atomic_json(output / 'results.json', {'status': 'complete', 'best_epoch': state['best_epoch'],
                                         'overflow_skipped_updates': state.get('overflow_skipped_updates', 0),
                                         'best_cider': state['best_cider'], 'completed_epochs': state['epoch'],
                                         'early_stopped': state['bad_epochs'] >= args.patience,
                                         'history': state['history'],
                                         'peak_allocated_memory_mib': torch.cuda.max_memory_allocated() / 2**20,
                                         'peak_reserved_memory_mib': torch.cuda.max_memory_reserved() / 2**20,
                                         'process_training_seconds': time.perf_counter() - all_started})
    status(output, status='complete', phase='finished', best_epoch=state['best_epoch'], best_cider=state['best_cider'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models/blip-base-source-82a3776')
    parser.add_argument('--feature-cache-dir', type=Path, help='Reuse a completed frozen-vision cache from another experiment')
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--validation-fraction', type=float, default=.2)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--gradient-accumulation', type=int, default=4)
    parser.add_argument('--loss-group-size', type=int, help='Preserve original microbatch token weighting inside larger physical batches')
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--cache-batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--max-length', type=int, default=32)
    parser.add_argument('--max-new-tokens', type=int, default=30)
    parser.add_argument('--checkpoint-every', type=int, default=250)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for key in ['epochs', 'patience', 'batch_size', 'gradient_accumulation', 'eval_batch_size',
                'cache_batch_size', 'max_length', 'max_new_tokens', 'checkpoint_every', 'log_every']:
        if getattr(args, key) < 1:
            parser.error(f'{key} must be positive')
    if args.workers < 0 or args.learning_rate <= 0:
        parser.error('Workers must be nonnegative and learning rate positive')
    if args.loss_group_size is not None and (args.loss_group_size < 1 or args.batch_size % args.loss_group_size):
        parser.error('Loss group size must be positive and divide the physical batch size')
    existing = args.output.resolve().exists()
    if existing and not args.resume:
        parser.error('Output exists; choose a new directory or use --resume')
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        training(args)
    except Exception:
        if args.output.exists() and (not existing or args.resume):
            status(args.output, status='failed', phase='error', error=traceback.format_exc())
        raise
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
