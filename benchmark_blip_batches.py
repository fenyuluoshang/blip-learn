"""Measure equal-objective, equal-effective-batch CUDA training throughput."""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

from blip_loss import grouped_caption_loss
from blip_training_data import read_references
from caption_data import DATA_DIR
from train_blip import decode_forward

ROOT = Path(__file__).resolve().parent


def benchmark(args):
    if args.output.exists():
        raise ValueError('Benchmark output already exists')
    torch.set_num_threads(8)
    torch.manual_seed(111)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    split = json.loads((args.baseline / 'split.json').read_text(encoding='utf-8'))
    config = json.loads((args.baseline / 'config.json').read_text(encoding='utf-8'))
    effective = config['effective_batch_size']
    group_size = config.get('loss_group_size') or config['batch_size']
    _, references, _ = read_references(DATA_DIR)
    names = sorted(split['train'] + split['validation'])
    indices = {name: i for i, name in enumerate(names)}
    cache = np.load(args.baseline / 'vision-features.npy', mmap_mode='r')
    source = ROOT / 'models/blip-base-source-82a3776'
    processor = BlipProcessor.from_pretrained(source, local_files_only=True, use_fast=False)
    model = BlipForConditionalGeneration.from_pretrained(source, local_files_only=True).to('cuda')
    model.vision_model.requires_grad_(False)
    model.train()
    model.vision_model.eval()
    pairs = [(name, text) for name in split['train'] for text in references[name]][:128]
    pairs = (pairs * ((effective + len(pairs) - 1) // len(pairs))) if len(pairs) < effective else pairs
    encoded = processor.tokenizer([text for _, text in pairs], padding='max_length', truncation=True,
                                   max_length=config['max_length'], return_tensors='pt')
    encoded['input_ids'][:, 0] = model.config.text_config.bos_token_id
    labels = encoded['input_ids'].clone().masked_fill(encoded['attention_mask'] == 0, -100)
    features = torch.from_numpy(np.stack([cache[indices[name]] for name, _ in pairs]))
    results = []
    sizes = [size for size in [2, 4, 8, 16] if size % group_size == 0 and effective % size == 0]
    for batch_size in sizes:
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-5, weight_decay=.01)
        scaler = torch.amp.GradScaler('cuda')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        try:
            for update in range(args.warmup + args.updates):
                torch.cuda.synchronize()
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                start = (update * effective) % (len(pairs) - effective + 1)
                for offset in range(0, effective, batch_size):
                    selection = slice(start + offset, start + offset + batch_size)
                    batch = {'encoder_hidden_states': features[selection].to('cuda'),
                             'input_ids': encoded['input_ids'][selection].to('cuda'),
                             'attention_mask': encoded['attention_mask'][selection].to('cuda')}
                    with torch.autocast('cuda', dtype=torch.float16):
                        decoded = decode_forward(model, dict(batch, use_cache=False))
                        loss, _ = grouped_caption_loss(decoded.logits, labels[selection].to('cuda'),
                                                       group_size, model.text_decoder.label_smoothing)
                    scaler.scale(loss * (batch_size / effective)).backward()
                    del decoded, batch, loss
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.)
                if not torch.isfinite(norm).item():
                    raise RuntimeError('Nonfinite benchmark gradients')
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
                if update >= args.warmup:
                    timings.append(time.perf_counter() - started)
            row = {'status': 'passed', 'batch_size': batch_size, 'gradient_accumulation': effective // batch_size,
                   'effective_batch_size': effective, 'loss_group_size': group_size,
                   'median_update_seconds': statistics.median(timings),
                   'caption_pairs_per_second': effective / statistics.median(timings),
                   'peak_allocated_memory_mib': torch.cuda.max_memory_allocated() / 2**20,
                   'measured_updates': len(timings)}
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            row = {'status': 'out_of_memory', 'batch_size': batch_size}
        results.append(row)
        print(json.dumps(row), flush=True)
        del optimizer, scaler
        torch.cuda.empty_cache()
        if row['status'] == 'out_of_memory':
            break
    passed = [row for row in results if row['status'] == 'passed']
    if not passed:
        raise RuntimeError('No CUDA batch configuration passed')
    best = min(passed, key=lambda row: row['median_update_seconds'])
    output = {'status': 'complete', 'selected': best, 'measurements': results,
              'baseline_update_seconds': passed[0]['median_update_seconds'],
              'measured_speedup': passed[0]['median_update_seconds'] / best['median_update_seconds'],
              'split_sha256': config['split_sha256'],
              'protocol': 'same effective batch 16, pairwise token-weighted loss, FP16 autocast, frozen vision cache; warmup excluded'}
    args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--updates', type=int, default=8)
    args = parser.parse_args()
    if args.warmup < 1 or args.updates < 2:
        parser.error('At least one warmup and two measured updates are required')
    benchmark(args)


if __name__ == '__main__':
    main()
