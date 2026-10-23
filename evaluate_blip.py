"""Independent local checkpoint reload and full validation generation."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

from blip_training_data import manifest_hash, read_references, validate_split
from caption_data import DATA_DIR
from caption_metrics import compute_metrics, PROTOCOL


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(111)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    config = json.loads((args.experiment / 'config.json').read_text(encoding='utf-8'))
    split = json.loads((args.experiment / 'split.json').read_text(encoding='utf-8'))
    images, references, caption_hash = read_references(args.data_dir)
    validate_split(split, references)
    if manifest_hash(split) != config['split_sha256'] or caption_hash != config['captions_sha256']:
        raise ValueError('Evaluation data or split differs from training')
    names = split['validation']
    all_names = sorted(split['train'] + split['validation'])
    cache_root = args.feature_cache_dir or args.experiment
    cache_metadata = json.loads((cache_root / 'vision-features.json').read_text(encoding='utf-8'))
    if cache_metadata['signature']['image_names'] != all_names:
        raise ValueError('Evaluation visual cache has the wrong image ordering')
    cache = np.load(cache_root / 'vision-features.npy', mmap_mode='r')
    if list(cache.shape) != cache_metadata['shape'] or cache.dtype != np.dtype('float16'):
        raise ValueError('Invalid evaluation visual cache')
    indices = {name: i for i, name in enumerate(all_names)}
    model_path = args.checkpoint or args.experiment / 'best'
    processor = BlipProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
    model, loading = BlipForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, output_loading_info=True)
    if any(loading.get(key) for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']):
        raise RuntimeError(f'Incomplete checkpoint restoration: {loading}')
    model.to('cuda').eval()
    if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise RuntimeError('Nonfinite checkpoint parameters')
    generation = {'do_sample': False, 'num_beams': args.num_beams,
                  'max_new_tokens': args.max_new_tokens or config['max_new_tokens'], 'length_penalty': 1.0}
    rows = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(names), args.batch_size):
            batch_names = names[offset:offset + args.batch_size]
            features = torch.from_numpy(np.stack([cache[indices[name]] for name in batch_names])).to('cuda')
            with torch.autocast('cuda', dtype=torch.float16):
                ids = model.text_decoder.generate(
                    input_ids=torch.full((len(batch_names), 1), model.config.text_config.bos_token_id,
                                         dtype=torch.long, device='cuda'),
                    encoder_hidden_states=features,
                    encoder_attention_mask=torch.ones(features.shape[:2], dtype=torch.long, device='cuda'),
                    eos_token_id=model.config.text_config.sep_token_id,
                    pad_token_id=model.config.text_config.pad_token_id, **generation)
            texts = processor.batch_decode(ids, skip_special_tokens=True)
            if not all(text.strip() for text in texts):
                raise RuntimeError('Empty generated caption')
            rows.extend({'image': name, 'caption': text, 'references': references[name]}
                        for name, text in zip(batch_names, texts))
            if len(rows) % (args.batch_size * 25) == 0 or len(rows) == len(names):
                print(f'Independent evaluation {len(rows)}/{len(names)}; beams={args.num_beams}', flush=True)
    dump(args.output / 'predictions.json', rows)
    matched = None
    if args.expected_predictions:
        expected = json.loads(args.expected_predictions.read_text(encoding='utf-8'))
        if [(row['image'], row['caption'], row['references']) for row in rows] != [
                (row['image'], row['caption'], row['references']) for row in expected]:
            raise RuntimeError('Independent checkpoint predictions differ from saved training validation predictions')
        matched = True
    metrics = compute_metrics(rows)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('Nonfinite evaluation metrics')
    raw_rows = []
    with torch.inference_mode():
        for name in names[:args.raw_check_count]:
            with Image.open(images / name) as image:
                inputs = processor(images=image.convert('RGB'), return_tensors='pt').to('cuda')
            with torch.autocast('cuda', dtype=torch.float16):
                ids = model.generate(**inputs, **generation)
            caption = processor.decode(ids[0], skip_special_tokens=True)
            if not caption.strip():
                raise RuntimeError('Raw image inference failed')
            raw_rows.append({'image': name, 'caption': caption, 'references': references[name]})
    dump(args.output / 'raw_image_predictions.json', raw_rows)
    cached_by_name = {row['image']: row['caption'] for row in rows}
    raw_predictions_match = all(row['caption'] == cached_by_name[row['image']] for row in raw_rows)
    weight_file = model_path / 'model.safetensors'
    if not weight_file.exists():
        weight_file = model_path / 'pytorch_model.bin'
    with weight_file.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = {'status': 'complete', 'experiment': str(args.experiment.resolve()),
              'checkpoint': str(model_path.resolve()), 'model_sha256': checksum,
              'split_sha256': config['split_sha256'], 'validation_images': len(rows),
              'generation': generation, 'metrics': metrics, 'metric_protocol': PROTOCOL,
              'independent_reload': 'passed', 'training_predictions_match': matched,
              'cached_and_raw_image_predictions_match': raw_predictions_match,
              'raw_image_inference_count': len(raw_rows), 'seconds': time.perf_counter() - started,
              'peak_allocated_memory_mib': torch.cuda.max_memory_allocated() / 2**20,
              'finished_at': datetime.now(timezone(timedelta(hours=8))).isoformat()}
    dump(args.output / 'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path,
                        help='Optional local checkpoint, e.g. the frozen pretrained source')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feature-cache-dir', type=Path)
    parser.add_argument('--expected-predictions', type=Path)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int,
                        help='Optional common generation budget for baseline comparisons')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--raw-check-count', type=int, default=8)
    args = parser.parse_args()
    if args.num_beams < 1 or args.batch_size < 1 or args.raw_check_count < 1:
        parser.error('Beam count, batch size and raw image count must be positive')
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error('max-new-tokens must be positive')
    evaluate(args)


if __name__ == '__main__':
    main()
