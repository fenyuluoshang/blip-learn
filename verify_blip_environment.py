"""Small Windows/CUDA/BLIP integration check, not a quality benchmark."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
import zipfile

from caption_data import DATA_DIR, prepare

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'Salesforce/blip-image-captioning-base'
REVISION = '82a37760796d32b1411fe092ab5d4e227313294b'
WEIGHTS_SHA256 = 'd6638651a5526cc2ede56f2b5104d6851b0755816d220e5e046870430180c767'
LOCAL_TZ = timezone(timedelta(hours=8))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def sample_data(data_dir):
    images, captions_file = prepare(data_dir)
    captions = {}
    with captions_file.open(encoding='utf-8-sig', newline='') as stream:
        # Keep raw references; do not use the baseline's cleaned/truncated text.
        for row in csv.DictReader(stream):
            captions.setdefault(row['image'], []).append(row['caption'])
    splits, hashes = {}, {}
    with zipfile.ZipFile(data_dir / 'Flickr8k_text.zip') as archive:
        for label, filename in [('train', 'Flickr_8k.trainImages.txt'),
                                ('validation', 'Flickr_8k.devImages.txt'),
                                ('test', 'Flickr_8k.testImages.txt')]:
            raw = archive.read(filename)
            names = [name.strip() for name in raw.decode('utf-8-sig').splitlines() if name.strip()]
            if len(names) != len(set(names)):
                raise ValueError(f'Duplicate image in {filename}')
            for name in names:
                if Path(name).name != name or not (images / name).is_file() or name not in captions:
                    raise ValueError(f'Missing/invalid image or references: {name}')
            splits[label] = names
            hashes[label] = hashlib.sha256(raw).hexdigest()
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if set(splits[left]) & set(splits[right]):
            raise ValueError(f'Overlapping {left}/{right} images')
    selected = {'train': sorted(splits['train'])[:2],
                'validation': sorted(splits['validation'])[:3]}
    inventory = {
        'images_with_captions': len(captions),
        'caption_rows': sum(map(len, captions.values())),
        'official_counts': {label: len(names) for label, names in splits.items()},
        'official_manifest_sha256': hashes,
        'images_outside_official_split': len(set(captions) - set.union(*(set(names) for names in splits.values()))),
        'selected_images': selected,
        'selection_rule': 'First sorted filenames; two official train images and three official validation images.',
        'test_images_used': 0,
    }
    return images, captions, selected, inventory


def generate(model, processor, images, names, torch):
    from PIL import Image
    pixels = []
    for name in names:
        with Image.open(images / name) as source:
            pixels.append(source.convert('RGB'))
    inputs = processor(images=pixels, return_tensors='pt').to('cuda')
    model.eval()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        sequences = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=30)
    torch.cuda.synchronize()
    captions = processor.batch_decode(sequences, skip_special_tokens=True)
    if not all(text.strip() for text in captions):
        raise RuntimeError('Empty generated caption')
    return captions, time.perf_counter() - started


def reload_check(checkpoint, predictions_path, data_dir):
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor
    saved = json.loads(predictions_path.read_text(encoding='utf-8'))
    processor = BlipProcessor.from_pretrained(checkpoint, local_files_only=True, use_fast=False)
    model = BlipForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True).to('cuda')
    images = data_dir / 'Flicker8k_Dataset'
    actual, elapsed = generate(model, processor, images, [row['image'] for row in saved], torch)
    expected = [row['caption'] for row in saved]
    if actual != expected:
        raise RuntimeError(f'Saved model predictions changed: {actual!r} != {expected!r}')
    print(json.dumps({'status': 'passed', 'captions_match': True, 'images': len(actual),
                      'generation_seconds': elapsed}))


def run(args, output, report):
    import numpy as np
    import torch
    import torchvision
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor

    random.seed(111)
    np.random.seed(111)
    torch.manual_seed(111)
    torch.cuda.manual_seed_all(111)
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install the CUDA PyTorch wheels, not CPU wheels.')
    device = torch.cuda.get_device_properties(0)
    report['environment'] = {
        'timestamp': datetime.now(LOCAL_TZ).isoformat(), 'system': platform.platform(),
        'python': sys.version.split()[0], 'executable': sys.executable,
        'packages': {name: importlib.metadata.version(name) for name in
                     ['torch', 'torchvision', 'transformers', 'Pillow', 'numpy', 'safetensors', 'huggingface-hub']},
        'cuda_available': True, 'torch_cuda_runtime': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(), 'gpu': device.name,
        'gpu_total_memory_mib': device.total_memory / 2**20,
        'cuda_architectures': torch.cuda.get_arch_list(),
    }
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
    report['environment']['nvidia_smi'] = gpu.stdout.strip()
    matrix = torch.randn(512, 512, device='cuda', requires_grad=True)
    (matrix @ matrix.T).square().mean().backward()
    if not torch.isfinite(matrix.grad).all().item():
        raise RuntimeError('CUDA matrix multiplication/backward produced nonfinite gradients')
    # torchvision's compiled CUDA extension is needed by common image pipelines.
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device='cuda')
    keep = torchvision.ops.nms(boxes, torch.tensor([.9, .8], device='cuda'), .5)
    if keep.tolist() != [0]:
        raise RuntimeError('torchvision CUDA extension failed')
    del matrix, boxes, keep
    report['gpu_operations'] = {'status': 'passed', 'matmul_backward': True, 'torchvision_cuda_nms': True}
    print('CUDA matrix/backward and torchvision extension: passed', flush=True)
    write_json(output / 'report.json', report)
    if args.gpu_only:
        return

    images, references, selected, inventory = sample_data(args.data_dir)
    report['data'] = inventory
    write_json(output / 'sample_manifest.json', inventory)
    report['checkpoint_source'] = {
        'model_id': MODEL_ID, 'revision': REVISION, 'license_from_model_card': 'bsd-3-clause',
        'model_card': f'https://huggingface.co/{MODEL_ID}/blob/{REVISION}/README.md',
        'model_card_training_description': 'Image captioning pretrained on COCO; ViT base backbone.',
        'pretraining_overlap_with_flickr8k': 'Not established; no decontamination claim.',
    }
    print(f'Loading BLIP checkpoint at pinned revision {REVISION}', flush=True)
    if args.model_dir:
        source = args.model_dir.resolve()
        metadata = json.loads((source / 'source.json').read_text(encoding='utf-8'))
        if metadata['model_id'] != MODEL_ID or metadata['revision'] != REVISION:
            raise RuntimeError('Local model source does not match the pinned checkpoint')
        weights_path = source / 'pytorch_model.bin'
        load_options = {'local_files_only': True}
    else:
        from huggingface_hub import hf_hub_download
        weights_path = Path(hf_hub_download(MODEL_ID, 'pytorch_model.bin', revision=REVISION,
                                            local_files_only=args.local_files_only))
        source = MODEL_ID
        load_options = {'revision': REVISION, 'local_files_only': args.local_files_only}
    with weights_path.open('rb') as stream:
        actual_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual_hash != WEIGHTS_SHA256:
        raise RuntimeError('Downloaded BLIP checkpoint SHA256 differs from its pinned LFS metadata')
    report['checkpoint_source']['weights_sha256'] = actual_hash
    processor = BlipProcessor.from_pretrained(source, **load_options, use_fast=False)
    model = BlipForConditionalGeneration.from_pretrained(source, **load_options).to('cuda')
    model.vision_model.requires_grad_(False)
    report['model'] = {'parameters': sum(p.numel() for p in model.parameters()),
                       'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                       'image_size': model.config.vision_config.image_size, 'seed': 111,
                       'precision': 'FP32 parameters, FP16 CUDA autocast, GradScaler',
                       'frozen_module': 'vision_model'}
    torch.cuda.reset_peak_memory_stats()
    zero_captions, zero_seconds = generate(model, processor, images, selected['validation'], torch)
    rows = [{'image': name, 'caption': text, 'references': references[name]}
            for name, text in zip(selected['validation'], zero_captions)]
    write_json(output / 'zero_shot_predictions.json', rows)
    report['zero_shot'] = {'images': len(rows), 'batch_size': len(rows), 'seconds': zero_seconds,
                           'generation': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 30}}
    print('Three-image zero-shot inference: passed', flush=True)
    write_json(output / 'report.json', report)

    pairs = [(name, caption) for name in selected['train'] for caption in references[name]]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    tracked = model.text_decoder.bert.embeddings.word_embeddings.weight
    before = tracked.detach().clone()
    losses, gradients = [], []
    model.train()
    model.vision_model.eval()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for offset in range(0, len(pairs), 2):
        batch = pairs[offset:offset + 2]
        batch_images = []
        for name, _ in batch:
            with Image.open(images / name) as source:
                batch_images.append(source.convert('RGB'))
        inputs = processor(images=batch_images, text=[caption for _, caption in batch],
                           padding='max_length', truncation=True, max_length=32, return_tensors='pt').to('cuda')
        # BLIP's caption decoder starts with its own [DEC] token, not BERT [CLS].
        inputs['input_ids'][:, 0] = model.config.text_config.bos_token_id
        labels = inputs['input_ids'].clone()
        labels.masked_fill_(inputs['attention_mask'] == 0, -100)
        if not (labels == -100).any().item():
            raise RuntimeError('The smoke batch did not exercise label padding masking')
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = model(**inputs, labels=labels).loss
        if not torch.isfinite(loss).item():
            raise RuntimeError('BLIP training loss is not finite')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.)
        if not torch.isfinite(gradient).item() or gradient.item() == 0:
            raise RuntimeError('BLIP gradients are nonfinite or zero')
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        gradients.append(gradient.item())
        print(f'Smoke optimizer step {len(losses)}: loss={loss.item():.4f}', flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    delta = (tracked.detach() - before).abs().max().item()
    if not math.isfinite(delta) or delta <= 0:
        raise RuntimeError('Trainable decoder weights did not change')
    if any(p.grad is not None for p in model.vision_model.parameters()):
        raise RuntimeError('Frozen vision encoder unexpectedly received gradients')
    report['training_smoke'] = {
        'images': len(selected['train']), 'caption_pairs': len(pairs), 'batch_size': 2,
        'optimizer_steps': len(losses), 'learning_rate': 1e-5,
        'losses': losses, 'gradient_norms_before_clipping': gradients,
        'seconds': elapsed, 'decoder_embedding_max_update': delta,
        'padding_labels_masked': True, 'vision_gradients_absent': True,
    }
    after_captions, _ = generate(model, processor, images, selected['validation'], torch)
    after_rows = [{'image': name, 'caption': text, 'references': references[name]}
                  for name, text in zip(selected['validation'], after_captions)]
    write_json(output / 'after_smoke_predictions.json', after_rows)
    checkpoint = output / 'checkpoint'
    model.save_pretrained(checkpoint, safe_serialization=True)
    processor.save_pretrained(checkpoint)
    # Record training state too; this script does not implement resume training.
    torch.save({'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                'steps': len(losses), 'manifest': inventory}, checkpoint / 'training_state.pt')
    report['peak_allocated_memory_mib'] = torch.cuda.max_memory_allocated() / 2**20
    report['peak_reserved_memory_mib'] = torch.cuda.max_memory_reserved() / 2**20
    write_json(output / 'report.json', report)
    del optimizer, scaler, tracked, before, inputs, labels, loss, gradient, model
    torch.cuda.empty_cache()
    print('Reloading the saved model in an independent offline process', flush=True)
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--reload-check', str(checkpoint),
                            '--predictions', str(output / 'after_smoke_predictions.json'),
                            '--data-dir', str(args.data_dir)], capture_output=True, text=True,
                           encoding='utf-8', timeout=180)
    (output / 'reload.log').write_text(child.stdout + child.stderr, encoding='utf-8')
    if child.returncode:
        raise RuntimeError(f'Independent model reload failed; see {output / "reload.log"}')
    report['independent_offline_reload'] = json.loads(child.stdout.strip())
    print('Independent offline model reload: passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--model-dir', type=Path, help='Local pinned source checkpoint with source.json')
    parser.add_argument('--gpu-only', action='store_true')
    parser.add_argument('--reload-check', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--predictions', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    if args.reload_check:
        reload_check(args.reload_check, args.predictions, args.data_dir)
        return 0
    output = (args.output or ROOT / 'runs' / ('blip-env-' + datetime.now(LOCAL_TZ).strftime('%Y%m%d-%H%M%S-%f'))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'scope': 'Environment integration only; not a quality or speed benchmark.'}
    try:
        run(args, output, report)
        report['status'] = 'passed'
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        print(report['error'], file=sys.stderr)
    write_json(output / 'report.json', report)
    print(f'Report: {output / "report.json"}', flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
