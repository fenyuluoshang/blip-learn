"""Generate a caption using a locally saved BLIP model and decoding settings."""
import argparse
import json
from pathlib import Path

from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    args = parser.parse_args()
    processor = BlipProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)
    model = BlipForConditionalGeneration.from_pretrained(args.model, local_files_only=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    configuration = args.model / 'inference_config.json'
    generation = json.loads(configuration.read_text(encoding='utf-8')) if configuration.exists() else {
        'do_sample': False, 'num_beams': 1, 'max_new_tokens': 30}
    with Image.open(args.image) as source:
        inputs = processor(images=source.convert('RGB'), return_tensors='pt').to(device)
    with torch.inference_mode(), torch.autocast(device, dtype=torch.float16, enabled=device == 'cuda'):
        ids = model.generate(**inputs, **generation)
    print(processor.decode(ids[0], skip_special_tokens=True))


if __name__ == '__main__':
    main()
