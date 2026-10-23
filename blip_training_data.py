"""Raw Flickr8k references and reproducible image-level random splits."""
import csv
import hashlib
import json
from pathlib import Path
import random

from caption_data import prepare


def read_references(data_dir):
    images, captions = prepare(data_dir)
    mapping = {}
    with captions.open(encoding='utf-8-sig', newline='') as stream:
        if stream.readline().strip() != 'image,caption':
            raise ValueError('BLIP training currently requires the local CSV captions.txt')
        stream.seek(0)
        for row in csv.DictReader(stream):
            name, text = row['image'], row['caption'].strip()
            if Path(name).name != name or not (images / name).is_file() or not text:
                raise ValueError(f'Invalid/missing image or caption: {name}')
            mapping.setdefault(name, []).append(text)
    if len(mapping) < 2:
        raise ValueError('At least two captioned images are required')
    return images, mapping, hashlib.sha256(captions.read_bytes()).hexdigest()


def make_split(references, seed=111, validation_fraction=.2, limit=None):
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must be between zero and one')
    names = sorted(references)
    random.Random(seed).shuffle(names)
    if limit is not None:
        if limit < 2:
            raise ValueError('Image limit must be at least two')
        names = names[:limit]
    pivot = max(1, min(len(names) - 1, int(len(names) * (1 - validation_fraction))))
    split = {'train': names[:pivot], 'validation': names[pivot:]}
    validate_split(split, references)
    return split


def validate_split(split, references):
    train, validation = split['train'], split['validation']
    if not train or not validation or len(train) != len(set(train)) or len(validation) != len(set(validation)):
        raise ValueError('Empty split or duplicate images')
    if set(train) & set(validation):
        raise ValueError('Train and validation images overlap')
    for name in train + validation:
        if Path(name).name != name or name not in references:
            raise ValueError(f'Invalid split image: {name}')


def manifest_hash(split):
    return hashlib.sha256(json.dumps(split, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
