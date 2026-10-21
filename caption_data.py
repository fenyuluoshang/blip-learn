"""Local Flickr8k data preparation; no network access or ML dependencies."""
import csv
import random
import re
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / '案例5-自动图像描述' / '案例5-代码和数据'


def prepare(data_dir=DATA_DIR):
    data_dir = Path(data_dir).resolve()
    images = data_dir / 'Flicker8k_Dataset'
    archive = data_dir / 'Flickr8k_Dataset.zip'
    if archive.exists():
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                # Extract only flat JPEG entries, ignoring macOS metadata.
                parts = Path(item.filename).parts
                if len(parts) == 2 and parts[0] == images.name and parts[1].lower().endswith('.jpg'):
                    target = images / parts[1]
                    if not target.exists() or target.stat().st_size != item.file_size:
                        images.mkdir(exist_ok=True)
                        with source.open(item) as src, target.open('wb') as dst:
                            import shutil
                            shutil.copyfileobj(src, dst)
    if not images.is_dir():
        raise FileNotFoundError(f'Missing local images or archive: {data_dir}')
    captions = data_dir / 'captions.txt'
    if not captions.exists():
        archive = data_dir / 'Flickr8k_text.zip'
        with zipfile.ZipFile(archive) as source:
            captions = data_dir / 'Flickr8k.token.txt'
            captions.write_bytes(source.read('Flickr8k.token.txt'))
    return images, captions


def load_captions(images, captions, sequence_length=25):
    mapping = {}
    with Path(captions).open(encoding='utf-8-sig') as stream:
        first = stream.readline()
        stream.seek(0)
        if first.strip() == 'image,caption':
            rows = ((row['image'], row['caption']) for row in csv.DictReader(stream))
        else:
            rows = (line.rstrip('\n').split('\t', 1) for line in stream if line.strip())
        for name, caption in rows:
            name = name.split('#')[0]
            if Path(name).name != name:
                raise ValueError(f'Invalid image name: {name}')
            path = Path(images) / name
            if not path.is_file():
                raise FileNotFoundError(path)
            words = re.sub(r'[^a-z0-9\s]', '', caption.lower()).split()
            if words:
                # Reserve two positions so the end marker is never truncated.
                text = '<start> ' + ' '.join(words[:sequence_length - 2]) + ' <end>'
                mapping.setdefault(str(path), []).append(text)
    if len(mapping) < 2:
        raise ValueError('At least two images with captions are required')
    return mapping


def split_data(mapping, seed=111, limit=None):
    names = sorted(mapping)
    random.Random(seed).shuffle(names)
    if limit is not None:
        if limit < 2:
            raise ValueError('--limit must be at least 2')
        names = names[:limit]
    pivot = max(1, min(len(names) - 1, int(len(names) * .8)))
    return ({key: mapping[key] for key in names[:pivot]},
            {key: mapping[key] for key in names[pivot:]})
