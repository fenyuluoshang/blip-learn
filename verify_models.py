"""Verify trained smoke checkpoints, causality and padding invariance."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('caption_lab', ROOT / '案例5-自动图像描述/案例5-代码和数据/fullname_lab.py')
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def verify(backbone):
    folder = ROOT / 'runs' / f'verify-{backbone}'
    config = json.loads((folder / 'config.json').read_text())
    vocabulary = json.loads((folder / 'vocabulary.json').read_text())
    split = json.loads((folder / 'split.json').read_text())
    assert not set(split['train']) & set(split['validation'])
    model = lab.build_model(tf, config, len(vocabulary))
    model.load_weights(folder / 'best.weights.h5', skip_mismatch=False)
    image = lab.read_image(tf, split['validation'][0], config['image_size'])[None]
    tokens = tf.constant([[vocabulary.index('<start>'), 4, 5] + [0] * 21])
    expected = model([image, tokens], training=False).numpy()
    assert expected.shape == (1, 24, len(vocabulary))
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(expected.sum(-1), 1, atol=1e-5)
    # Future input changes must not affect earlier predictions.
    changed = tf.constant([[vocabulary.index('<start>'), 4, 6] + [0] * 21])
    actual = model([image, changed], training=False).numpy()
    np.testing.assert_allclose(expected[:, :2], actual[:, :2], atol=1e-5)
    target = tf.constant([[4, 5, vocabulary.index('<end>')] + [0] * 21])
    altered = expected.copy()
    altered[:, 3:] = 1 / len(vocabulary)
    np.testing.assert_allclose(model.loss(target, expected), model.loss(target, altered), atol=1e-6)
    # Rebuilding in a fresh object must restore exactly the same predictions.
    restored = lab.build_model(tf, config, len(vocabulary))
    restored.load_weights(folder / 'best.weights.h5', skip_mismatch=False)
    np.testing.assert_allclose(expected, restored([image, tokens], training=False).numpy(), atol=1e-6)
    print(f'{backbone}: checkpoint restore, finite probabilities, causal mask, padding loss, split isolation PASS', flush=True)
    tf.keras.backend.clear_session()


if __name__ == '__main__':
    for backbone in ['efficientnet', 'densenet']:
        verify(backbone)
