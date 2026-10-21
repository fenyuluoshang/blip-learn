"""Flickr8k CNN + Transformer captioning. See root README.md for commands."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from caption_data import DATA_DIR, prepare, load_captions, split_data


def build_model(tf, config, vocabulary_size, weights=None):
    layers = tf.keras.layers
    size, length, dim = config['image_size'], config['sequence_length'], config['embed_dim']
    # ImageNet EfficientNet adds a fixed rescaling layer; preserve it on reload.
    if config.get('weights') == 'imagenet':
        weights = 'imagenet'
    image = layers.Input((size, size, 3), name='image')
    tokens = layers.Input((length - 1,), dtype='int32', name='tokens')
    if config['backbone'] == 'efficientnet':
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights=weights, input_shape=(size, size, 3))
        visual_input = image  # EfficientNet includes its own rescaling; input is [0,255].
    else:
        base = tf.keras.applications.DenseNet121(include_top=False, weights=weights, input_shape=(size, size, 3))
        visual_input = tf.keras.applications.densenet.preprocess_input(image)
    base.trainable = config['train_backbone']
    visual = base(visual_input, training=False)
    visual = layers.Reshape((-1, visual.shape[-1]), name='visual_features')(visual)
    visual = layers.Dense(dim, activation='relu')(visual)
    attention = layers.MultiHeadAttention(num_heads=2, key_dim=dim // 2)(visual, visual)
    visual = layers.LayerNormalization()(visual + attention)
    visual = layers.LayerNormalization()(visual + layers.Dense(dim)(layers.Dense(dim * 2, activation='relu')(visual)))
    word = layers.Embedding(vocabulary_size, dim)(tokens)
    position_ids = layers.Lambda(lambda t: tf.tile(tf.range(length - 1)[None, :], [tf.shape(t)[0], 1]))(tokens)
    positions = layers.Embedding(length - 1, dim)(position_ids)
    text = word + positions
    # Padding keys are invisible; causal masking prevents looking at future words.
    padding = layers.Lambda(lambda t: tf.not_equal(t[:, None, :], 0))(tokens)
    attention = layers.MultiHeadAttention(num_heads=2, key_dim=dim // 2, dropout=.1)(text, text, attention_mask=padding, use_causal_mask=True)
    text = layers.LayerNormalization()(text + attention)
    attention = layers.MultiHeadAttention(num_heads=2, key_dim=dim // 2, dropout=.1)(text, visual)
    text = layers.LayerNormalization()(text + attention)
    feed = layers.Dense(dim)(layers.Dropout(.1)(layers.Dense(dim * 2, activation='relu')(text)))
    text = layers.LayerNormalization()(text + feed)
    output = layers.Dense(vocabulary_size, activation='softmax')(text)
    model = tf.keras.Model([image, tokens], output)

    def masked_loss(target, prediction):
        loss = tf.keras.losses.sparse_categorical_crossentropy(target, prediction)
        mask = tf.cast(target != 0, loss.dtype)
        return tf.math.divide_no_nan(tf.reduce_sum(loss * mask), tf.reduce_sum(mask))

    def masked_accuracy(target, prediction):
        mask = tf.cast(target != 0, tf.float32)
        correct = tf.cast(tf.equal(tf.cast(target, tf.int64), tf.argmax(prediction, -1)), tf.float32)
        return tf.math.divide_no_nan(tf.reduce_sum(correct * mask), tf.reduce_sum(mask))

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=masked_loss, metrics=[masked_accuracy])
    model.caption_accuracy = masked_accuracy
    return model


def read_image(tf, path, size):
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    return tf.image.resize(tf.cast(image, tf.float32), (size, size))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'predict'])
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / 'efficientnet')
    parser.add_argument('--backbone', choices=['efficientnet', 'densenet'], default='efficientnet')
    parser.add_argument('--weights', default='imagenet', help='imagenet, none, or local backbone weights file')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--cache-features', action='store_true', help='Cache frozen CNN features once per image')
    parser.add_argument('--image', type=Path)
    parser.add_argument('--train-backbone', action='store_true', help='Train randomly initialized or fine-tune pretrained CNN')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    if args.command == 'prepare':
        images, captions = prepare(args.data_dir)
        mapping = load_captions(images, captions)
        train, valid = split_data(mapping, limit=args.limit)
        print(json.dumps({'images': len(mapping), 'captions': sum(map(len, mapping.values())), 'train_images': len(train), 'validation_images': len(valid)}, indent=2))
        return
    try:
        import tensorflow as tf
    except ImportError:
        parser.exit(1, 'TensorFlow is required: python -m pip install -r requirements.txt\n')
    tf.keras.utils.set_random_seed(111)
    output = args.output.resolve()
    if args.command == 'predict':
        if not args.image or not args.image.is_file():
            parser.error('predict requires --image pointing to a local JPEG')
        config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
        vocabulary = json.loads((output / 'vocabulary.json').read_text(encoding='utf-8'))
        model = build_model(tf, config, len(vocabulary))
        model.load_weights(output / 'best.weights.h5')
        image = read_image(tf, str(args.image), config['image_size'])[None]
        words = [vocabulary.index('<start>')]
        for _ in range(config['sequence_length'] - 1):
            padded = words + [0] * (config['sequence_length'] - 1 - len(words))
            scores = model([image, tf.constant([padded])], training=False)[0, len(words) - 1].numpy()
            scores[0] = scores[1] = scores[vocabulary.index('<start>')] = -1
            token = int(scores.argmax())
            if vocabulary[token] == '<end>':
                break
            words.append(token)
        print(' '.join(vocabulary[i] for i in words[1:]))
        return
    if args.cache_features and args.train_backbone:
        parser.error('--cache-features requires a frozen backbone')
    if args.weights == 'none' and not args.train_backbone:
        parser.error('--weights none requires --train-backbone (random frozen features are unsuitable)')
    images, captions = prepare(args.data_dir)
    train, valid = split_data(load_captions(images, captions), limit=args.limit)
    config = dict(backbone=args.backbone, image_size=224, sequence_length=25, embed_dim=256, train_backbone=args.train_backbone, seed=111, batch_size=args.batch_size, epochs=args.epochs, weights=args.weights)
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=10000, standardize=None, output_sequence_length=25)
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices([text for texts in train.values() for text in texts]).batch(128))
    vocabulary = vectorizer.get_vocabulary()
    if not all(token in vocabulary for token in ['<start>', '<end>']):
        raise ValueError('Vocabulary is missing sentence markers')

    def dataset(mapping, training):
        paths, texts = zip(*[(path, text) for path, captions in mapping.items() for text in captions])
        data = tf.data.Dataset.from_tensor_slices((list(paths), list(texts)))
        if training:
            data = data.shuffle(len(paths), seed=111)
        def convert(path, text):
            sequence = vectorizer(text)
            return (read_image(tf, path, 224), sequence[:-1]), sequence[1:]
        return data.map(convert, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_model(tf, config, len(vocabulary), None if args.weights == 'none' else args.weights)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (output / 'vocabulary.json').write_text(json.dumps(vocabulary), encoding='utf-8')
    (output / 'split.json').write_text(json.dumps({'train': list(train), 'validation': list(valid)}, indent=2), encoding='utf-8')
    start = time.perf_counter()
    training_model = model
    if args.cache_features:
        import numpy as np
        feature_tensor = model.get_layer('visual_features').output
        extractor = tf.keras.Model(model.inputs[0], feature_tensor)
        training_model = tf.keras.Model([feature_tensor, model.inputs[1]], model.output)
        training_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=model.loss, metrics=[model.caption_accuracy])
        paths = list(train) + list(valid)
        feature_shape = tuple(feature_tensor.shape[1:])
        # Memory-mapped cache avoids keeping all feature vectors in Python RAM.
        cache = np.lib.format.open_memmap(output / 'features.npy', mode='w+', dtype='float32', shape=(len(paths), *feature_shape))
        for offset in range(0, len(paths), args.batch_size):
            batch = tf.stack([read_image(tf, path, 224) for path in paths[offset:offset + args.batch_size]])
            cache[offset:offset + len(batch)] = extractor(batch, training=False).numpy()
            if offset % (args.batch_size * 20) == 0:
                print(f'CNN feature cache: {min(offset + len(batch), len(paths))}/{len(paths)} images', flush=True)
        cache.flush()
        indexes = {path: i for i, path in enumerate(paths)}
        def dataset(mapping, training):
            rows = [(indexes[path], text) for path, texts in mapping.items() for text in texts]
            ids, texts = zip(*rows)
            data = tf.data.Dataset.from_tensor_slices((list(ids), list(texts)))
            if training:
                data = data.shuffle(len(rows), seed=111)
            def convert(index, text):
                feature = tf.numpy_function(lambda i: np.array(cache[int(i)]), [index], tf.float32)
                feature.set_shape(feature_shape)
                sequence = vectorizer(text)
                return (feature, sequence[:-1]), sequence[1:]
            return data.map(convert, num_parallel_calls=2).batch(args.batch_size).prefetch(2)

    # The cached head shares variables with the complete image model. Save the
    # complete model so predict uses exactly the same checkpoint format.
    class FullModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
        def set_model(self, _model):
            super().set_model(model)

    history = training_model.fit(dataset(train, True), validation_data=dataset(valid, False), epochs=args.epochs, verbose=2, shuffle=False, callbacks=[FullModelCheckpoint(str(output / 'best.weights.h5'), save_weights_only=True, save_best_only=True), tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True), tf.keras.callbacks.CSVLogger(str(output / 'history.csv'))])
    result = {'training_seconds': time.perf_counter() - start, 'parameters': model.count_params(), 'train_images': len(train), 'validation_images': len(valid), 'history': history.history}
    (output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
