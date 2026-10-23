import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import BlipProcessor

from blip_training_data import make_split, manifest_hash, validate_split
from caption_metrics import compute_metrics
from train_blip import CaptionDataset


class BlipTrainingTests(unittest.TestCase):
    def test_full_split_is_image_grouped_reproducible_and_portable(self):
        references = {f'{index:05d}.jpg': ['first reference', 'second reference'] for index in range(8091)}
        split = make_split(references)
        self.assertEqual(len(split['train']), 6472)
        self.assertEqual(len(split['validation']), 1619)
        self.assertFalse(set(split['train']) & set(split['validation']))
        self.assertEqual(set(split['train'] + split['validation']), set(references))
        self.assertEqual(split, make_split(dict(reversed(list(references.items())))))
        self.assertNotEqual(split, make_split(references, seed=112))
        self.assertEqual(manifest_hash(split), manifest_hash(make_split(references)))
        with self.assertRaises(ValueError):
            validate_split({'train': ['00000.jpg'], 'validation': ['00000.jpg']}, references)

    def test_cached_dataset_preserves_references_and_masks_padding(self):
        processor = BlipProcessor.from_pretrained('models/blip-base-source-82a3776', local_files_only=True, use_fast=False)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'features.npy'
            np.save(path, np.ones((1, 2, 3), dtype='float16'))
            data = CaptionDataset(path, [('one.jpg', 'A dog running.'), ('one.jpg', 'A playful dog.')],
                                  {'one.jpg': 0}, processor.tokenizer, 32, 30522)
            self.assertEqual(len(data), 2)
            item = data[0]
            self.assertEqual(item['input_ids'][0].item(), 30522)
            self.assertTrue(torch.all(item['labels'][item['attention_mask'] == 0] == -100))
            self.assertTrue(torch.equal(item['input_ids'][item['attention_mask'] == 1],
                                        item['labels'][item['attention_mask'] == 1]))
            self.assertEqual(item['encoder_hidden_states'].shape, (2, 3))
            self.assertTrue(torch.equal(item['encoder_hidden_states'], data[1]['encoder_hidden_states']))
            del data  # Release the Windows memory mapping before TemporaryDirectory cleanup.

    def test_generation_metrics_reward_matching_references(self):
        references = ['a dog is running on green grass', 'two children are playing in the snow',
                      'a red car is driving down the street']
        good = [{'caption': text, 'references': [text]} for text in references]
        bad = [{'caption': 'unrelated object', 'references': [text]} for text in references]
        good_metrics, bad_metrics = compute_metrics(good), compute_metrics(bad)
        self.assertAlmostEqual(good_metrics['bleu4'], 1.)
        for key in good_metrics:
            self.assertGreater(good_metrics[key], bad_metrics[key], key)


if __name__ == '__main__':
    unittest.main()
