import tempfile
import unittest
from pathlib import Path

from caption_data import load_captions, split_data


class CaptionDataTests(unittest.TestCase):
    def test_csv_and_token_formats_preserve_markers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ['one.jpg', 'two.jpg']:
                (root / name).touch()
            csv = root / 'captions.txt'
            csv.write_text('image,caption\none.jpg,"A dog, running!"\ntwo.jpg,' + 'word ' * 40 + '\n', encoding='utf-8')
            mapping = load_captions(root, csv)
            self.assertEqual(mapping[str(root / 'one.jpg')], ['<start> a dog running <end>'])
            sentence = mapping[str(root / 'two.jpg')][0].split()
            self.assertEqual(len(sentence), 25)
            self.assertEqual(sentence[-1], '<end>')
            token = root / 'tokens.txt'
            token.write_text('one.jpg#0\tA dog, running!\ntwo.jpg#0\t' + 'word ' * 40, encoding='utf-8')
            self.assertEqual(load_captions(root, token), mapping)

    def test_split_keeps_image_groups_disjoint_and_reproducible(self):
        mapping = {str(i): ['caption one', 'caption two'] for i in range(20)}
        train, valid = split_data(mapping)
        self.assertFalse(set(train) & set(valid))
        self.assertEqual(set(train) | set(valid), set(mapping))
        self.assertEqual((train, valid), split_data(mapping))
        with self.assertRaises(ValueError):
            split_data(mapping, limit=1)


if __name__ == '__main__':
    unittest.main()
