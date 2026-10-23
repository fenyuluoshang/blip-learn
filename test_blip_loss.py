import math
import unittest

import torch
import torch.nn.functional as F

from blip_loss import accumulation_weight, grouped_caption_loss


class GroupedLossTests(unittest.TestCase):
    def test_gradient_matches_original_two_example_microbatches_including_short_tail(self):
        generator = torch.Generator().manual_seed(111)
        for examples in [7, 9, 15, 16, 23]:
            scores = torch.randn(examples, 7, 13, dtype=torch.float64, generator=generator, requires_grad=True)
            labels = torch.randint(0, 13, (examples, 7), generator=generator)
            labels[::2, -3:] = -100
            labels[1::2, -1:] = -100
            original = sum(F.cross_entropy(scores[i:i+2, :-1].reshape(-1, 13),
                                          labels[i:i+2, 1:].reshape(-1), label_smoothing=.1)
                           for i in range(0, examples, 2)) / math.ceil(examples / 2)
            original_grad = torch.autograd.grad(original, scores, retain_graph=True)[0]
            # Each test uses one logical accumulation window, including ragged final batches.
            physical = 8
            accumulation = math.ceil(examples / physical)
            larger = sum(grouped_caption_loss(scores[i:i+physical], labels[i:i+physical], 2, .1)[0]
                         * accumulation_weight(examples, i // physical, physical, accumulation, 2)
                         for i in range(0, examples, physical))
            larger_grad = torch.autograd.grad(larger, scores)[0]
            self.assertAlmostEqual(original.item(), larger.item(), places=12)
            torch.testing.assert_close(original_grad, larger_grad, rtol=1e-12, atol=1e-12)

    def test_original_configuration_resumes_without_new_loss_key(self):
        for examples in range(1, 34):
            batches = math.ceil(examples / 2)
            weights = [accumulation_weight(examples, i, 2, 8, 2) for i in range(batches)]
            for start in range(0, batches, 8):
                self.assertAlmostEqual(sum(weights[start:start+8]), 1.)


if __name__ == '__main__':
    unittest.main()
