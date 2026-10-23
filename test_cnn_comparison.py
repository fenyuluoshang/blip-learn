"""Scientific contracts for the original-case CUDA replication."""
import unittest

from train_cnn_comparison import load_builder, masked_loss_accuracy, namespace
import keras
import torch


class ComparisonContracts(unittest.TestCase):
    def test_original_masked_loss_and_gradients_ignore_padding(self):
        model = load_builder()(namespace(), dict(image_size=224, sequence_length=25,
            embed_dim=256, backbone='efficientnet', weights='none', train_backbone=False), 20)
        logits = torch.randn(2, 24, 20, requires_grad=True, device='cuda')
        probabilities = logits.softmax(-1)
        targets = torch.zeros(2, 24, dtype=torch.long, device='cuda')
        targets[:, :3] = torch.tensor([[2, 3, 4], [4, 3, 2]], device='cuda')
        expected = model.loss(targets, probabilities)
        actual, _, count = masked_loss_accuracy(probabilities, targets)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        actual.backward()
        self.assertEqual(count.item(), 6)
        torch.testing.assert_close(logits.grad[:, 3:], torch.zeros_like(logits.grad[:, 3:]))

    def test_decoder_cannot_see_future_tokens_and_cnn_is_frozen(self):
        model = load_builder()(namespace(), dict(image_size=224, sequence_length=25,
            embed_dim=256, backbone='efficientnet', weights='none', train_backbone=False), 20)
        model.cuda().eval()
        head = keras.Model([model.get_layer('visual_features').output, model.inputs[1]], model.output)
        visual = torch.randn(2, 49, 1280, device='cuda')
        original = torch.full((2, 24), 2, dtype=torch.long, device='cuda')
        changed = original.clone()
        changed[:, 8:] = 7
        with torch.inference_mode():
            first = head([visual, original], training=False)
            second = head([visual, changed], training=False)
        torch.testing.assert_close(first[:, :8], second[:, :8], rtol=1e-5, atol=1e-6)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.get_layer('efficientnetb0').parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in head.parameters()))


if __name__ == '__main__':
    unittest.main()
