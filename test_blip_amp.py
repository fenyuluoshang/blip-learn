import unittest

import torch

from blip_amp import amp_optimizer_step


class AmpStepTests(unittest.TestCase):
    def check_overflow_recovery(self, device):
        parameter = torch.nn.Parameter(torch.tensor([1.], device=device))
        optimizer = torch.optim.SGD([parameter], lr=.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
        scaler = torch.amp.GradScaler(device, init_scale=8.)
        scaler.scale(parameter.square().sum()).backward()
        parameter.grad.fill_(float('inf'))
        updated, _ = amp_optimizer_step(optimizer, scaler, [parameter])
        if updated:
            scheduler.step()
        self.assertFalse(updated)
        self.assertEqual(parameter.item(), 1.)
        self.assertEqual(scaler.get_scale(), 4.)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertIsNone(parameter.grad)
        scaler.scale(parameter.square().sum()).backward()
        updated, _ = amp_optimizer_step(optimizer, scaler, [parameter])
        if updated:
            scheduler.step()
        self.assertTrue(updated)
        self.assertLess(parameter.item(), 1.)
        self.assertEqual(scheduler.last_epoch, 1)

    def test_cpu_overflow_does_not_update_weights_or_scheduler_and_next_update_recovers(self):
        self.check_overflow_recovery('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_overflow_does_not_update_weights_or_scheduler_and_next_update_recovers(self):
        self.check_overflow_recovery('cuda')


if __name__ == '__main__':
    unittest.main()
