"""unit tests for transforms"""

import unittest

import torch

from radiomana.transforms import BenignSpurs, LogNoise, RandomFreqFlip, RandomFreqShift, RandomTimeCrop, RandomTimeRoll, SyntheticJammer


class TestCustomTransforms(unittest.TestCase):
    def test_log_noise(self):
        """is LogNoise custom transform working as expected?"""
        transform = LogNoise(noise_power_db=-90, p=0.5)
        # test with (tensor, target) tuple
        batch = (torch.randn(16, 512, 243), torch.randint(0, 9, (16,)))
        output = transform(batch)

        self.assertEqual(batch[0].shape, output[0].shape)
        # ensure labels are unchanged
        self.assertTrue(torch.equal(batch[1], output[1]))
        # ensure no nan values in output
        self.assertFalse(torch.isnan(output[0]).any())

    def test_random_time_crop(self):
        """is RandomTimeCrop custom transform working as expected?"""
        crop_width = 211
        transform = RandomTimeCrop(crop_width=crop_width)
        # test with (tensor, target) tuple
        batch = (torch.randn(16, 512, 243), torch.randint(0, 9, (16,)))
        output = transform(batch)
        # ensure non-time dimensions are unchanged
        self.assertEqual(batch[0].shape[0], output[0].shape[0])
        self.assertEqual(batch[0].shape[1], output[0].shape[1])
        # ensure time dimension is cropped to fixed width
        self.assertEqual(output[0].shape[2], crop_width)
        # ensure labels are unchanged
        self.assertTrue(torch.equal(batch[1], output[1]))

        # test with smaller input (should return unchanged)
        small_batch = (torch.randn(16, 512, 100), torch.randint(0, 9, (16,)))
        small_output = transform(small_batch)
        self.assertEqual(small_output[0].shape[2], 100)  # unchanged

    def test_roll_shift_flip(self):
        """do the cross-site geometric transforms keep shape, values and labels?"""
        psd = torch.randn(512, 243)
        for transform in [RandomTimeRoll(), RandomFreqShift(max_fraction=0.1), RandomFreqFlip(p=1)]:
            output, label = transform((psd, 3))
            self.assertEqual(output.shape, psd.shape)
            self.assertEqual(label, 3)
            # circular shifts and flips only reorder values
            self.assertTrue(torch.allclose(output.flatten().sort().values, psd.flatten().sort().values))

    def test_synthetic_jammer(self):
        """does SyntheticJammer relabel clean samples, add power, and leave jammed samples alone?"""
        torch.manual_seed(0)
        psd = torch.randn(512, 243) * 3 - 40
        original = psd.clone()
        transform = SyntheticJammer(p=1, jammed_label=4)
        for _ in range(10):  # covers all three jammer kinds
            output, label = transform((psd, 1))
            self.assertEqual(label, 4)
            self.assertEqual(output.shape, psd.shape)
            self.assertTrue((output >= psd).all())
            self.assertGreater((output - psd).max().item(), 5)
        # input is not modified in place
        self.assertTrue(torch.equal(psd, original))
        output, label = transform((psd, 6))
        self.assertEqual(label, 6)
        self.assertTrue(torch.equal(output, psd))

    def test_benign_spurs(self):
        """do BenignSpurs add faint lines to clean samples without relabelling?"""
        torch.manual_seed(0)
        psd = torch.randn(512, 243) * 3 - 40
        output, label = BenignSpurs(p=1)((psd, 2))
        self.assertEqual(label, 2)
        rows_changed = ((output - psd).abs() > 1e-4).any(dim=1).sum().item()
        self.assertTrue(1 <= rows_changed <= 4)
        self.assertLess((output - psd).max().item(), 15)
        self.assertFalse(torch.isnan(output).any())
        output, label = BenignSpurs(p=1)((psd, 5))
        self.assertTrue(torch.equal(output, psd))
