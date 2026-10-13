from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from src.steganalysis.srnet import DEFAULT_SRNET_ROOT, initialize_srnet


@unittest.skipUnless(DEFAULT_SRNET_ROOT.exists(), "SRNet reference is not installed")
class InitializationTests(unittest.TestCase):
    """The initialization must match the reference, and start near ln(2).

    A network that starts with large logits can lower its loss by making the
    output constant, which also removes its sensitivity to the embedding. A
    network that starts near ln(2) has no such shortcut.
    """

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.model = initialize_srnet(DEFAULT_SRNET_ROOT, torch.device("cpu"))

    def test_dense_layer_is_small(self) -> None:
        weight = self.model.dense.weight
        self.assertAlmostEqual(float(weight.std()), 0.01, delta=0.004)
        self.assertTrue(torch.all(self.model.dense.bias == 0))

    def test_convolutions_use_xavier_uniform(self) -> None:
        for module in self.model.modules():
            if not isinstance(module, nn.Conv2d):
                continue
            fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            fan_out = module.out_channels * module.kernel_size[0] * module.kernel_size[1]
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            peak = float(module.weight.detach().abs().max())
            self.assertLessEqual(peak, bound + 1e-6)

    @staticmethod
    def structured_batch(count: int = 8) -> torch.Tensor:
        """A batch with spatial structure, closer to a photograph than noise.

        Uniform noise hides the difference between the two schemes, because
        the normalization layers then see the same statistics either way.
        """
        torch.manual_seed(7)
        grid = torch.linspace(0, 1, 256)
        rows, columns = torch.meshgrid(grid, grid, indexing="ij")
        base = 0.5 + 0.4 * torch.sin(12 * columns) * torch.cos(9 * rows)
        images = [base + 0.05 * torch.randn(256, 256) for _ in range(count)]
        return torch.stack(images).unsqueeze(1).clamp(0, 1)

    def test_initial_output_sits_near_ln_two(self) -> None:
        """Measured in train mode, where the normalization uses batch values.

        The earlier scheme gave 0.6998 here, which is above ln(2). A network
        that starts above ln(2) can lower its loss by going quiet, and it
        loses its sensitivity to the embedding on the way.
        """
        self.model.train()
        with torch.no_grad():
            output = self.model(self.structured_batch())
        loss = nn.NLLLoss()(output, torch.tensor([0, 1] * 4))
        self.assertAlmostEqual(float(loss), math.log(2), delta=0.003)

    def test_initial_output_still_responds_to_the_input(self) -> None:
        """Near-uniform output must not mean an input-blind network."""
        self.model.train()
        images = self.structured_batch(4)
        stego = images.clone()
        stego[:, :, ::4, ::4] += 1.0 / 255.0
        with torch.no_grad():
            gap = float((self.model(images) - self.model(stego)).abs().max())
        self.assertGreater(gap, 1e-7)


if __name__ == "__main__":
    unittest.main()
