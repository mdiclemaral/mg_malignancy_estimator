from io import BytesIO
from typing import Self

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn

__all__ = [
    "MassSegmentation",
]


class MassSegmentation:
    def __init__(
        self,
        device: torch.device,
    ) -> Self:
        """..."""
        self.device = device

    def load(
        self,
        checkpoint: bytes,
    ) -> None:
        """..."""
        self.net = DeepLabV3Plus(in_channels=3, out_channels=1)
        self.net.load_state_dict(torch.load(BytesIO(checkpoint), map_location="cpu"))
        for param in self.net.parameters():
            param.requires_grad = False
        self.net = self.net.to(self.device)

    def __call__(
        self,
        pixel_array: torch.Tensor | None,
    ) -> np.ndarray | list:
        """..."""
        if pixel_array is None:
            return []

        # self.net.train()
        with torch.no_grad():
            pred = self.net(pixel_array.to(self.device))

        return pred.squeeze(1).cpu().numpy()


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
    ) -> Self:
        """..."""
        super(DeepLabV3Plus, self).__init__()

        self.model = smp.DeepLabV3Plus(
            encoder_name="resnext50_32x4d",
            encoder_depth=5,
            encoder_weights=None,
            decoder_channels=256,
            decoder_atrous_rates=(12, 24, 36),
            in_channels=in_channels,
            classes=out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """..."""
        return torch.sigmoid(self.model(x))
