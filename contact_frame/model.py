"""
model.py  –  ContactDetector

R(2+1)D-18 backbone (pretrained on Kinetics-400) with a lightweight binary head.

R(2+1)D decomposes each 3-D convolution into a 2-D spatial conv followed by a
1-D temporal conv.  This captures both appearance (bat/ball shape) and motion
(swing velocity, approach trajectory) while being faster than a full 3-D conv.

Input  : (B, 3, T, H, W)  – video clip, T frames
Output : (B,)              – contact score as a logit (apply sigmoid for probability)
"""

import torch
import torch.nn as nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18


class ContactDetector(nn.Module):

    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None
        backbone = r2plus1d_18(weights=weights)

        # Replace the 400-class head with a contact-score head
        in_features = backbone.fc.in_features   # 512 for R(2+1)D-18
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)   # (B,)
