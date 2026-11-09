"""torchvision ResNet-18 on one 224 x 224 image, built on the meta device.

    uv run model-viewer capture examples/resnet.py:build -o out/resnet18
"""

import torch
from torchvision.models import resnet18


def build():
    with torch.device("meta"):
        model = resnet18().eval()
    return model, (torch.randn(1, 3, 224, 224, device="meta"),)
