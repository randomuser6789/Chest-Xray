"""Pretrained DenseNet-121 with final layer swapped for 3-way multi-label output."""

import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121

NUM_CLASSES = 3


def build_model(freeze_features=False):
    """Return a DenseNet-121 with its classifier replaced for 3-way multi-label logit output.

    freeze_features=True freezes the pretrained conv layers (linear-probe style),
    leaving only the new classifier trainable. False fine-tunes the whole network.
    """
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, NUM_CLASSES)

    return model
