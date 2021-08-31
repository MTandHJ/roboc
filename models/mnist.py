

import torch
import torch.nn as nn
from .base import AdversarialDefensiveModel, generate_weights



class MNIST(AdversarialDefensiveModel):

    def __init__(
        self, dim_feature=256, 
        num_classes=10,
        scale=10.
    ):
        super(MNIST, self).__init__()

        self.conv = nn.Sequential( # 1 x 28 x 28
            nn.Conv2d(1, 32, 3),   # 32 x 26 x 26
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, 3),  # 32 x 24 x 24
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2),       # 32 x 12 x 12
            nn.Conv2d(32, 64, 3),  # 64 x 10 x 10
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3),  # 64 x 8 x 8
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2)        # 64 x 4 x 4
        )

        self.dense = nn.Sequential(
            nn.Linear(64 * 4 * 4, 200),
            nn.PReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, dim_feature),
            nn.BatchNorm1d(dim_feature)
        )
        self.activation = nn.PReLU()
        self.fc = nn.Linear(dim_feature, num_classes)
        _weights = generate_weights(dim_feature)[:num_classes] * scale
        self.fc.weight.data.copy_(_weights)
        self.fc.requires_grad_(False)

    def forward(self, x):
        x = self.conv(x).flatten(start_dim=1)
        features = self.activation(self.dense(x))
        logits = self.fc(features)
        if self.training:
            return features, logits
        elif self.attacking:
            return features
        else:
            return logits