
from constants import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StateClassifier(nn.Module):
    """
    A classifier that generates labels to put into the
    state history.
    The outputs of this classifier are fed as part of the
    input to the policy.
    """

    def __init__(self):
        super().__init__()
        self.cnn = FixupCNN(IMAGE_SIZE, 3)
        self.final_layer = nn.Linear(256, NUM_LABELS)

    def forward(self, x):
        mask = np.ones(x.shape[1:], dtype=np.uint8)
        mask[0:10] = 0.0
        float_obs = (torch.from_numpy(mask).to(x.device) * x).float() / 255.0
        features = self.cnn(float_obs)
        logits = self.final_layer(features)
        return logits


class StateFeatures:
    """
    Generate the part of state vectors that reflect the
    observation. This does not include rewards or actions.
    """

    def __init__(self, path='save_classifier.pkl'):
        self.classifier = StateClassifier()
        self.classifier.load_state_dict(torch.load(path))
        self.classifier.to(torch.device('cuda'))

    def features(self, obses):
        res = []
        for obs in obses:
            # Check if we have a key.
            if (obs[3] != 0).any():
                res.append([1.0])
            else:
                res.append([0.0])
        device = next(self.classifier.parameters()).device
        obs_tensor = torch.from_numpy(obses).to(device)
        class_out = torch.sigmoid(self.classifier(obs_tensor)).detach().cpu().numpy()
        return np.concatenate([np.array(res), class_out], axis=-1)


class FixupCNN(nn.Module):
    """
    A larger version of the IMPALA CNN with Fixup init.
    See Fixup: https://arxiv.org/abs/1901.09321.
    """

    def __init__(self, image_size, depth_in):
        super().__init__()
        layers = []
        for depth_out in [32, 64, 64]:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                FixupResidual(depth_out, 8),
                FixupResidual(depth_out, 8),
            ])
            depth_in = depth_out
        layers.extend([
            FixupResidual(depth_in, 8),
            FixupResidual(depth_in, 8),
        ])
        self.conv_layers = nn.Sequential(*layers)
        self.linear = nn.Linear(math.ceil(image_size / 8) ** 2 * depth_in, 256)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv_layers(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = F.relu(x)
        return x


class FixupResidual(nn.Module):
    def __init__(self, depth, num_residual):
        super().__init__()
        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        for p in self.conv1.parameters():
            p.data.mul_(1 / math.sqrt(num_residual))
        for p in self.conv2.parameters():
            p.data.zero_()
        self.bias1 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias2 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias3 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias4 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.scale = nn.Parameter(torch.ones([depth, 1, 1]))

    def forward(self, x):
        x = F.relu(x)
        out = x + self.bias1
        out = self.conv1(out)
        out = out + self.bias2
        out = F.relu(out)
        out = out + self.bias3
        out = self.conv2(out)
        out = out * self.scale
        out = out + self.bias4
        return out + x
