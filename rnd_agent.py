import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import math
import numpy as np
import random
from torch.nn import init
from constants import args
#from torch.nn.modules import Flatten

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class RNDModel(nn.Module):
    def __init__(self):
        super(RNDModel, self).__init__()

        # self.input_size = input_size
        # self.output_size = output_size
        self.device = torch.device('cpu' if args.no_cuda else 'cuda')

        feature_output = 7 * 7 * 64
        self.predictor = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512)
        )

        self.target = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512)
            #nn.Linear(feature_output, 512)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        # for param in self.target.parameters():
        #     param.requires_grad = False

    def forward(self, next_obs):
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature


class RNDAgent:
    def __init__(self):
        self.rnd = RNDModel()
        self.rnd = self.rnd.to(self.rnd.device)
        self.optimizer = optim.Adam(list(self.rnd.predictor.parameters()), lr=args.lr)

    def intrinsic_reward(self, next_obs):
        predict_feature, target_feature = self.rnd.forward(next_obs)
        intrinsic_reward = (target_feature - predict_feature).pow(2).sum(1) / 512
        return predict_feature, target_feature, intrinsic_reward

    def train_rnd(self, predict_feature_batch, target_feature_batch):
        update_proportion = 0.25
        forward_mse = nn.MSELoss(reduction='none')
        forward_loss = forward_mse(predict_feature_batch[0], target_feature_batch[0].detach()).mean(-1)
        mask = torch.rand(len(forward_loss)).to(self.rnd.device)
        mask = (mask < update_proportion).type(torch.FloatTensor).to(self.rnd.device)
        forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.FloatTensor([1]).to(self.rnd.device))

        self.optimizer.zero_grad()
        forward_loss.backward()
        self.optimizer.step()