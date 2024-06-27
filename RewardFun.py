import torch
import torch.nn as nn


class RewardNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # layer
        self.layer_1 = nn.Linear(in_dim, 30)
        self.output = nn.Linear(30, out_dim)

    def forward(self, s):
        a = self.layer_1(s)
        a = torch.relu(a)
        a = self.output(a)
        return a
