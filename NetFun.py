import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()
        # layer
        self.layer_1 = nn.Linear(in_dim, n_hidden_1)
        self.output = nn.Linear(n_hidden_2, out_dim)

    def forward(self, s):
        a = self.layer_1(s)
        a = torch.relu(a)
        a = self.output(a)
        return a
