import numpy as np
import torch
import torch.nn as nn


class DeepRitzNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, m=20, actv=nn.Tanh()):
        super(DeepRitzNet, self).__init__()
        self.actv = actv
        self.linear_input = nn.Linear(input_dim, m)
        self.linear2 = nn.Linear(m, m)
        self.linear3 = nn.Linear(m, m)
        self.linear4 = nn.Linear(m, m)
        self.linear5 = nn.Linear(m, m)
        self.linear6 = nn.Linear(m, m)
        self.linear7 = nn.Linear(m, m)
        self.linear_output = nn.Linear(m, output_dim)

    def forward(self, x):
        y = self.linear_input(x)
        y = y + self.actv(self.linear3(self.actv(self.linear2(y))))
        y = y + self.actv(self.linear5(self.actv(self.linear4(y))))
        y = y + self.actv(self.linear7(self.actv(self.linear6(y))))
        output = self.linear_output(y)
        return output
