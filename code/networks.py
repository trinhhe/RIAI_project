from enum import Enum
import torch
import torch.nn as nn

MEAN = 0.1307
SIGMA = 0.3081


class SPUImpl(Enum):
    SPU = 0
    OMIT = 1
    IDENTITY = 2


def antinorm(x, add=True):
    return x * SIGMA + MEAN if add else x * SIGMA


class Normalization(nn.Module):
    def __init__(self, device):
        super(Normalization, self).__init__()
        self.mean = torch.Tensor([MEAN]).view((1, 1, 1, 1)).to(device)
        self.sigma = torch.Tensor([SIGMA]).view((1, 1, 1, 1)).to(device)

    def forward(self, x):
        return (x - self.mean) / self.sigma


class SPU(nn.Module):
    def forward(self, x):
        return torch.where(x > 0, x ** 2 - 0.5, torch.sigmoid(-x) - 1)


class Identity(nn.Module):
    def forward(self, x):
        return x


class FullyConnected(nn.Module):

    def __init__(self, device, input_size, fc_layers, spu_impl=SPUImpl.SPU):
        super(FullyConnected, self).__init__()
        self.input_size = input_size
        self.fc_layers = fc_layers

        layers = [Normalization(device), nn.Flatten()]
        prev_fc_size = input_size * input_size
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                if spu_impl == SPUImpl.SPU:
                    layers += [SPU()]
                elif spu_impl == SPUImpl.IDENTITY:
                    layers += [Identity()]
                else:
                    pass
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
