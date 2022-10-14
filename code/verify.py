import copy
import torch
import cfg
import deeppoly
import networks
import util


def create_verifier(net: networks.FullyConnected, true_label, eps) -> deeppoly.FullyConnected:
    net.requires_grad_(False)
    L = net.layers[-1].out_features
    net_copy = copy.deepcopy(net)
    net_copy.layers.add_module('prove_layer', torch.nn.Linear(L, L - 1, bias=True))
    net_copy.layers[-1].requires_grad_(False)
    net_copy.layers[-1].bias.fill_(0.0)
    weight = torch.zeros(L - 1, L - 1).fill_diagonal_(-1)
    weight = torch.cat((weight[:, 0:true_label], torch.ones(L - 1, 1), weight[:, true_label:]), dim=1)
    net_copy.layers[-1].weight = torch.nn.Parameter(weight)

    for param in net_copy.parameters():
        param.requires_grad_(False)
    verifier = deeppoly.FullyConnected(cfg.DEVICE, net_copy, eps)
    return verifier


def run_verifier_get_loss(deep: deeppoly.FullyConnected, inputs):
    l, _, _, _ = deep(inputs)
    return -l.amin(1)


def verifies(deep: deeppoly.FullyConnected, inputs):
    largest_competitor = run_verifier_get_loss(deep, inputs)
    return largest_competitor < 0  # 1-D bool Tensor


def train_and_verify(deep: deeppoly.FullyConnected, input, iterations=100):
    input.requires_grad_(False)
    learning_rate = .1
    optimizer = None  # delayed initialisation
    for _ in range(iterations):
        loss = run_verifier_get_loss(deep, input)
        if cfg.DEBUG:
            print(float(loss))
        if loss < 0:
            return True

        if optimizer is None:
            optimizer = torch.optim.Adam(deep.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.95)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return False
