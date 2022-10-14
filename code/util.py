import functools
import itertools
import random
import torch
import torch.nn as nn
import cfg
import deeppoly
import networks

detrandom = random.Random()
detrandom.seed(123)


def dummy_net_and_inputs(device, apparent_eps=.1, input_size=3, layers=None, spu=networks.SPUImpl.IDENTITY,
                         num_inputs=100, rng=detrandom):
    """Creates a dummy network with custom dimensions, and some inputs for it, all filled with deterministically random values.

    By default omits SPU layers, so that losslessness of affine layers can be easily tested."""
    # We want to counter the effect of normalisation, so that "visually nice"
    # epsilons get propagated through the network after normalisation
    eps = apparent_eps * networks.SIGMA
    layers = layers or [3, 4]
    net = networks.FullyConnected(device, input_size, layers, spu_impl=spu).to(device)
    make_reasonable_parameters(net, rng)
    inputs = torch.empty((num_inputs, 1, input_size, input_size))
    # XXX: the constants below are hand-crafted so that the input stays in [0, 1]
    fill_with_nice_randomness(inputs, spread=16, step=.1, rng=rng)
    inputs += 1.2
    # inputs are in range [-.4, 2.8], antinorm makes them in [0, 1]
    inputs = networks.antinorm(inputs)
    assert torch.all(inputs >= 0) and torch.all(inputs <= 1)
    return net, inputs, eps


def fill_with_nice_randomness(tensor, spread=20, step=.1, rng=detrandom):
    """Fills the tensor with visually nice deterministic randomness."""
    v = tensor.view(-1)
    for i in range(len(v)):
        v.data[i] = rng.randint(-spread, spread) * step


def make_reasonable_parameters(net, rng=detrandom):
    """Fills the parameters of the network with visually nice deterministic randomness."""
    for param in net.parameters():
        fill_with_nice_randomness(param, rng=rng)


def bruteforce_bounds(net, eps, inputs):
    """Bruteforces the bounds of an affine-only network by trying all vertices of the eps-ball."""

    seq = net.children().__next__()
    for layer in seq:
        assert (isinstance(layer, nn.Linear)
                or isinstance(layer, nn.Flatten)
                or isinstance(layer, networks.Normalization)
                or isinstance(layer, networks.Identity)
                )

    mi = ma = net(inputs)
    dim = functools.reduce(lambda a, b: a * b, inputs.shape[1:])
    assert dim < 30, "Dimension too big for a bruteforce"
    for i, vertex in enumerate(itertools.product((-eps, eps), repeat=dim)):
        delta = torch.Tensor(vertex).reshape(inputs.shape[1:])
        new_input = (inputs + delta).clamp(0, 1)
        out = net(new_input)
        mi = mi.minimum(out)
        ma = ma.maximum(out)

    return mi, ma


def spread(t, k):
    "Turns a [a, b, c, …]-shaped tensor into a [k, a, b, c, …]-shaped one, repeating it k times."""
    tensor = t[:]
    tensor.unsqueeze_(0)
    rep = [1 if i else k for i, _ in enumerate(tensor.shape)]
    return tensor.repeat(rep)


def load_net(name):
    if name.endswith('fc1'):
        net = networks.FullyConnected(cfg.DEVICE, cfg.INPUT_SIZE, [50, 10]).to(cfg.DEVICE)
    elif name.endswith('fc2'):
        net = networks.FullyConnected(cfg.DEVICE, cfg.INPUT_SIZE, [100, 50, 10]).to(cfg.DEVICE)
    elif name.endswith('fc3'):
        net = networks.FullyConnected(cfg.DEVICE, cfg.INPUT_SIZE, [100, 100, 10]).to(cfg.DEVICE)
    elif name.endswith('fc4'):
        net = networks.FullyConnected(cfg.DEVICE, cfg.INPUT_SIZE, [100, 100, 50, 10]).to(cfg.DEVICE)
    elif name.endswith('fc5'):
        net = networks.FullyConnected(cfg.DEVICE, cfg.INPUT_SIZE, [100, 100, 100, 100, 10]).to(cfg.DEVICE)
    else:
        assert False

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % name, map_location=torch.device(cfg.DEVICE)))
    net.eval()
    net.requires_grad_(False)
    return net


def raw_output_to_label(out):
    return out.max(dim=1)[1].item()
