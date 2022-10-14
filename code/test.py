import torch
import random
import deeppoly
import cfg
import networks
import util
import verify
from torchvision import datasets, transforms
import sys

if len(sys.argv) > 1 and sys.argv[1] == "--debug":
    cfg.DEBUG = True


def affine_matches_bruteforce(net: networks.FullyConnected, inputs, eps):
    deep = deeppoly.FullyConnected(cfg.DEVICE, net, eps)
    l, u, al, au = deep(inputs)
    brl, bru = util.bruteforce_bounds(net, eps, inputs)
    ldiff = torch.max(torch.abs(l - brl))
    udiff = torch.max(torch.abs(u - bru))
    assert ldiff <= 1e-4, f"Affine transformer does not work, difference between bruteforce and DeepPoly = {ldiff}"
    assert udiff <= 1e-4, f"Affine transformer does not work, difference between bruteforce and DeepPoly = {udiff}"
    return True


def test_affine_matches_bruteforce():
    print("Test: affine matches bruteforce")
    for i in range(10):
        net, inputs, _ = util.dummy_net_and_inputs(cfg.DEVICE, num_inputs=500, spu=networks.SPUImpl.IDENTITY)
        eps = i ** 2 * .001
        assert affine_matches_bruteforce(net, inputs, eps)


def test_is_deterministic():
    print("Test: is deterministic")
    rng = random.Random()
    SEED = 123
    rng.seed(SEED)
    net1, inputs1, eps1 = util.dummy_net_and_inputs(cfg.DEVICE, spu=networks.SPUImpl.IDENTITY, rng=rng)
    rng.seed(SEED)
    net2, inputs2, eps2 = util.dummy_net_and_inputs(cfg.DEVICE, spu=networks.SPUImpl.IDENTITY, rng=rng)
    assert torch.all(inputs1 == inputs2)
    assert torch.all(net1(inputs1) == net2(inputs2))
    deep1 = deeppoly.FullyConnected(cfg.DEVICE, net1, eps1)
    deep2 = deeppoly.FullyConnected(cfg.DEVICE, net2, eps2)
    assert (all(torch.all(x == y) for x, y in zip(deep1(inputs1), deep2(inputs2))))


def metric(l, u):
    assert torch.all(u >= l)
    logwidth = (u - l).log().flatten()
    logvolume = torch.ones_like(logwidth).dot(logwidth)
    return logvolume


def test_playground():
    print("Test: playground (work-in-progress)")

    net, inputs, eps = util.dummy_net_and_inputs(cfg.DEVICE, num_inputs=1, spu=networks.SPUImpl.SPU)
    gold = util.raw_output_to_label(net(inputs))
    deep = verify.create_verifier(net, gold, eps)
    verify.train_and_verify(deep, inputs)


default_meanrange = (-10, 5)
default_sizerange = (-10, 5)


def gen_l_u(shape, meanrange=default_meanrange, sizerange=default_sizerange):
    mean = torch.Tensor(*shape).uniform_(*meanrange)
    offset = torch.Tensor(*shape).uniform_(*sizerange)
    offset.exp_()
    l, u = mean - offset, mean + offset
    return l, u


def check_spu_on_inputs(x, l, u):
    eps_crude = 1e-5  # all points should have absolute error <= this
    eps_fine = 1e-7  # all points should have either absolute or relative error <= this
    assert l.shape == u.shape
    assert list(x.shape)[:-1] == list(l.shape)
    dummy = networks.SPU()
    dummy_prev = deeppoly.Identity(cfg.DEVICE, dummy)
    spu = deeppoly.SPUDefault(cfg.DEVICE, dummy_prev, dummy)
    lw, lb, uw, ub = spu.spu(l, u)
    actual_val = dummy(x)
    actual_val.transpose_(1, 2)
    actual_val.transpose_(0, 1)
    x.transpose_(1, 2)
    x.transpose_(0, 1)
    bs = x.shape[0]
    lw = util.spread(lw, bs)
    lb = util.spread(lb, bs)
    uw = util.spread(uw, bs)
    ub = util.spread(ub, bs)
    lower_bound = lw * x + lb
    upper_bound = uw * x + ub
    lower_abserr = lower_bound - actual_val
    upper_abserr = actual_val - upper_bound

    def check_one(abserr, name, bound):
        relerr = torch.abs(torch.maximum(torch.Tensor([0.]), abserr) / actual_val)
        ok = (abserr <= eps_crude) & ((abserr <= eps_fine) | (relerr <= eps_fine))
        if not torch.all(ok):
            ix = tuple(map(int, (~ok).nonzero()[0]))
            orig_ix = ix[1:]
            assert torch.all(
                ok), f"Error at {ix}: abserr = {abserr[ix]}, relerr = {relerr[ix]}, value = {actual_val[ix]}, {name} bound = {bound[ix]}, l = {l[orig_ix]}, u = {u[orig_ix]}, x = {x[ix]}"

    check_one(lower_abserr, "lower", lower_bound)
    check_one(upper_abserr, "upper", upper_bound)


def load_nets():
    net_names = ['net0_fc1', 'net0_fc2', 'net0_fc3', 'net0_fc4', 'net0_fc5', 'net1_fc1', 'net1_fc2', 'net1_fc3',
                 'net1_fc4', 'net1_fc5']
    return [(name, util.load_net(name)) for name in net_names]


def test_spu_xyz_is_sound():
    print("Test: SPU soundness")
    N = 100
    M = 200
    for i in range(3):
        l, u = gen_l_u((i + 1, N))
        x = torch.empty((i + 1, N, M))
        for j in range(l.shape[0]):
            for k in range(l.shape[1]):
                x[j][k][:] = torch.linspace(l[j][k], u[j][k], M)

    check_spu_on_inputs(x, l, u)


def test_spu_xyz_is_deterministic():
    print("Test: SPU is deterministic and shape-agnostic")

    def dupl(tens):
        res = util.spread(tens, 2).reshape(2, N)
        return res

    N = 50
    dummy = networks.SPU()
    dummy_prev = deeppoly.Identity(cfg.DEVICE, dummy)
    spu = deeppoly.SPUDefault(cfg.DEVICE, dummy_prev, dummy)
    l, u = gen_l_u((1, N))
    ll, uu = dupl(l), dupl(u)
    lw, lb, uw, ub = spu.spu(l, u)
    llw, llb, uuw, uub = spu.spu(ll, uu)
    for two, one in zip((llw, llb, uuw, uub), (lw, lb, uw, ub)):
        diff = torch.abs(two - dupl(one))
        maxdiff = diff.max().max()
        assert maxdiff <= 1e-5, f"result on two duplicates != 2 * [result on the original]\n|diff|={maxdiff}"


def test_spu_xyz_is_sound2():
    print("Test: SPU soundness 2")
    N = M = 100
    K = 200
    offset = torch.linspace(default_meanrange[0], default_meanrange[1], N).exp()
    two = util.spread(offset, 2)
    N *= 2
    two[1] = torch.flip(-two[1], (0,))
    mean = two.flatten()
    mean = util.spread(mean, M)
    spread = torch.linspace(default_sizerange[0], default_sizerange[1], M).exp()
    spread = util.spread(spread, N).transpose(0, 1)
    assert torch.all(spread > 0)
    l, u = mean - spread, mean + spread
    x = torch.empty((K, M, N))
    for i in range(K):
        alpha = i / (K - 1)
        x[i] = l * alpha + u * (1 - alpha)
    x = x.transpose(0, 1)
    x = x.transpose(1, 2)
    check_spu_on_inputs(x, l, u)