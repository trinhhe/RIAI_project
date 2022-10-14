import torch
import torch.nn as nn
import networks
import util


class Module(nn.Module):
    def __init__(self, device, prev, layer):
        super(Module, self).__init__()
        self.device = device
        self.prev = prev
        self.layer = layer

    def cache(self, l, u, al, au):
        self.l, self.u, self.al, self.au = l, u, al, au
        return l, u, al, au


class Converter(Module):
    """A layer that converts from the standard input representation to the DeepPoly representation.

    Assumes the initial bounds are an [-eps, eps]^n-box around the input, eps must be given on initialisation. Clamps the box to [0, 1]^n."""

    def __init__(self, device, eps):
        super(Converter, self).__init__(device, None, None)
        self.device = device
        self.eps = eps
        self.min_val = float(networks.Normalization(device)(torch.Tensor([0])))
        self.max_val = float(networks.Normalization(device)(torch.Tensor([1])))

    def forward(self, x):
        l = (x - self.eps).to(self.device).clamp(self.min_val, self.max_val)
        u = (x + self.eps).to(self.device).clamp(self.min_val, self.max_val)
        al = l.unsqueeze(-1).to(self.device)
        au = u.unsqueeze(-1).to(self.device)
        return self.cache(l, u, al, au)


class Linear(Module):
    def __init__(self, device, prev, layer):
        super(Linear, self).__init__(device, prev, layer)

    def forward(self, x):
        l, u, al, au = x
        bs = l.shape[0]
        nal = torch.cat((self.layer.bias.unsqueeze(1), self.layer.weight), 1)
        nal = util.spread(nal, bs)
        nau = nal

        nl, nu = backsubst(nal, nau, self)
        return self.cache(nl, nu, nal, nau)


class Identity(Module):
    def __init__(self, device, prev):
        super(Identity, self).__init__(device, prev, None)

    def forward(self, x):
        l, u, al, au = x
        bs, n = l.shape
        nl, nu = l, u
        nal = torch.cat((torch.zeros((n, 1)), torch.eye(n)), 1)
        nal = util.spread(nal, bs)
        nau = nal
        return self.cache(nl, nu, nal, nau)


class SPU(Module):
    def __init__(self, device, prev, layer):
        super(SPU, self).__init__(device, prev, layer)

    def forward(self, x):
        l, u, _, _ = x
        bs, n = l.shape
        lw, lb, uw, ub = self.spu(l, u)
        nal = torch.zeros((bs, n, n + 1))
        nal.diagonal(1, -2, -1)[:] = lw
        nal[:, :, 0] = lb
        nau = torch.zeros((bs, n, n + 1))
        nau.diagonal(1, -2, -1)[:] = uw
        nau[:, :, 0] = ub
        # XXX: not needed really, can replace with simple bound calculation (possibility for speedup)
        nl, nu = backsubst(nal, nau, self)
        return self.cache(nl, nu, nal, nau)

    def spu(self, l, u):
        """Calculates SPU lower and upper bounds pointwise. Receives l and u of
        shape (b, n), should output lw, lb, uw, ub each of shape (bs, n) such
        that for each b, i and each x \in [l[b][i], u[b][i]], it holds that
        lw[b][i]·x + lb[b][i] ≤ SPU(x) ≤ uw[b][i]·x + ub[b][i]."""

        raise NotImplementedError("This should be implemented by the child class.")


class SPUDumb(SPU):
    def spu(self, l, u):
        ones = torch.ones_like(l)
        return 0 * ones, -.5 * ones, 0 * ones, u ** 2 * ones


class SPUPointwise(SPU):
    def spu(self, l, u):
        lw, lb, uw, ub = (torch.empty_like(l) for _ in range(4))
        for b in range(l.shape[0]):
            for i in range(l.shape[1]):
                lw[b][i], lb[b][i], uw[b][i], ub[b][i] = self.spu_evaluate(l[b][i], u[b][i])

        return lw, lb, uw, ub

    def spu_evaluate(self, l, u):
        if u < 0:
            return self.spu_negative(l, u)
        elif l > 0:
            return self.spu_positive(l, u)
        elif _spu(u) > _spu(l):
            return self.spu_mixed_normal_u(l, u)
        else:  # _spu(u) < _spu(l)
            return self.spu_mixed_small_u(l, u)

    def spu_negative(self, l, u):
        lw, lb = calc_line_between(l, u)
        uw, ub = spu_tangent_parametrised(l, u)
        return lw, lb, uw, ub

    def spu_positive(self, l, u):
        lw, lb = spu_tangent_parametrised(l, u)
        uw, ub = calc_line_between(l, u)
        return lw, lb, uw, ub

    def spu_mixed_small_u(self, l, u):
        lower_bounds = [(0, -0.5),
                        calc_line_between(l, torch.Tensor([0]))]
        uw, ub = 0, _spu(l)
        lw, lb = self._try_lower_bounds(lower_bounds, l, u, uw, ub)
        return lw, lb, uw, ub

    def spu_mixed_normal_u(self, l, u):
        lower_bounds = [(0, -0.5),
                        calc_line_between(l, torch.Tensor([0])),
                        spu_tangent_parametrised(torch.Tensor([0]), u)]
        uw, ub = calc_line_between(l, u)
        lw, lb = self._try_lower_bounds(lower_bounds, l, u, uw, ub)
        return lw, lb, uw, ub

    def _try_lower_bounds(self, lower_bounds, l, u, uw, ub):
        # areas = [find_area(l, u, lower_bound[0], lower_bound[1], uw, ub) for lower_bound in lower_bounds]
        areas2 = [find_area_proxy(l, u, lower_bound[0], lower_bound[1]) for lower_bound in lower_bounds]
        # print(list(a1 - a2 / 2 for a1, a2 in zip(areas, areas2)))
        min_index = int(torch.argmin(torch.Tensor(areas2)))
        return lower_bounds[min_index]


# Area between lines a1x + b1 and a2x + b2 in range between x1 and x2
# Correct assuming lines do not cross
def find_area(x1, x2, a1, b1, a2, b2):
    y11 = a1 * x1 + b1
    y12 = a1 * x2 + b1
    y21 = a2 * x1 + b2
    y22 = a2 * x2 + b2
    sum_ver = abs(y11 - y21) + abs(y12 - y22)
    hor_dist = x2 - x1
    return hor_dist * sum_ver / 2


_spu = networks.SPU().forward


def calc_line_between(l, u):
    yl = _spu(l)
    yu = _spu(u)
    a = (yu - yl) / (u - l + 1e-8)
    b = yl - l * a
    return a, b


# -2 * area between lines a1x + b1 and 0, x \in [x1, x2]
def find_area_proxy(x1, x2, a1, b1):
    return -(x2 - x1) * (a1 * x1 + b1 + a1 * x2 + b1)


def spu_tangent_parametrised(l, u, t=0.5):
    return spu_tangent(l * (1 - t) + u * t)


def spu_grad(x0):
    toonegative = x0 <= -50
    positive = x0 >= 0
    negative = ~toonegative & ~positive
    out = torch.zeros_like(x0)
    out[toonegative] = 0
    out[negative] = -.5 / (x0[negative].cosh() + 1)
    out[positive] = 2 * x0[positive]
    return out


def spu_tangent(x0):
    a = spu_grad(x0)
    b = _spu(x0) - a * x0
    return a, b


def spu_shifted_over_x(x):
    normal = ((x <= -1e-5) | (x > 0))
    res = (_spu(x) + .5) / x
    res.masked_fill_(~normal, -.25)
    return res


class SPUTrainable(SPU):
    first_run = True

    def spu(self, l, u):
        if self.first_run:
            self.first_run = False
            self._tl = torch.nn.Parameter(0 * torch.ones_like(l))
            self._tu = torch.nn.Parameter(0 * torch.ones_like(u))
            self._tl.requires_grad_()
            self._tu.requires_grad_()
        self.tl = torch.sigmoid(4 * self._tl)
        self.tu = torch.sigmoid(4 * self._tu)

        assert torch.all(0 <= self.tl) and torch.all(self.tl <= 1)
        assert torch.all(0 <= self.tu) and torch.all(self.tu <= 1)

        negative = u <= 0
        positive = l >= 0
        crossing = ~positive & ~negative

        endpoint_line = calc_line_between(l, u)

        def mask(m, out):
            res = []
            for el in out:
                a = torch.zeros_like(el)
                a[m] = el[m]
                res.append(a)
            return tuple(res)

        def sum_up(*things):
            return tuple(sum(a) for a in zip(*things))

        def spu_negative(l, u):
            tang = spu_tangent_parametrised(l, u, self.tu)
            return endpoint_line[0], endpoint_line[1], tang[0], tang[1]

        def spu_positive(l, u):
            tang = spu_tangent_parametrised(l, u, self.tl)
            return tang[0], tang[1], endpoint_line[0], endpoint_line[1]

        def spu_crossing(l, u):
            u_end = self.binsearch_acceptable_range(l, u)
            u_out = spu_tangent_parametrised(l, u_end, self.tu)
            not_covered = u_out[0] * u + u_out[1] <= _spu(u)
            u_out = sum_up(mask(not_covered, endpoint_line), mask(~not_covered, u_out))
            l_out = spu_bottom_tangent(l, u, self.tl)
            return l_out[0], l_out[1], u_out[0], u_out[1]

        def spu_bottom_tangent(l, u, tl):
            threshold = (-l) / (u - l)
            is_normal = tl >= threshold
            normal_line = spu_tangent_parametrised(l, u, tl)
            steepest_slope = spu_shifted_over_x(l)
            weird_line = steepest_slope * (1 - (tl / threshold) ** 2), torch.ones_like(normal_line[1]) * (-.5)
            return sum_up(mask(is_normal, normal_line), mask(~is_normal, weird_line))

        return sum_up(
            mask(negative, spu_negative(negative * l, negative * u)),
            mask(positive, spu_positive(positive * l, positive * u)),
            mask(crossing, spu_crossing(l, crossing * u)),
        )

    @staticmethod
    def binsearch_acceptable_range(l, u):
        cover_requirements = _spu(u)
        bl = l
        br = torch.zeros_like(l)
        for _ in range(20):
            c = (bl + br) / 2
            a, b = spu_tangent(c)
            at_u = a * u + b
            covers = at_u >= cover_requirements
            # if covers with c, it will cover with all smaller c', so we want to search to the right
            bl = (~covers) * bl + covers * c
            br = covers * br + (~covers) * c
        return bl


class SPUDefault(SPUTrainable):
    pass


class FullyConnected(nn.Module):
    def __init__(self, device, input_net, eps):
        super(FullyConnected, self).__init__()
        for i, sequential in enumerate(input_net.children()):
            pass
        assert i == 0
        assert isinstance(sequential, nn.Sequential)

        layers = []
        for l in sequential.children():
            if isinstance(l, networks.Normalization):
                layers.append(networks.Normalization(device))
                eps /= networks.SIGMA
            elif isinstance(l, nn.Flatten):
                layers.append(nn.Flatten())
                layers.append(Converter(device, eps))
            # TODO: assert they are in the right order etc. as a sanity check
            elif isinstance(l, nn.Linear):
                layers.append(Linear(device, layers[-1], l))
            elif isinstance(l, networks.SPU):
                layers.append(SPUDefault(device, layers[-1], l))
            elif isinstance(l, networks.Identity):
                layers.append(Identity(device, layers[-1]))
            else:
                raise NotImplementedError(l)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def debug_net(net):
    for i, sequential in enumerate(net.children()):
        pass
    assert i == 0
    for layer in sequential:
        print(layer)
        if not isinstance(layer, Module):
            continue
        print(layer.l, layer.u, layer.al, layer.au, sep="\n")
        print("---")


def debug_grad(net: FullyConnected):
    for layer in net.layers:
        if isinstance(layer, SPU):
            print(layer._tu.grad)
            print(layer._tl.grad)


def substitute(x, yp, yn):
    """Performs one step of the backsubstition.

    (For the sake of simplicity, we ignore the first batch dimension in the description.)

    `x` contains the matrix for either the ≥ or ≤ constraints on variables from
    the current layer, where x[i] represents the coefficients variable x_i,
    i.e., x_i ≤/≥ x[i][0] + Σ_{j=1} x[i][j] y_j.

    `yp` and `yn` contain the matrices for the ≥ and ≤ constraints on variables
    y_j from some previous layer, where again yp[j] expressess y_j ≤/≥ y[j][0]
    + Σ_{k=1} y[j][k] z_k, similarly for yn[j]. It holds that if x represents
    ≥, then (yp, yn) represent (≥, ≤), and if x represents ≤, then (yp, yn)
    represent (≤, ≥).

    The procedure expresses all x_i's in terms of z_k. This amounts to matrix
    multiplication, but we need to handle positive and negative entries of x
    differently. For positive entries, we substitute x[i][j] y_j from yp, and
    for negative entries, we substitute them from yn.

    All of this is actually done for different `x`, `yp` and `yn` in parallel,
    since these tensors actually contain `bs` different matrices.
    """

    assert yp.shape == yn.shape
    assert yp.shape[0] == x.shape[0], "Incompatible batch size"
    assert x.shape[2] - 1 == yp.shape[1], "Incompatible matrix size"
    zero = torch.Tensor([0.])
    px = x.where(x > 0, zero)
    nx = x.where(x <= 0, zero)
    pnew = px[:, :, 1:] @ yp  # "1:" for the first constant term
    nnew = nx[:, :, 1:] @ yn
    new = pnew + nnew
    new[:, :, 0:1] += x[:, :, 0:1]  # add the constant terms separately
    return new


def backsubst(al, au, layer):
    """Backsubstitutes into al and au until we arrive at the first layer.
    
    `layer` is the current layer which has produced al and au.
    """

    while layer.prev is not None:
        # Converter has layer.prev = None, so this is where we terminate
        layer = layer.prev

        back_al, back_au = layer.al, layer.au
        al = substitute(al, back_al, back_au)
        au = substitute(au, back_au, back_al)

    bs, n_out, one = al.shape
    assert al.shape[-1] == 1, "Backsubstitution did not reach the input layer, or Converter is broken."
    l = al.flatten(-2)
    u = au.flatten(-2)
    assert torch.all(l < u + 1e-5)
    return l, u
