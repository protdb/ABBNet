import torch
import torch.nn as nn

from config.config import ModelParams


class Normalize(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias


class GaussianSmearing(nn.Module):
    def __init__(self):
        super(GaussianSmearing, self).__init__()
        config = ModelParams()
        self.cutoff_lower = config.rbf_cutoff_lower
        self.cutoff_upper = config.rbf_cutoff_upper
        self.num_rbf = config.num_rbf
        offset, coeff = self._initial_params()
        self.register_parameter("coeff", nn.Parameter(coeff))
        self.register_parameter("offset", nn.Parameter(offset))

    def _initial_params(self):
        offset = torch.linspace(self.cutoff_lower, self.cutoff_upper, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class DihedralFeatures(nn.Module):
    # 3 dihedral angles; sin and cos of each angle
    node_in = 6

    def __init__(self):
        super(DihedralFeatures, self).__init__()
        config = ModelParams()
        node_embed_dim = config.dihedral_embed_dim
        self.node_embedding = nn.Linear(self.node_in, node_embed_dim, bias=True)
        self.norm_nodes = Normalize(node_embed_dim)

    def forward(self, x):
        v = self.node_embedding(x)
        v = self.norm_nodes(v)
        return v


def normalize(tensor, dim=-1):
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def tuple_sum(*args):
    return tuple(map(sum, zip(*args)))


def tuple_cat(*args, dim=-1):
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x, idx):
    return x[0][idx], x[1][idx]


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _split(x, nv):
    v = torch.reshape(x[..., -3 * nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3 * nv]
    return s, v


def _merge(s, v):
    v = torch.reshape(v, v.shape[:-2] + (3 * v.shape[-2],))
    return torch.cat([s, v], -1)

