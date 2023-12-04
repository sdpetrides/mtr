# src: https://github.com/r0mainK/outperformer/blob/main/src/fast_attention.py
# MIT License

import math

import torch
from torch import nn


def apply_scaling(scale, x):
    return torch.einsum("...n,...nd->...nd", scale, x)


def create_orf(d_k, m):
    blocks = torch.randn(math.ceil(m / d_k), d_k, d_k)
    blocks, _ = torch.linalg.qr(blocks)
    scale = torch.randn(m, d_k).norm(dim=1)
    return apply_scaling(scale, blocks.reshape(-1, d_k)[:m])


def apply_regular_feature_map(x, orf, epsilon=1e-6):
    m, d_k = orf.shape
    proj_x = x @ orf.T / math.pow(d_k, 1 / 4)
    norm = (x**2).sum(dim=-1, keepdim=True) / (2 * math.sqrt(d_k))
    return (torch.exp(proj_x - norm) + epsilon) / math.sqrt(m)


def apply_hyperbolic_feature_map(x, orf, epsilon=1e-6):
    m, d_k = orf.shape
    proj_x = x @ orf.T / math.pow(d_k, 1 / 4)
    proj_x = torch.cat([proj_x, -proj_x], dim=-1)
    norm = (x**2).sum(dim=-1, keepdim=True) / (2 * math.sqrt(d_k))
    return (torch.exp(proj_x - norm) + epsilon) / math.sqrt(2 * m)


def fast_attention(query, key, value):
    buffer = torch.cat(
        [key.transpose(1, 2).bmm(value), key.sum(1).unsqueeze(-1)], dim=-1
    )
    buffer = query.bmm(buffer)
    return apply_scaling(1 / buffer[:, :, -1], buffer[:, :, :-1])


class FastSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, m, use_hyperbolic=False):
        super(FastSelfAttention, self).__init__()
        self.nhead = nhead
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.register_buffer(
            "orf", create_orf(d_model // self.nhead, m), persistent=False
        )
        self.apply_feature_map = apply_regular_feature_map
        if use_hyperbolic:
            self.apply_feature_map = apply_hyperbolic_feature_map

    def redraw_orf(self):
        m, d_k = self.orf.shape
        orf = create_orf(d_k, m)
        orf = orf.to(self.orf.device)
        self.register_buffer("orf", orf, persistent=False)

    def split_by_head(self, x, B, L):
        return (
            x.view(B, L, self.nhead, -1)
            .permute(0, 2, 1, 3)
            .reshape(B * self.nhead, L, -1)
        )

    def concat_by_head(self, x, B, L):
        return x.reshape(B, self.nhead, L, -1).permute(0, 2, 1, 3).reshape(B, L, -1)

    def forward(self, x):
        B, L, _ = x.shape
        query, key, value = (self.split_by_head(l(x), B, L) for l in self.linears[:3])
        query = self.apply_feature_map(query, self.orf)
        key = self.apply_feature_map(key, self.orf)
        out = fast_attention(query, key, value)
        out = self.concat_by_head(out, B, L)
        out = self.linears[3](out)
        return out
