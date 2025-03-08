from typing import Optional

import numpy as np
import torch
import torch.nn as nn

class ImageTokenizer(nn.Module):
    def __init__(self, patch_size: int = 8, n_channels: int = 3, embedding_dim: int = 512):
        super().__init__()
        self._patch_size = patch_size
        self._conv = nn.Conv2d(n_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        # BxCxHxW -> BxTxD
        bs, _, h, w = x.size()
        out = self._conv(x)
        return out.view(bs, (h // self._patch_size) * (w // self._patch_size), -1)

class Attention(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int):
        super().__init__()
        self._qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self._n_heads = n_heads

    def forward(self, x: torch.Tensor):
        # scaled dot product attention
        # for each qkv: (B,T,nh,hs) -> (B,nh,T,hs), and we can do this first before splitting dim
        # (B,T,D) -> (B,T,3D) -> (B,T,nh,3*hs) -> (B,nh,T,3*hs) -> tuple[(B,nh,T,hs), ...]
        bs, T, _ = x.size()
        nh = self._n_heads
        hs = x.size(-1) // nh
        q, k, v = torch.split(self._qkv(x).view(bs, T, nh, 3*hs).transpose(1, 2), hs, dim=-1)
        # (B,nh,T,hs) \dotprod (B,nh,hs,T) -> (B,nh,T,T)
        # (B,nh,T,T) \dotprod (B,nh,T,hs) -> (B,nh,T,hs)
        out = torch.nn.functional.softmax((q @ k.transpose(-1, -2) / np.sqrt(hs)), dim=-1) @ v
        return out.transpose(1, 2).reshape(bs, T, -1)

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, mlp_dim: int):
        super().__init__()
        self._linear1 = nn.Linear(input_dim, mlp_dim)
        self._relu = nn.ReLU()
        self._linear2 = nn.Linear(mlp_dim, output_dim)

    def forward(self, x: torch.Tensor):
        out = self._linear1(x)
        out = self._relu(out)
        out = self._linear2(out)
        return out

class Block(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, n_heads: int):
        super().__init__()
        self._ln = nn.LayerNorm(embedding_dim)
        self._mha = Attention(embedding_dim, n_heads)
        self._mlp = MLP(embedding_dim, embedding_dim, mlp_dim)

    def forward(self, x: torch.Tensor):
        # Pre-Norm -> MHA -> MLP
        out = self._ln(x)
        out = self._mha(out)
        out = self._mlp(out)
        return out

class ViT(nn.Module):
    def __init__(self,
                 patch_size: int = 8,
                 n_channels: int = 3,
                 n_layers: int = 6,
                 embedding_dim: int = 512,
                 mlp_dim: int = 1024,
                 n_heads: int = 8,
                 n_classes: int = 10):
        super().__init__()
        self._image_tokenizer = ImageTokenizer(patch_size, n_channels, embedding_dim)
        # TODO add position embedding
        self._blocks = nn.ModuleList([Block(embedding_dim, mlp_dim, n_heads) for _ in range(n_layers)])
        self._mlp = MLP(embedding_dim, n_classes, mlp_dim)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = self._image_tokenizer(x)
        for block in self._blocks:
            out = block(out)
        logits = self._mlp(out)
        loss = None
        if y is not None:
            loss = nn.functional.cross_entropy(logits, y)
        return logits, loss
