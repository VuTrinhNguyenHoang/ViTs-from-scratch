import torch
import torch.nn as nn
import math

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div(keep)
        return x * mask

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: torch.Tensor | None = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                          # [B, N, H, D]
        q = q.transpose(1, 2)                                # [B, H, N, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale        # [B, H, N, N]
        if attn_mask is not None:
            attn = attn + attn_mask                          # mask broadcastable
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                       # [B, H, N, D]
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0,
                 attn_drop=0.0, proj_drop=0.0, mlp_drop=0.0,
                 drop_path=0.0, qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_drop, proj_drop, qkv_bias)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, mlp_ratio, mlp_drop)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x, attn_mask: torch.Tensor | None = None):
        x = x + self.drop_path1(self.attn(self.norm1(x), attn_mask))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, depth, **layer_kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(**layer_kwargs) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(layer_kwargs["dim"])
        
    def forward(self, x, attn_mask: torch.Tensor | None = None):
        for blk in self.layers:
            x = blk(x, attn_mask)
        x = self.norm(x)
        return x
