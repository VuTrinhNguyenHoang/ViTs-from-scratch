import torch
import torch.nn as nn
from .transformer import TransformerEncoder

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        
    def forward(self, x):
        x = self.proj(x)                       # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)       # [B, N, D]
        return x

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, num_classes=10,
                 embed_dim=128, depth=6, num_heads=8, mlp_ratio=4.0,
                 attn_drop=0.0, proj_drop=0.0, mlp_drop=0.0, drop_path=0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.encoder = TransformerEncoder(
            depth=depth,
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
            drop_path=drop_path,
            qkv_bias=True,
        )
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        return self.head(x[:, 0])
