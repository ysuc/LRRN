#ViT

"""
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al.

https://arxiv.org/abs/2010.11929
"""

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    """层归一化"""

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)  # 非多头时不投影

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class T(nn.Module):
    def __init__(self):
        super(T, self).__init__()

    def forward(self, x):
        return torch.transpose(x, 1, 2)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        """
        :param image_size:int. Image size. If you have rectangular images, make sure your image size is the maximum of the width and height
        :param patch_size: int. Number of patches. image_size must be divisible by patch_size
        :param num_classes: int. Number of classes to classify
        :param dim: int. Last dimension of output tensor after linear transformation nn.Linear(..., dim).
        :param depth: int. Number of Transformer blocks.
        :param heads: int. Number of heads in Multi-head Attention layer.
        :param mlp_dim:int. Dimension of the MLP (FeedForward) layer.
        :param pool: string, either cls token pooling or mean pooling
        :param channels: int, default 3. Number of image's channels.
        :param dim_head: the dimension of each head
        :param dropout:float between [0, 1], default 0.. Dropout rate.
        :param emb_dropout:float between [0, 1], default 0. Embedding dropout rate.
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)  # n must be greater than 16.
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
        #     nn.Linear(patch_dim, dim),
        # )

        self.to_patch_embedding = nn.Sequential(
            # batch_size, patch_size**2*channel, num_patches**0.5, num_patches**0.5
            nn.Conv2d(channels, patch_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),  # batch_size, patch_size**2*channel, num_patches
            T(),  # batch_size,  num_patches, patch_size**2*channel,
            nn.Linear(patch_dim, dim)  # batch_size,  num_patches, dim
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)  # 2, 64, 1024

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # 2, 1, 1024
        x = torch.cat((cls_tokens, x), dim=1)  # 2,65,1024
        x += self.pos_embedding  # 1, 65, 1024
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == '__main__':
    v = ViT(
        image_size=256,
        patch_size=24,
        num_classes=45,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.3,
        emb_dropout=0.1
    )

    img = torch.randn(2, 3, 256, 256)

    preds = v(img)  # (2, 1000)