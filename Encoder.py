import torch
import torch.nn as nn
from utils import DropPath


class Attention(nn.Module):
    """
    multi-head-attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qkv_scale=None, atten_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_scale = self.head_dim ** -0.5 if qkv_scale is None else qkv_scale
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, S, E = x.shape  # Batch_size, Seq_len, Embedding_dim
        qkv = self.qkv(x)  # Batch_size, Seq_len, Embedding_dim * 3
        qkv = qkv.reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Batch_size, num_heads, Seq_len, head_dim
        a = torch.matmul(q, k.transpose(-2, -1)) * self.qkv_scale
        a = torch.softmax(a, dim=-1)  # Batch_size, num_heads, Seq_len, Seq_len
        a = self.atten_drop(a)
        x = torch.matmul(a, v)  # Batch_size, num_heads, Seq_len, head_dim
        x = x.permute(0, 2, 1, 3)  # Batch_size, Seq_len, num_heads, head_dim
        x = x.reshape(B, S, E)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qkv_scale=None, atten_drop=0., proj_drop=0.,
        drop_path=0., mlp_ratio = 4.,
    ):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.atten = Attention(dim, num_heads, qkv_bias, qkv_scale, atten_drop, proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.drop_path(self.atten(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



def main():
    x = torch.randn(2, 16, 768)
    print(f"input shape is {x.shape}")
    model = Attention(dim=768)
    out = model(x)
    print(f"output shape is {out.shape}")


if __name__ == "__main__":
    main()





