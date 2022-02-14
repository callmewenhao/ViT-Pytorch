import torch
import torch.nn as nn
from PatchsEmbedding import PatchEmbedding
from Encoder import Block


class VisionTransformer(nn.Module):
    def __init__(
        self, img_size, patch_size, in_channels, num_classes, embed_dim, pos_drop,
        depth, num_heads, qkv_bias=False, qkv_scale=None, atten_drop=0., proj_drop=0.,
        drop_path=0., mlp_ratio=4.,
    ):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = self.embed_dim
        self.patch_embed = PatchEmbedding(
            in_channels, img_size, patch_size, embed_dim,
        )
        self.pos_drop = nn.Dropout(pos_drop)
        self.blocks = nn.Sequential(*[
            Block(
                embed_dim, num_heads, qkv_bias, qkv_scale,
                atten_drop, proj_drop, drop_path, mlp_ratio,
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # Representation layer
        self.pre_logits = nn.Sequential(
            nn.Linear(embed_dim, self.num_features),
            nn.Tanh(),
        )
        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes)
        self.init_weights()

    def init_weights(self):
        # thanks to Aladdin Persson
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                # nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.pre_logits(x[:, 0])  # 取出Class Token位置的tensor送入linear
        return self.head(x)


def main():
    x = torch.randn(2, 3, 32, 32)
    print(f"input shape is {x.shape}")
    img_size = (32, 32)
    patch_size = (8, 8)
    embed_dim = 8 * 8 * 3
    depth = 6
    num_heads = 8
    model = VisionTransformer(
        img_size, patch_size, 3, 10, embed_dim, pos_drop=0.2,
        depth=depth, num_heads=num_heads, qkv_bias=False, qkv_scale=None,
        atten_drop=0.2, proj_drop=0.2, drop_path=0.2, mlp_ratio=4.,
    )
    out = model(x)
    print(f"output shape is {out.shape}")


if __name__ == "__main__":
    main()

