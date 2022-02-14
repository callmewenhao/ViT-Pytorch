import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Patch Embedding & Position Embedding & Class Token
    """
    def __init__(self, in_channels, img_size, patch_size, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) \
                           * (img_size[1] // patch_size[1])
        self.patchEmbedding = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # Patch Embedding
        x = self.patchEmbedding(x)
        x = x.flatten(2).transpose(1, 2)  # .transpose(1, 2) == .permute(0, 2, 1)
        # Class Token
        cls_token = self.cls_token.expand(B, -1, -1)  # 扩展B维度，其他维度不变
        x = torch.cat((cls_token, x), dim=1)
        # Position Embedding
        pos_embed = self.pos_embed.expand(B, -1, -1)
        return x + pos_embed


def main():
    img_size = (224, 224)
    patch_size = (16, 16)
    embed_dim = 768
    x = torch.randn(2, 3, 224, 224)
    print(f"input shape is {x.shape}")
    model = PatchEmbedding(3, img_size, patch_size, embed_dim)
    out = model(x)
    print(f"output shape is {out.shape}")


if __name__ == "__main__":
    main()

