import torch
import torch.nn as nn

# Patch-to-Pixel Head
class PatchMaskHead(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, img_size=256, hidden=256):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.h = img_size // patch_size
        self.w = img_size // patch_size

        # Step1: token → patch pixels (用小 MLP)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, patch_size * patch_size)
        )

        # Step2: Conv refine
        self.refine = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        B, N, D = x.shape
        patch_mask = self.mlp(x)  # (B,N,patch²)
        patch_mask = patch_mask.view(B, self.h, self.w, self.patch_size, self.patch_size)
        patch_mask = patch_mask.permute(0,1,3,2,4).reshape(B, 1, self.h*self.patch_size, self.w*self.patch_size)
        mask_refined = self.refine(patch_mask)
        return mask_refined

# Multi-Scale Decoder Head
class MultiScaleDecoder(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, img_size=256):
        super().__init__()
        self.h = img_size // patch_size
        self.w = img_size // patch_size
        self.proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim//2, 4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim//2),
            nn.SiLU(),
            nn.ConvTranspose2d(embed_dim//2, embed_dim//4, 4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim//4),
            nn.SiLU(),
            nn.ConvTranspose2d(embed_dim//4, embed_dim//8, 4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim//8),
            nn.SiLU(),
            nn.Conv2d(embed_dim//8, 1, 1)
        )

    def forward(self, x):
        B, N, D = x.shape
        feat = x.transpose(1,2).reshape(B, D, self.h, self.w)
        feat = self.proj(feat)
        return self.decoder(feat)  # (B,1,H,W)


# Final Fusion Model
class TamperLocalizationDiT(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, img_size=256):
        super().__init__()
        self.patch_head = PatchMaskHead(embed_dim, patch_size, img_size)
        self.decoder_head = MultiScaleDecoder(embed_dim, patch_size, img_size)
        # 融合层 (两个 1通道特征图拼接 → conv)
        self.fuse = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x_tokens):
        mask_patch = self.patch_head(x_tokens)   # 局部细粒度
        mask_decoder = self.decoder_head(x_tokens)  # 全局一致性
        mask = self.fuse(torch.cat([mask_patch, mask_decoder], dim=1))
        return mask


tokens = torch.randn(2, 256, 768)  # (B,N,D), e.g. 256 tokens for 16x16 grid
model = TamperLocalizationDiT(embed_dim=768, patch_size=16, img_size=256)
mask = model(tokens)
print(mask.shape)  # torch.Size([2, 1, 256, 256])
