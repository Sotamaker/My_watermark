"""
Watermark-DiT: 单流 DiT 做图像隐水印（工程可用最小实现）
- 编码器：Single-Stream DiT，将图像 patch tokens 与水印 bit tokens 拼接做联合自注意力；AdaLN-Zero 用时间步/强度调制；输出水印图像 x_tilde。
- 解码器：轻量 CNN（可替换为你的 MoE-ResNet/Transformer 解码器）从 x_tilde 中恢复 bit。
- 训练示例：综合图像重建损失（L1/LPIPS 可扩展）+ bit BCE；含简单的鲁棒性增强插槽（JPEG/噪声/模糊，可按需替换）。

接口：
  enc = WatermarkDiTEncoder(...)
  dec = WatermarkDecoder(...)
  x_tilde = enc(x, bits, t, strength)
  bits_pred = dec(attack(x_tilde))

说明：
- 本实现聚焦结构清晰与可跑性；未包含 scheduler/扩散噪声注入（若你要基于扩散训练，可将 x→x+noise 并用 v/eps 目标）。
- bits 输入为 {0,1}，在模型内映射到 token；也可切换到 {-1,1}。
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import PatchWithSecEmbed, PatchEmbed
# ---------- 基础模块 ----------

class Unpatchify(nn.Module):
    def __init__(self, img_size: int, patch: int, out_ch: int, embed: int):
        super().__init__()
        self.grid = img_size // patch
        self.proj = nn.ConvTranspose2d(embed, out_ch, patch, patch)

    def forward(self, tok):
        B, N, D = tok.shape
        g = int(math.sqrt(N))
        x = tok.transpose(1, 2).reshape(B, D, g, g)
        return self.proj(x)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (无均值中心化)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True) * (1.0 / math.sqrt(x.shape[-1]))
        return x / (norm + self.eps) * self.weight

class Attention(nn.Module):
    def __init__(self, dim, heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.d = dim // heads
        self.scale = self.d ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x

class MLP(nn.Module):
    def __init__(self, dim, ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class MMDiTBlock(nn.Module):
    """
    MMDiT 风格 Transformer 块:
      - Attention: gate-only 调制
      - MLP: scale + shift + gate 调制
      - 使用 RMSNorm
    """
    def __init__(self, dim, cond_dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = Attention(dim, heads, attn_drop=dropout, proj_drop=dropout)
        self.mlp = MLP(dim, ratio=mlp_ratio, drop=dropout)

        # 条件线性映射
        self.to_gate1 = nn.Linear(cond_dim, dim)        # Attention: 只 gate
        self.to_mod2  = nn.Linear(cond_dim, dim * 3)    # MLP: scale, shift, gate

        nn.init.zeros_(self.to_gate1.weight)
        nn.init.zeros_(self.to_gate1.bias)
        nn.init.zeros_(self.to_mod2.weight)
        nn.init.zeros_(self.to_mod2.bias)

    def forward(self, x, cond=None):
        """
        x: (B, N, D)
        cond: (B, cond_dim)
        """
        # --- Attention path: gate only ---
        h = self.norm1(x)
        attn_out = self.attn(h)
        if cond is not None:
            gate1 = self.to_gate1(cond)[:, None, :].tanh()  # (B,1,D)
            x = x + gate1 * attn_out
        else:
            x = x + attn_out

        # --- MLP path: scale, shift, gate ---
        h = self.norm2(x)
        if cond is not None:
            scale, shift, gate2 = self.to_mod2(cond).chunk(3, dim=-1)
            h = h * (1 + scale[:, None, :]) + shift[:, None, :]
            mlp_out = self.mlp(h)
            x = x + gate2[:, None, :].tanh() * mlp_out
        else:
            mlp_out = self.mlp(h)
            x = x + mlp_out
        
        return x



# ------------------------
# 编码器：Watermark-DiT（单流联合注意）
# ------------------------

class WatermarkDiTEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 512,
        width: int = 512,
        height: int = 512,
        patch_size: int = 16,
        num_patches: int = 1024,
        in_channels: int = 3,
        sec_p_dim: int = 3,
        hidden: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        nbit: int = 64,
        dropout: float = 0.0,
        pos_embed_type: str = "sincos",   # "sincos" or "learned"
        pos_embed_max_size: Optional[int] = None,  # 若要支持裁剪
        scale: float = 1.0,               # 缩放位置坐标
        extra_tokens: int = 0,            # 预留额外token数
    ):
        super().__init__()
        # Patch embedding/unpatch
        self.patchemb = PatchWithSecEmbed(width=width, height=height, patch_size=patch_size, in_chans=in_channels, sec_p_dim=sec_p_dim,
                 embed_dim=hidden, pos_embed_type=pos_embed_type, pos_embed_max_size=pos_embed_max_size, scale=scale, extra_tokens=extra_tokens)
        self.unpatch = Unpatchify(img_size, patch_size, in_channels, hidden)
        self.N_img = num_patches

        # 条件嵌入
        
        self.cond_dim = hidden
        self.sec_mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.SiLU(),
            nn.Linear(hidden * 4, hidden)
        )

        # Transformer 堆叠
        self.blocks = nn.ModuleList([
            MMDiTBlock(hidden, self.cond_dim, heads, mlp_ratio,  dropout)
            for _ in range(depth)
        ])
        self.final_norm = RMSNorm(hidden)
        self.final_proj = nn.Linear(hidden, hidden)

        # 输出融合：Conv2d 替代残差
        self.fuse = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.out_gain = nn.Parameter(torch.tensor(0.0))  # 仍可留作调节强度

    def _build_cond(self, sec_emb: torch.Tensor):
        # t: (B,)  strength: (B,) in [0,1] 控制嵌入强度
        sec_emb = self.sec_mlp(sec_emb.float())
        return sec_emb

    def forward(self, img: torch.Tensor, sec_all_emb: torch.Tensor, sec_pix_emb: torch.Tensor, sec_tok_emb: torch.Tensor, sec_tok_sum_emb: torch.Tensor):
        """
        x: (B,C,H,W) 原图
        bits: (B, bit_len) in {0,1}
        t: (B,) 时间步（若不使用扩散，可填随机数）
        strength: (B,) [0,1] 水印注入强度（训练可随机抖动）
        返回 x_tilde: (B,C,H,W)
        """
        B, C, H, W = img.shape

        img_tok = self.patchemb(img, sec_all_emb, sec_pix_emb)
        # 拼接（单流联合注意）
        tokens = torch.cat([img_tok, sec_tok_emb], dim=1)  # (B, N_img+L_bits, D)

        cond = self._build_cond(sec_tok_sum_emb)
        for blk in self.blocks:
            tokens = blk(tokens, cond)
        tokens = self.final_proj(self.final_norm(tokens))

        # 仅取回图像部分并重构
        img_tokens = tokens[:, : self.N_img, :]
        delta = self.unpatch(img_tokens)  # (B,C,H,W)
        x_tilde = self.fuse(torch.cat([img, delta], dim=1))
        return x_tilde #.clamp(-1, 1)


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
            nn.ConvTranspose2d(embed_dim//8, embed_dim//16, 4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim//16),
            nn.SiLU(),
            nn.Conv2d(embed_dim//16, 1, 1)
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
            #nn.Sigmoid()
        )

    def forward(self, x_tokens):
        mask_patch = self.patch_head(x_tokens)   # 局部细粒度
        mask_decoder = self.decoder_head(x_tokens)  # 全局一致性
        mask = self.fuse(torch.cat([mask_patch, mask_decoder], dim=1))
        return mask

class WatermarkDiTDecoder(nn.Module):
    def __init__(
        self,
        img_size: int = 512,
        width: int = 512,
        height: int = 512,
        patch_size: int = 16,
        num_patches: int = 1024,
        in_channels: int = 3,
        out_channels: int = 8,
        hidden: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        nbit: int = 64,
        dropout: float = 0.0,
        pos_embed_type: str = "sincos",   # "sincos" or "learned"
        pos_embed_max_size: Optional[int] = None,  # 若要支持裁剪
        scale: float = 1.0,               # 缩放位置坐标
        extra_tokens: int = 0,            # 预留额外token数
        use_cls_token: bool = True,
        use_learn_bit_probe: bool = True,
    ):
        super().__init__()
        # Patch embedding/unpatch
        self.patchemb = PatchEmbed(width=width, height=height, patch_size=patch_size, in_chans=in_channels,
                 embed_dim=hidden, pos_embed_type=pos_embed_type, pos_embed_max_size=pos_embed_max_size, scale=scale, extra_tokens=extra_tokens)
        self.unpatch = Unpatchify(img_size, patch_size, in_channels, hidden)
        self.N_img = num_patches
        self.cond_dim = hidden
        self.use_learn_bit_probe = use_learn_bit_probe
        if self.use_learn_bit_probe:
            self.learned_bit_probe = nn.Parameter(torch.randn(1, hidden))

        self.use_cls_token = use_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        
        self.sec_mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.SiLU(),
            nn.Linear(hidden * 4, hidden)
        )


        self.blocks = nn.ModuleList([
            MMDiTBlock(hidden, self.cond_dim, heads, mlp_ratio,  dropout)
            for _ in range(depth)
        ])
        
        self.final_norm = RMSNorm(hidden)
        self.final_proj = nn.Linear(hidden, hidden)
        self.sce_patch_extract_head = nn.Sequential(nn.Linear(hidden, nbit * nbit * 2),
                                 nn.SiLU(),
                                 nn.Linear(nbit * nbit * 2, nbit * nbit + 1),
                                 )
        
        self.sec_patch_pred_head = nn.Linear(nbit * nbit, nbit) 

        self.sce_img_pred_head = nn.Sequential(
                                nn.Conv2d(out_channels, 32, 3, stride=1, padding=1),
                                nn.GroupNorm(num_groups=16, num_channels=32, affine=True),
                                nn.SiLU(),
                                nn.Conv2d(32, 4, 3,  stride=1, padding=1),
                                nn.AdaptiveAvgPool2d(64),
                                nn.Flatten(1),
                                nn.Linear(64*64*4, nbit)
                                )
        
        #self.aggregation_
    
        self.unpatch = Unpatchify(img_size, patch_size, out_channels, hidden)
        
        self.mask_pred_head = TamperLocalizationDiT(hidden,patch_size,img_size)

        self.sig = nn.Sigmoid()
    
    def _build_cond(self, sec_emb: torch.Tensor):
        # t: (B,)  strength: (B,) in [0,1] 控制嵌入强度
        sec_emb = self.sec_mlp(sec_emb.float())
        return sec_emb


    def forward(self, img: torch.Tensor):
        """
        x: (B,C,H,W) 原图
        bits: (B, bit_len) in {0,1}
        t: (B,) 时间步（若不使用扩散，可填随机数）
        strength: (B,) [0,1] 水印注入强度（训练可随机抖动）
        返回 x_tilde: (B,C,H,W)
        """
        B, C, H, W = img.shape

        img_tok = self.patchemb(img)
        # 拼接（单流联合注意）
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), img_tok], dim=1)  # (B, N_img+L_bits, D)

        cond = self._build_cond(self.learned_bit_probe)
        for blk in self.blocks:
            tokens = blk(tokens, cond)
        tokens = self.final_proj(self.final_norm(tokens))

        # 仅取回图像部分并重构
        img_tokens_with_cls = tokens[:, : self.N_img + 1, :]
        img_tokens = tokens[:, 1 : self.N_img + 1, :]
        decode_sec_patch_info = self.sce_patch_extract_head(img_tokens_with_cls)

        decode_sec_patch_weight = self.sig(decode_sec_patch_info[:,:,0])
        decode_sec_patch_re_weight = decode_sec_patch_weight / decode_sec_patch_weight.sum(dim=1, keepdim=True)

        decode_sec_patch_feat = decode_sec_patch_info[:,:,1:]
        weighted_mean = (decode_sec_patch_re_weight.unsqueeze(-1) * decode_sec_patch_feat).sum(dim=1)

        decode_sec_patch = self.sec_patch_pred_head(weighted_mean)

        delta = self.unpatch(img_tokens)  # (B,C,H,W)
        decode_sec_img = self.sce_img_pred_head(delta)

        decode_mask = self.mask_pred_head(img_tokens)

        return decode_sec_patch_weight, decode_sec_patch, decode_sec_img, decode_mask





# img = torch.randn(2,3,512,512)

# decoder = WatermarkDiTDecoder(

# )

# a,b,c,d = decoder(img)
# ------------------------
# 解码器：轻量 CNN（可替换）
# ------------------------
class WatermarkDecoder(nn.Module):
    def __init__(self, in_ch: int = 3, hidden: int = 64, bit_len: int = 64):
        super().__init__()
        self.bit_len = bit_len
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1), nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(hidden, hidden * 2, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(hidden * 2, bit_len)

    def forward(self, x):
        B = x.shape[0]
        h = self.net(x).view(B, -1)
        logits = self.head(h)
        return logits  # (B, bit_len)

# ------------------------
# 简单攻击/退化（可扩展）：
# ------------------------
class SimpleAttacks(nn.Module):
    def __init__(self, sigma=0.01, jpeg_quality: Optional[int] = None, blur_ksize: int = 0):
        super().__init__()
        self.sigma = sigma
        self.jpeg_quality = jpeg_quality
        self.blur_ksize = blur_ksize

    def forward(self, x):
        y = x
        if self.sigma and self.sigma > 0:
            y = y + torch.randn_like(y) * self.sigma
        if self.blur_ksize and self.blur_ksize > 1:
            # 盒状滤波近似（可替换为高斯卷积）
            k = self.blur_ksize
            pad = k // 2
            weight = torch.ones(1, 1, k, k, device=x.device) / (k * k)
            y = F.pad(y, (pad, pad, pad, pad), mode='reflect')
            y = F.conv2d(y, weight.expand(y.shape[1], 1, k, k), groups=y.shape[1])
        # JPEG 可使用 Kornia/torchjpeg 等库，这里留空位
        return y.clamp(-1, 1)

# ------------------------
# 最小可运行训练样例
# ------------------------
