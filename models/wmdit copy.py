#### 
# 实际运行第一版
####

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

# import math
# from typing import Optional
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .embedding import PatchWithSecEmbed, PatchEmbed
# # ---------- 基础模块 ----------

# class Unpatchify(nn.Module):
#     def __init__(self, img_size: int, patch: int, out_ch: int, embed: int):
#         super().__init__()
#         self.grid = img_size // patch
#         self.proj = nn.ConvTranspose2d(embed, out_ch, patch, patch)

#     def forward(self, tok):
#         B, N, D = tok.shape
#         g = int(math.sqrt(N))
#         x = tok.transpose(1, 2).reshape(B, D, g, g)
#         return self.proj(x)

# class RMSNorm(nn.Module):
#     """Root Mean Square Layer Normalization (无均值中心化)."""
#     def __init__(self, dim, eps=1e-6):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))

#     def forward(self, x):
#         norm = x.norm(2, dim=-1, keepdim=True) * (1.0 / math.sqrt(x.shape[-1]))
#         return x / (norm + self.eps) * self.weight

# class Attention(nn.Module):
#     def __init__(self, dim, heads, attn_drop=0.0, proj_drop=0.0):
#         super().__init__()
#         assert dim % heads == 0
#         self.h = heads
#         self.d = dim // heads
#         self.scale = self.d ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3)
#         self.proj = nn.Linear(dim, dim)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         attn = (q * self.scale) @ k.transpose(-2, -1)
#         attn = attn.softmax(-1)
#         attn = self.attn_drop(attn)
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj_drop(self.proj(x))
#         return x

# class MLP(nn.Module):
#     def __init__(self, dim, ratio=4.0, drop=0.0):
#         super().__init__()
#         hidden = int(dim * ratio)
#         self.fc1 = nn.Linear(dim, hidden)
#         self.act = nn.GELU()
#         self.fc2 = nn.Linear(hidden, dim)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         return self.drop(self.fc2(self.act(self.fc1(x))))

# class MMDiTBlock(nn.Module):
#     """
#     MMDiT 风格 Transformer 块:
#       - Attention: gate-only 调制
#       - MLP: scale + shift + gate 调制
#       - 使用 RMSNorm
#     """
#     def __init__(self, dim, cond_dim, heads, mlp_ratio=4.0, dropout=0.0):
#         super().__init__()
#         self.norm1 = RMSNorm(dim)
#         self.norm2 = RMSNorm(dim)
#         self.attn = Attention(dim, heads, attn_drop=dropout, proj_drop=dropout)
#         self.mlp = MLP(dim, ratio=mlp_ratio, drop=dropout)

#         # 条件线性映射
#         self.to_gate1 = nn.Linear(cond_dim, dim)        # Attention: 只 gate
#         self.to_mod2  = nn.Linear(cond_dim, dim * 3)    # MLP: scale, shift, gate

#         nn.init.zeros_(self.to_gate1.weight)
#         nn.init.zeros_(self.to_gate1.bias)
#         nn.init.zeros_(self.to_mod2.weight)
#         nn.init.zeros_(self.to_mod2.bias)

#     def forward(self, x, cond=None):
#         """
#         x: (B, N, D)
#         cond: (B, cond_dim)
#         """
#         # --- Attention path: gate only ---
#         h = self.norm1(x)
#         attn_out = self.attn(h)
#         if cond is not None:
#             gate1 = self.to_gate1(cond)[:, None, :].tanh()  # (B,1,D)
#             x = x + gate1 * attn_out
#         else:
#             x = x + attn_out

#         # --- MLP path: scale, shift, gate ---
#         h = self.norm2(x)
#         if cond is not None:
#             scale, shift, gate2 = self.to_mod2(cond).chunk(3, dim=-1)
#             h = h * (1 + scale[:, None, :]) + shift[:, None, :]
#             mlp_out = self.mlp(h)
#             x = x + gate2[:, None, :].tanh() * mlp_out
#         else:
#             mlp_out = self.mlp(h)
#             x = x + mlp_out
        
#         return x


# # ------------------------
# # 编码器：Watermark-DiT（单流联合注意）
# # ------------------------

# class WatermarkDiTEncoder(nn.Module):
#     def __init__(
#         self,
#         img_size: int = 512,
#         width: int = 512,
#         height: int = 512,
#         patch_size: int = 16,
#         num_patches: int = 1024,
#         in_channels: int = 3,
#         sec_p_dim: int = 0,
#         hidden: int = 768,
#         depth: int = 12,
#         heads: int = 12,
#         mlp_ratio: float = 4.0,
#         nbit: int = 64,
#         dropout: float = 0.0,
#         pos_embed_type: str = "sincos",   # "sincos" or "learned"
#         pos_embed_max_size: Optional[int] = None,  # 若要支持裁剪
#         scale: float = 1.0,               # 缩放位置坐标
#         extra_tokens: int = 0,            # 预留额外token数
#     ):
#         super().__init__()
#         # Patch embedding/unpatch
#         self.patchemb = PatchWithSecEmbed(width=width, height=height, patch_size=patch_size, in_chans=in_channels, sec_p_dim=sec_p_dim,
#                  embed_dim=hidden, pos_embed_type=pos_embed_type, pos_embed_max_size=pos_embed_max_size, scale=scale, extra_tokens=extra_tokens)
#         self.unpatch = Unpatchify(img_size, patch_size, in_channels, hidden)
#         self.N_img = num_patches

#         # 条件嵌入
        
#         self.cond_dim = hidden
#         self.sec_mlp = nn.Sequential(
#             nn.Linear(hidden, hidden * 4),
#             nn.SiLU(),
#             nn.Linear(hidden * 4, hidden)
#         )

#         # Transformer 堆叠
#         self.blocks = nn.ModuleList([
#             MMDiTBlock(hidden, self.cond_dim, heads, mlp_ratio,  dropout)
#             for _ in range(depth)
#         ])
#         self.final_norm = RMSNorm(hidden)
#         self.final_proj = nn.Linear(hidden, hidden)

#         # 输出融合：Conv2d 替代残差
#         self.fuse = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
#         self.out_gain = nn.Parameter(torch.tensor(0.0))  # 仍可留作调节强度

#     def _build_cond(self, sec_emb: torch.Tensor):
#         # t: (B,)  strength: (B,) in [0,1] 控制嵌入强度
#         sec_emb = self.sec_mlp(sec_emb.float())
#         return sec_emb

#     def forward(self, img: torch.Tensor, sec_pos_emb: torch.Tensor, sec_cond_emb: torch.Tensor, sec_tok_emb: torch.Tensor):
#         """
#         x: (B,C,H,W) 原图
#         bits: (B, bit_len) in {0,1}
#         t: (B,) 时间步（若不使用扩散，可填随机数）
#         strength: (B,) [0,1] 水印注入强度（训练可随机抖动）
#         返回 x_tilde: (B,C,H,W)
#         """
#         B, C, H, W = img.shape

#         img_tok = self.patchemb(img, sec_pos_emb)  #, sec_pix_emb
#         # 拼接（单流联合注意）
#         if self.use_sec_tok:
#             tokens = torch.cat([img_tok, sec_tok_emb], dim=1)  # (B, N_img+L_bits, D)
#         else:
#             tokens = img_tok

#         cond = self._build_cond(sec_cond_emb)
#         for blk in self.blocks:
#             tokens = blk(tokens, cond)
#         tokens = self.final_proj(self.final_norm(tokens))

#         # 仅取回图像部分并重构
#         img_tokens = tokens[:, : self.N_img, :]
#         delta = self.unpatch(img_tokens)  # (B,C,H,W)
#         x_tilde = self.fuse(torch.cat([img, delta], dim=1))
#         return x_tilde #.clamp(-1, 1)


# # Patch-to-Pixel Head
# class PatchMaskHead(nn.Module):
#     def __init__(self, embed_dim=768, patch_size=16, img_size=256, hidden=256):
#         super().__init__()
#         self.patch_size = patch_size
#         self.img_size = img_size
#         self.h = img_size // patch_size
#         self.w = img_size // patch_size

#         # Step1: token → patch pixels (用小 MLP)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, hidden),
#             nn.SiLU(),
#             nn.Linear(hidden, patch_size * patch_size)
#         )

#         # Step2: Conv refine
#         self.refine = nn.Sequential(
#             nn.Conv2d(1, 32, 3, padding=1),
#             nn.GroupNorm(8, 32),
#             nn.SiLU(),
#             nn.Conv2d(32, 1, 3, padding=1)
#         )

#     def forward(self, x):
#         B, N, D = x.shape
#         patch_mask = self.mlp(x)  # (B,N,patch²)
#         patch_mask = patch_mask.view(B, self.h, self.w, self.patch_size, self.patch_size)
#         patch_mask = patch_mask.permute(0,1,3,2,4).reshape(B, 1, self.h*self.patch_size, self.w*self.patch_size)
#         mask_refined = self.refine(patch_mask)
#         return mask_refined

# # Multi-Scale Decoder Head
# class MultiScaleDecoder(nn.Module):
#     def __init__(self, embed_dim=768, patch_size=16, img_size=256):
#         super().__init__()
#         self.h = img_size // patch_size
#         self.w = img_size // patch_size
#         self.proj = nn.Conv2d(embed_dim, embed_dim, 1)
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(embed_dim, embed_dim//2, 4, stride=2, padding=1),
#             nn.BatchNorm2d(embed_dim//2),
#             nn.SiLU(),
#             nn.ConvTranspose2d(embed_dim//2, embed_dim//4, 4, stride=2, padding=1),
#             nn.BatchNorm2d(embed_dim//4),
#             nn.SiLU(),
#             nn.ConvTranspose2d(embed_dim//4, embed_dim//8, 4, stride=2, padding=1),
#             nn.BatchNorm2d(embed_dim//8),
#             nn.SiLU(),
#             nn.ConvTranspose2d(embed_dim//8, embed_dim//16, 4, stride=2, padding=1),
#             nn.BatchNorm2d(embed_dim//16),
#             nn.SiLU(),
#             nn.Conv2d(embed_dim//16, 1, 1)
#         )

#     def forward(self, x):
#         B, N, D = x.shape
#         feat = x.transpose(1,2).reshape(B, D, self.h, self.w)
#         feat = self.proj(feat)
#         return self.decoder(feat)  # (B,1,H,W)


# # Final Fusion Model
# class TamperLocalizationDiT(nn.Module):
#     def __init__(self, embed_dim=768, patch_size=16, img_size=256):
#         super().__init__()
#         self.patch_head = PatchMaskHead(embed_dim, patch_size, img_size)
#         self.decoder_head = MultiScaleDecoder(embed_dim, patch_size, img_size)
#         # 融合层 (两个 1通道特征图拼接 → conv)
#         self.fuse = nn.Sequential(
#             nn.Conv2d(2, 1, 3, padding=1),
#             #nn.Sigmoid()
#         )

#     def forward(self, x_tokens):
#         mask_patch = self.patch_head(x_tokens)   # 局部细粒度
#         mask_decoder = self.decoder_head(x_tokens)  # 全局一致性
#         mask = self.fuse(torch.cat([mask_patch, mask_decoder], dim=1))
#         return mask

# class WatermarkDiTDecoder(nn.Module):
#     def __init__(
#         self,
#         img_size: int = 512,
#         width: int = 512,
#         height: int = 512,
#         patch_size: int = 16,
#         num_patches: int = 1024,
#         in_channels: int = 3,
#         out_channels: int = 8,
#         hidden: int = 768,
#         depth: int = 12,
#         heads: int = 12,
#         mlp_ratio: float = 4.0,
#         nbit: int = 64,
#         dropout: float = 0.0,
#         pos_embed_type: str = "sincos",   # "sincos" or "learned"
#         pos_embed_max_size: Optional[int] = None,  # 若要支持裁剪
#         scale: float = 1.0,               # 缩放位置坐标
#         extra_tokens: int = 0,            # 预留额外token数
#         use_cls_token: bool = True,
#         use_learn_bit_probe: bool = True,
#     ):
#         super().__init__()
#         # Patch embedding/unpatch
#         self.patchemb = PatchEmbed(width=width, height=height, patch_size=patch_size, in_chans=in_channels,
#                  embed_dim=hidden, pos_embed_type=pos_embed_type, pos_embed_max_size=pos_embed_max_size, scale=scale, extra_tokens=extra_tokens)
#         self.unpatch = Unpatchify(img_size, patch_size, in_channels, hidden)
#         self.N_img = num_patches
#         self.cond_dim = hidden
#         self.use_learn_bit_probe = use_learn_bit_probe
#         if self.use_learn_bit_probe:
#             self.learned_bit_probe = nn.Parameter(torch.randn(1, hidden))

#         self.use_cls_token = use_cls_token
#         if self.use_cls_token:
#             self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        
#         self.sec_mlp = nn.Sequential(
#             nn.Linear(hidden, hidden * 4),
#             nn.SiLU(),
#             nn.Linear(hidden * 4, hidden)
#         )


#         self.blocks = nn.ModuleList([
#             MMDiTBlock(hidden, self.cond_dim, heads, mlp_ratio,  dropout)
#             for _ in range(depth)
#         ])
        
#         self.final_norm = RMSNorm(hidden)
#         self.final_proj = nn.Linear(hidden, hidden)
#         self.sce_patch_extract_head = nn.Sequential(nn.Linear(hidden, nbit * nbit * 2),
#                                  nn.SiLU(),
#                                  nn.Linear(nbit * nbit * 2, nbit * nbit + 1),
#                                  )
        
#         self.sec_patch_pred_head = nn.Linear(nbit * nbit, nbit) 

#         self.sce_img_pred_head = nn.Sequential(
#                                 nn.Conv2d(out_channels, 32, 3, stride=1, padding=1),
#                                 nn.GroupNorm(num_groups=16, num_channels=32, affine=True),
#                                 nn.SiLU(),
#                                 nn.Conv2d(32, 4, 3,  stride=1, padding=1),
#                                 nn.AdaptiveAvgPool2d(64),
#                                 nn.Flatten(1),
#                                 nn.Linear(64*64*4, nbit)
#                                 )
        
#         #self.aggregation_
    
#         self.unpatch = Unpatchify(img_size, patch_size, out_channels, hidden)
        
#         self.mask_pred_head = TamperLocalizationDiT(hidden,patch_size,img_size)

#         self.sig = nn.Sigmoid()
    
#     def _build_cond(self, sec_emb: torch.Tensor):
#         # t: (B,)  strength: (B,) in [0,1] 控制嵌入强度
#         sec_emb = self.sec_mlp(sec_emb.float())
#         return sec_emb


#     def forward(self, img: torch.Tensor):
#         """
#         x: (B,C,H,W) 原图
#         bits: (B, bit_len) in {0,1}
#         t: (B,) 时间步（若不使用扩散，可填随机数）
#         strength: (B,) [0,1] 水印注入强度（训练可随机抖动）
#         返回 x_tilde: (B,C,H,W)
#         """
#         B, C, H, W = img.shape

#         img_tok = self.patchemb(img)
#         # 拼接（单流联合注意）
#         tokens = torch.cat([self.cls_token.expand(B, -1, -1), img_tok], dim=1)  # (B, N_img+L_bits, D)

#         cond = self._build_cond(self.learned_bit_probe)
#         for blk in self.blocks:
#             tokens = blk(tokens, cond)
#         tokens = self.final_proj(self.final_norm(tokens))

#         # 仅取回图像部分并重构
#         img_tokens_with_cls = tokens[:, : self.N_img + 1, :]
#         img_tokens = tokens[:, 1 : self.N_img + 1, :]
#         decode_sec_patch_info = self.sce_patch_extract_head(img_tokens_with_cls)

#         decode_sec_patch_weight = self.sig(decode_sec_patch_info[:,:,0])
#         decode_sec_patch_re_weight = decode_sec_patch_weight / decode_sec_patch_weight.sum(dim=1, keepdim=True)

#         decode_sec_patch_feat = decode_sec_patch_info[:,:,1:]
#         weighted_mean = (decode_sec_patch_re_weight.unsqueeze(-1) * decode_sec_patch_feat).sum(dim=1)

#         decode_sec_patch = self.sec_patch_pred_head(weighted_mean)

#         delta = self.unpatch(img_tokens)  # (B,C,H,W)
#         decode_sec_img = self.sce_img_pred_head(delta)

#         decode_mask = self.mask_pred_head(img_tokens)

#         return decode_sec_patch_weight, decode_sec_patch, decode_sec_img, decode_mask





# # img = torch.randn(2,3,512,512)

# # decoder = WatermarkDiTDecoder(

# # )

# # a,b,c,d = decoder(img)
# # ------------------------
# # 解码器：轻量 CNN（可替换）
# # ------------------------
# class WatermarkDecoder(nn.Module):
#     def __init__(self, in_ch: int = 3, hidden: int = 64, bit_len: int = 64):
#         super().__init__()
#         self.bit_len = bit_len
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, hidden, 3, 1, 1), nn.ReLU(inplace=True),
#             nn.Conv2d(hidden, hidden, 3, 1, 1), nn.ReLU(inplace=True),
#             nn.AvgPool2d(2),
#             nn.Conv2d(hidden, hidden * 2, 3, 1, 1), nn.ReLU(inplace=True),
#             nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1), nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d(1),
#         )
#         self.head = nn.Linear(hidden * 2, bit_len)

#     def forward(self, x):
#         B = x.shape[0]
#         h = self.net(x).view(B, -1)
#         logits = self.head(h)
#         return logits  # (B, bit_len)

# # ------------------------
# # 简单攻击/退化（可扩展）：
# # ------------------------
# class SimpleAttacks(nn.Module):
#     def __init__(self, sigma=0.01, jpeg_quality: Optional[int] = None, blur_ksize: int = 0):
#         super().__init__()
#         self.sigma = sigma
#         self.jpeg_quality = jpeg_quality
#         self.blur_ksize = blur_ksize

#     def forward(self, x):
#         y = x
#         if self.sigma and self.sigma > 0:
#             y = y + torch.randn_like(y) * self.sigma
#         if self.blur_ksize and self.blur_ksize > 1:
#             # 盒状滤波近似（可替换为高斯卷积）
#             k = self.blur_ksize
#             pad = k // 2
#             weight = torch.ones(1, 1, k, k, device=x.device) / (k * k)
#             y = F.pad(y, (pad, pad, pad, pad), mode='reflect')
#             y = F.conv2d(y, weight.expand(y.shape[1], 1, k, k), groups=y.shape[1])
#         # JPEG 可使用 Kornia/torchjpeg 等库，这里留空位
#         return y.clamp(-1, 1)

# # ------------------------
# # 最小可运行训练样例
# # ------------------------


# class BitDiTDecoder(nn.Module):
#     def __init__(self, nbit=64, jpeg_quality: Optional[int] = None, blur_ksize: int = 0):


















# """
# MM-DiT: Multi-Modal Diffusion Transformer（简化但工程可用）
# - 思路参考 SD3 的 MMDiT：将图像 patch token 与文本 token 拼接后做联合自注意力（co-attention），
#   通过 AdaLN-Zero 用扩散时间步（以及可选类条件）调制所有块。
# - 文本侧默认已是嵌入后的 token（如 T5/CLIP 的最后隐表示），在本实现中以 (B, L_txt, D) 直接输入。
# - 图像侧通过 PatchEmbed/Unpatchify 实现 (B, C, H, W) ↔ (B, N_img, D)。
# - 支持：
#   * 时间步正弦嵌入 + MLP
#   * 图像 2D 正弦位置编码；文本 1D 正弦位置编码
#   * 模态类型嵌入（image/text）
#   * 文本 padding mask（不会影响图像 token 的注意力）
#   * Classifier-Free Guidance（可选的 label/null 条件）

# 接口：
#   out = MMDiT(...)(img, t, txt_emb, txt_attn_mask, y=None)
#   - img: (B, C, H, W)
#   - t:   (B,) 扩散时间
#   - txt_emb: (B, L_txt, D) 预先编码好的文本隐表示（与 hidden_size 一致）
#   - txt_attn_mask: (B, L_txt) 1=有效, 0=padding（或 None）
#   - y: (B,) 类别 id 或 None（若启用 num_classes>0）
#   返回：与 UNet 对齐的 (B, C, H, W)（预测噪声/velocity）

# 注意：
# - 为聚焦结构与接口，本实现不包含调度器与训练循环。
# - 若你的文本编码维度不等于 hidden_size，可在输入前接一层线性映射对齐维度。
# """
# from __future__ import annotations
# import math
# from dataclasses import dataclass
# from typing import Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ------------------------
# # 基础：正弦位置编码（1D/2D）& 时间步/标签嵌入
# # ------------------------

# def build_sincos_1d(L: int, D: int, device=None) -> torch.Tensor:
#     """[L, D] 的 1D 正弦位置编码。"""
#     half = D // 2
#     pos = torch.arange(L, device=device)[:, None]
#     freqs = torch.exp(torch.linspace(0, math.log(10000), steps=half, device=device))[None, :]
#     ang = pos / freqs
#     emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
#     if D % 2 == 1:
#         emb = F.pad(emb, (0,1))
#     return emb  # (L, D)


# def build_sincos_2d(H: int, W: int, D: int, device=None) -> torch.Tensor:
#     """[H*W, D] 的 2D 正弦位置编码，x/y 各占一半。"""
#     assert D % 2 == 0
#     D_half = D // 2
#     emb_h = build_sincos_1d(H, D_half, device=device)  # (H, D/2)
#     emb_w = build_sincos_1d(W, D_half, device=device)  # (W, D/2)
#     # 外积式拼接：每个网格位置 (i,j) → [emb_h[i], emb_w[j]]
#     emb = (
#         emb_h[:, None, :].expand(H, W, D_half),
#         emb_w[None, :, :].expand(H, W, D_half)
#     )
#     emb = torch.cat(emb, dim=-1).reshape(H * W, D)
#     return emb  # (H*W, D)


# class SinTimeEmbedding(nn.Module):
#     def __init__(self, hidden_size: int, time_embed_dim: Optional[int] = None):
#         super().__init__()
#         time_embed_dim = time_embed_dim or hidden_size
#         self.time_embed_dim = time_embed_dim
#         self.mlp = nn.Sequential(
#             nn.Linear(time_embed_dim, hidden_size * 4),
#             nn.SiLU(),
#             nn.Linear(hidden_size * 4, hidden_size),
#         )

#     @staticmethod
#     def sin_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
#         device = t.device
#         half = dim // 2
#         freqs = torch.exp(torch.linspace(0, math.log(10000), steps=half, device=device))
#         ang = t[:, None] / freqs[None, :]
#         emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
#         if dim % 2 == 1:
#             emb = F.pad(emb, (0,1))
#         return emb

#     def forward(self, t: torch.Tensor) -> torch.Tensor:
#         if t.dim() == 0:
#             t = t[None]
#         t = t.float()
#         s = self.sin_embed(t, self.time_embed_dim)
#         return self.mlp(s)


# class LabelEmbedding(nn.Module):
#     def __init__(self, num_classes: int, hidden_size: int, p_drop: float = 0.1):
#         super().__init__()
#         self.emb = nn.Embedding(num_classes, hidden_size)
#         self.null = nn.Parameter(torch.zeros(1, hidden_size))
#         self.p_drop = p_drop

#     def forward(self, y: Optional[torch.Tensor]) -> torch.Tensor:
#         if y is None:
#             return self.null  # (1, D)
#         # CFG：训练时随机置空
#         if self.training and self.p_drop > 0:
#             mask = torch.rand_like(y.float()) < self.p_drop
#             y = y.masked_fill(mask, -1)
#         out = []
#         for yi in y:
#             out.append(self.null if yi < 0 else self.emb(yi))
#         return torch.stack(out, 0)  # (B, D)


# # ------------------------
# # ViT 组件（注意：这里的注意力在拼接后的 img+txt 上做联合自注意）
# # ------------------------
# class PatchEmbed(nn.Module):
#     def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
#         super().__init__()
#         assert img_size % patch_size == 0
#         self.grid = img_size // patch_size
#         self.num_patches = self.grid * self.grid
#         self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # (B,C,H,W) → (B, N_img, D)
#         x = self.proj(x)
#         x = x.flatten(2).transpose(1, 2)
#         return x


# class Unpatchify(nn.Module):
#     def __init__(self, img_size: int, patch_size: int, out_chans: int, embed_dim: int):
#         super().__init__()
#         self.grid = img_size // patch_size
#         self.proj = nn.ConvTranspose2d(embed_dim, out_chans, patch_size, patch_size)

#     def forward(self, tokens: torch.Tensor) -> torch.Tensor:
#         # (B, N_img, D) → (B,C,H,W)
#         B, N, D = tokens.shape
#         g = int(math.sqrt(N))
#         x = tokens.transpose(1, 2).reshape(B, D, g, g)
#         return self.proj(x)


# class Attention(nn.Module):
#     def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
#         super().__init__()
#         assert dim % num_heads == 0
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=True)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         # x: (B, N, D); attn_mask: (B, 1, N, N) with 0=mask, 1=keep
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, Hd)
#         attn = (q * self.scale) @ k.transpose(-2, -1)  # (B,H,N,N)
#         if attn_mask is not None:
#             attn = attn.masked_fill(attn_mask == 0, float('-inf'))
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class MLP(nn.Module):
#     def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
#         super().__init__()
#         hidden = int(dim * mlp_ratio)
#         self.fc1 = nn.Linear(dim, hidden)
#         self.act = nn.GELU()
#         self.fc2 = nn.Linear(hidden, dim)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.fc1(x); x = self.act(x); x = self.drop(x)
#         x = self.fc2(x); x = self.drop(x)
#         return x


# class AdaLNZero(nn.Module):
#     def __init__(self, dim: int, cond_dim: int):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim, elementwise_affine=False)
#         self.to_params = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 3))
#         nn.init.zeros_(self.to_params[-1].weight)
#         nn.init.zeros_(self.to_params[-1].bias)

#     def forward(self, x: torch.Tensor, cond: torch.Tensor, fn: nn.Module, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         scale, shift, gate = self.to_params(cond).chunk(3, dim=-1)
#         h = self.norm(x)
#         h = h * (1 + scale) + shift
#         # 注意：fn 可能是 Attention 或 MLP；Attention 需要 attn_mask
#         if isinstance(fn, Attention):
#             h = fn(h, attn_mask)
#         else:
#             h = fn(h)
#         return x + gate * h


# class MMDiTBlock(nn.Module):
#     def __init__(self, dim: int, num_heads: int, mlp_ratio: float, cond_dim: int, drop: float = 0.0):
#         super().__init__()
#         self.attn = Attention(dim, num_heads)
#         self.mlp = MLP(dim, mlp_ratio, drop)
#         self.adaln1 = AdaLNZero(dim, cond_dim)
#         self.adaln2 = AdaLNZero(dim, cond_dim)

#     def forward(self, x: torch.Tensor, cond: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
#         x = self.adaln1(x, cond, self.attn, attn_mask)
#         x = self.adaln2(x, cond, self.mlp, None)
#         return x


# # ------------------------
# # 主干：MMDiT
# # ------------------------
# @dataclass
# class MMDiTConfig:
#     img_size: int = 256
#     patch_size: int = 2
#     in_channels: int = 3
#     hidden_size: int = 1024
#     depth: int = 16
#     num_heads: int = 16
#     mlp_ratio: float = 4.0
#     dropout: float = 0.0
#     num_classes: int = 0  # 0 表示无类别条件
#     prediction_type: str = "eps"  # or "v"


# class MMDiT(nn.Module):
#     def __init__(self,
#                  img_size: int = 256,
#                  patch_size: int = 2,
#                  in_channels: int = 3,
#                  hidden_size: int = 1024,
#                  depth: int = 16,
#                  num_heads: int = 16,
#                  mlp_ratio: float = 4.0,
#                  dropout: float = 0.0,
#                  num_classes: int = 0,
#                  prediction_type: str = "eps",
#                  max_text_len: int = 256):
#         super().__init__()
#         self.cfg = MMDiTConfig(img_size, patch_size, in_channels, hidden_size, depth, num_heads, mlp_ratio, dropout, num_classes, prediction_type)
#         self.hidden = hidden_size

#         # 图像 patch ↔ token
#         self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
#         self.unpatchify = Unpatchify(img_size, patch_size, in_channels, hidden_size)
#         self.N_img = self.patch_embed.num_patches

#         # 位置与类型嵌入
#         self.pos_img = nn.Parameter(build_sincos_2d(self.patch_embed.grid, self.patch_embed.grid, hidden_size))  # (N_img, D)
#         self.pos_txt = nn.Parameter(build_sincos_1d(max_text_len, hidden_size))  # (L_txt_max, D)
#         self.type_img = nn.Parameter(torch.zeros(1, 1, hidden_size))
#         self.type_txt = nn.Parameter(torch.zeros(1, 1, hidden_size))

#         # 条件嵌入
#         self.time_embed = SinTimeEmbedding(hidden_size)
#         self.label_embed = LabelEmbedding(num_classes, hidden_size) if num_classes > 0 else None
#         self.cond_dim = hidden_size if num_classes == 0 else hidden_size * 2

#         # block 堆叠
#         self.blocks = nn.ModuleList([
#             MMDiTBlock(hidden_size, num_heads, mlp_ratio, self.cond_dim, dropout) for _ in range(depth)
#         ])
#         self.final_adaln = AdaLNZero(hidden_size, self.cond_dim)
#         self.head = nn.Linear(hidden_size, hidden_size)

#     # ---------- helpers ----------
#     def _build_cond(self, t: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
#         t_emb = self.time_embed(t)
#         if self.label_embed is not None:
#             y_emb = self.label_embed(y)
#             if y is None:
#                 y_emb = y_emb.expand(t_emb.shape[0], -1)
#             cond = torch.cat([t_emb, y_emb], dim=-1)
#         else:
#             cond = t_emb
#         return cond  # (B, cond_dim)

#     def _concat_tokens(self, img_tok: torch.Tensor, txt_emb: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
#         """返回拼接后的 tokens 和一个索引，用于从联合序列中还原图像部分。"""
#         B, N_img, D = img_tok.shape
#         if txt_emb is None:
#             x = img_tok
#         else:
#             x = torch.cat([img_tok, txt_emb], dim=1)
#         img_idx = torch.arange(N_img, device=img_tok.device)
#         return x, img_idx

#     def _build_attn_mask(self, B: int, N_img: int, L_txt: int, txt_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
#         """
#         返回 (B,1,N,N) 的注意力 mask，文本 padding 不可见；图像都可见。
#         1=keep, 0=mask。若 txt_mask is None，返回 None。
#         """
#         if txt_mask is None:
#             return None
#         N = N_img + L_txt
#         base = torch.ones(B, N, device=txt_mask.device)
#         # 将 padding 的文本 token 屏蔽：位置 i 若是 padding，则其它 token 对它的注意力应被 mask。
#         # 我们构造一个 (B,N) 的可见标志，再外积得到 (B,N,N)。
#         vis = base.clone()
#         vis[:, N_img:] = txt_mask.float()  # (B, N) 前 N_img 为图像，后面是文本
#         attn_mask = (vis[:, None, :] * vis[:, :, None]).to(dtype=torch.bool)  # (B,N,N)
#         return attn_mask[:, None, :, :]  # (B,1,N,N)

#     # ---------- forward ----------
#     def forward(self,
#                 img: torch.Tensor,
#                 t: torch.Tensor,
#                 txt_emb: Optional[torch.Tensor] = None,
#                 txt_attn_mask: Optional[torch.Tensor] = None,
#                 y: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         img: (B,C,H,W)
#         t:   (B,)
#         txt_emb: (B,L_txt,D) 或 None（若不用文本条件）
#         txt_attn_mask: (B,L_txt) 1=有效,0=padding（或 None）
#         y: (B,) 类别 id 或 None
#         返回：(B,C,H,W)
#         """
#         B, C, H, W = img.shape
#         assert H == self.cfg.img_size and W == self.cfg.img_size
#         # 图像 tokens
#         img_tok = self.patch_embed(img)  # (B,N_img,D)
#         # 位置 + 类型
#         img_tok = img_tok + self.pos_img[None, :, :] + self.type_img

#         # 文本 tokens：对齐维度并加上 1D 位置与类型
#         if txt_emb is not None:
#             B2, L_txt, D_txt = txt_emb.shape
#             assert B2 == B and D_txt == self.hidden, "txt_emb 维度需与 hidden_size 对齐"
#             pos = self.pos_txt[:L_txt, :][None, :, :]  # (1,L,D)
#             txt_tok = txt_emb + pos + self.type_txt
#         else:
#             L_txt = 0
#             txt_tok = None

#         # 拼接
#         x, img_idx = self._concat_tokens(img_tok, txt_tok)
#         # 构造联合注意力 mask
#         attn_mask = self._build_attn_mask(B, self.N_img, L_txt, txt_attn_mask)

#         # 条件
#         cond = self._build_cond(t, y)

#         # blocks
#         for blk in self.blocks:
#             x = blk(x, cond, attn_mask)
#         x = self.final_adaln(x, cond, self.head)

#         # 取回图像部分并还原
#         img_tokens = x[:, :self.N_img, :]
#         out = self.unpatchify(img_tokens)
#         return out


# # ------------------------
# # 最小可运行示例
# # ------------------------
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = MMDiT(
#         img_size=64, patch_size=2, in_channels=3,
#         hidden_size=512, depth=8, num_heads=8, mlp_ratio=4.0,
#         num_classes=0, prediction_type="eps", max_text_len=128
#     ).to(device)

#     B = 2
#     img = torch.randn(B, 3, 64, 64, device=device)
#     t = torch.randint(0, 1000, (B,), device=device)

#     # 模拟一个文本 encoder 的输出（已对齐到 D=hidden_size）
#     L_txt = 20
#     txt_emb = torch.randn(B, L_txt, 512, device=device)
#     # 文本 padding mask：前 16 个有效，后 4 个 padding（举例）
#     txt_mask = torch.ones(B, L_txt, device=device)
#     txt_mask[:, -4:] = 0

#     target = torch.randn_like(img)
#     pred = model(img, t, txt_emb, txt_mask)
#     loss = F.mse_loss(pred, target)
#     loss.backward()
#     print("MMDiT forward ok, loss=", float(loss))





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
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------
# 工具：正弦位置编码（1D/2D）
# ------------------------

def build_sincos_1d(L: int, D: int, device=None):
    half = D // 2
    pos = torch.arange(L, device=device)[:, None]
    freqs = torch.exp(torch.linspace(0, math.log(10000), steps=half, device=device))[None, :]
    ang = pos / freqs
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
    if D % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb  # (L,D)


def build_sincos_2d(H: int, W: int, D: int, device=None):
    assert D % 2 == 0
    Dh = D // 2
    ey = build_sincos_1d(H, Dh, device)
    ex = build_sincos_1d(W, Dh, device)
    emb = torch.cat(
        [ey[:, None, :].expand(H, W, Dh), ex[None, :, :].expand(H, W, Dh)], dim=-1
    ).reshape(H * W, D)
    return emb

# ------------------------
# 时间步与水印 bit 嵌入
# ------------------------
class SinTimeEmbedding(nn.Module):
    def __init__(self, hidden: int, time_dim: Optional[int] = None):
        super().__init__()
        time_dim = time_dim or hidden
        self.time_dim = time_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden * 4), nn.SiLU(), nn.Linear(hidden * 4, hidden)
        )

    @staticmethod
    def sin_embed(t: torch.Tensor, dim: int):
        device = t.device
        half = dim // 2
        freqs = torch.exp(torch.linspace(0, math.log(10000), steps=half, device=device))
        ang = t[:, None] / freqs[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb

    def forward(self, t: torch.Tensor):
        if t.dim() == 0:
            t = t[None]
        s = self.sin_embed(t.float(), self.time_dim)
        return self.mlp(s)


class BitTokenizer(nn.Module):
    """把 {0,1} 序列转成 token 嵌入；可选 learnable [NULL] 占位（支持随机 bit dropout）。"""
    def __init__(self, bit_len: int, hidden: int, p_dropout_bits: float = 0.0):
        super().__init__()
        self.bit_len = bit_len
        self.emb0 = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.emb1 = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.null = nn.Parameter(torch.zeros(1, 1, hidden))
        self.p_dropout = p_dropout_bits
        self.pos = nn.Parameter(build_sincos_1d(bit_len, hidden))

    def forward(self, bits: torch.Tensor):
        """bits: (B, bit_len) in {0,1} (torch.long 或 float)
        返回: (B, bit_len, hidden)
        """
        B, L = bits.shape
        bits = bits.float()
        if self.training and self.p_dropout > 0:
            drop = (torch.rand(B, L, device=bits.device) < self.p_dropout).float()
            # 置空位使用 null token
            emb = bits.unsqueeze(-1) * self.emb1 + (1 - bits).unsqueeze(-1) * self.emb0
            emb = emb + self.pos[None, :, :]
            emb = emb * (1 - drop.unsqueeze(-1)) + self.null * drop.unsqueeze(-1)
        else:
            emb = bits.unsqueeze(-1) * self.emb1 + (1 - bits).unsqueeze(-1) * self.emb0
            emb = emb + self.pos[None, :, :]
        return emb

# ------------------------
# ViT 组件：PatchEmbed/Unpatchify/Attention/MLP/AdaLN-Zero
# ------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch: int, in_ch: int, embed: int):
        super().__init__()
        assert img_size % patch == 0
        self.grid = img_size // patch
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_ch, embed, patch, patch)

    def forward(self, x):
        x = self.proj(x)  # (B,D,H',W')
        return x.flatten(2).transpose(1, 2)  # (B,N,D)


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


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.d = dim // heads
        self.scale = self.d ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(-1)
        attn = self.drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


# class AdaLNZero(nn.Module):
#     def __init__(self, dim: int, cond_dim: int):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim, elementwise_affine=False)
#         self.to_params = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 3))
#         nn.init.zeros_(self.to_params[-1].weight)
#         nn.init.zeros_(self.to_params[-1].bias)

#     def forward(self, x, cond, fn):
#         scale, shift, gate = self.to_params(cond).chunk(3, dim=-1)
#         h = self.norm(x)
#         h = h * (1 + scale) + shift
#         h = fn(h)
#         return x + gate * h

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g

class AdaLNZero(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        #self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm = RMSNorm(dim)
        self.to_params = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 3))
        nn.init.zeros_(self.to_params[-1].weight)
        nn.init.zeros_(self.to_params[-1].bias)


    def forward(self, x, cond, fn):
        """
        x: (B, N, D)
        cond: (B, C) → 通过线性映射产生 (B, 3*D)
        注意：需要在 seq 维度做 broadcast，因此要在 dim=1 上 unsqueeze。
        """
        scale, shift, gate = self.to_params(cond).chunk(3, dim=-1) # (B, D) each
        # broadcast 到 (B, N, D)
        scale = scale[:, None, :]
        shift = shift[:, None, :]
        gate = gate[:, None, :]
        h = self.norm(x)
        h = h * (1 + scale) + shift
        h = fn(h)
        return x + gate * h

class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, cond_dim: int, drop: float = 0.0):
        super().__init__()
        self.attn = Attention(dim, heads)
        self.mlp = MLP(dim, mlp_ratio, drop)
        self.adaln1 = AdaLNZero(dim, cond_dim)
        self.adaln2 = AdaLNZero(dim, cond_dim)

    def forward(self, x, cond):
        x = self.adaln1(x, cond, self.attn)
        x = self.adaln2(x, cond, self.mlp)
        return x

# ------------------------
# 编码器：Watermark-DiT（单流联合注意）
# ------------------------



class WatermarkDiTEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        bit_len: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Patch embedding/unpatch
        self.patch = PatchEmbed(img_size, patch_size, in_channels, hidden)
        self.unpatch = Unpatchify(img_size, patch_size, in_channels, hidden)
        self.N_img = self.patch.num_patches

        # 位置与类型嵌入
        self.pos_img = nn.Parameter(build_sincos_2d(self.patch.grid, self.patch.grid, hidden))
        self.type_img = nn.Parameter(torch.zeros(1, 1, hidden))
        self.type_bit = nn.Parameter(torch.zeros(1, 1, hidden))

        # 条件嵌入
        self.t_embed = SinTimeEmbedding(hidden)
        self.bit_tok = BitTokenizer(bit_len, hidden, p_dropout_bits=0.1)
        self.cond_dim = hidden * 2  # concat(t_emb, strength_emb)
        self.strength_mlp = nn.Sequential(
            nn.Linear(1, hidden * 4),
            nn.SiLU(),
            nn.Linear(hidden * 4, hidden)
        )

        # Transformer 堆叠
        self.blocks = nn.ModuleList([
            DiTBlock(hidden, heads, mlp_ratio, self.cond_dim, dropout)
            for _ in range(depth)
        ])
        self.final_adaln = AdaLNZero(hidden, self.cond_dim)
        self.head = nn.Linear(hidden, hidden)

        # 输出融合：Conv2d 替代残差
        self.fuse = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.out_gain = nn.Parameter(torch.tensor(0.0))  # 仍可留作调节强度

    def _build_cond(self, t: torch.Tensor, strength: torch.Tensor):
        # t: (B,)  strength: (B,) in [0,1] 控制嵌入强度
        t_emb = self.t_embed(t)
        s_emb = self.strength_mlp(strength[:, None].float())
        return torch.cat([t_emb, s_emb], dim=-1)

    def forward(self, x: torch.Tensor, bits: torch.Tensor, t: torch.Tensor, strength: torch.Tensor):
        """
        x: (B,C,H,W) 原图
        bits: (B, bit_len) in {0,1}
        t: (B,) 时间步（若不使用扩散，可填随机数）
        strength: (B,) [0,1] 水印注入强度（训练可随机抖动）
        返回 x_tilde: (B,C,H,W)
        """
        B, C, H, W = x.shape
        assert H == self.cfg.img_size and W == self.cfg.img_size

        img_tok = self.patch(x) + self.pos_img[None, :, :] + self.type_img
        bit_tok = self.bit_tok(bits) + self.type_bit
        # 拼接（单流联合注意）
        tokens = torch.cat([img_tok, bit_tok], dim=1)  # (B, N_img+L_bits, D)

        cond = self._build_cond(t, strength)
        for blk in self.blocks:
            tokens = blk(tokens, cond)
        tokens = self.final_adaln(tokens, cond, self.head)

        # 仅取回图像部分并重构
        img_tokens = tokens[:, : self.N_img, :]
        delta = self.unpatch(img_tokens)  # (B,C,H,W)
        x_tilde = self.fuse(torch.cat([x, delta], dim=1))
        return x_tilde.clamp(-1, 1)

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
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = WatermarkDiTConfig(
        img_size=64, patch_size=2, in_channels=3,
        hidden=256, depth=8, heads=8, mlp_ratio=4.0,
        bit_len=64, dropout=0.0
    )
    enc = WatermarkDiTEncoder(cfg).to(device)
    dec = WatermarkDecoder(in_ch=3, hidden=64, bit_len=cfg.bit_len).to(device)
    atk = SimpleAttacks(sigma=0.01, blur_ksize=3).to(device)

    B = 4
    x = torch.randn(B, 3, cfg.img_size, cfg.img_size, device=device).clamp(-1, 1)
    bits = torch.randint(0, 2, (B, cfg.bit_len), device=device)
    t = torch.randint(0, 1000, (B,), device=device)
    strength = torch.rand(B, device=device) * 0.5 + 0.25  # 0.25~0.75

    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=1e-4, weight_decay=1e-2)

    # 前向
    x_tilde = enc(x, bits, t, strength)
    x_attacked = atk(x_tilde)
    logits = dec(x_attacked)

    # 损失
    recon_l1 = F.l1_loss(x_tilde, x)
    bit_bce = F.binary_cross_entropy_with_logits(logits, bits.float())
    loss = recon_l1 * 1.0 + bit_bce * 1.0

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
    opt.step()

    with torch.no_grad():
        acc = ((logits.sigmoid() > 0.5).long() == bits).float().mean()
    print({"loss": float(loss), "recon_l1": float(recon_l1), "bit_acc": float(acc)})


# ------------------------
# 工具：正弦位置编码（1D/2D）
# ------------------------


# ------------------------
# 时间步与水印 bit 嵌入
# ------------------------
class SinTimeEmbedding(nn.Module):
    def __init__(self, hidden: int, time_dim: Optional[int] = None):
        super().__init__()
        time_dim = time_dim or hidden
        self.time_dim = time_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden * 4), nn.SiLU(), nn.Linear(hidden * 4, hidden)
        )

    @staticmethod
    def sin_embed(t: torch.Tensor, dim: int):
        device = t.device
        half = dim // 2
        freqs = torch.exp(torch.linspace(0, math.log(10000), steps=half, device=device))
        ang = t[:, None] / freqs[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb

    def forward(self, t: torch.Tensor):
        if t.dim() == 0:
            t = t[None]
        s = self.sin_embed(t.float(), self.time_dim)
        return self.mlp(s)


class BitTokenizer(nn.Module):
    """把 {0,1} 序列转成 token 嵌入；可选 learnable [NULL] 占位（支持随机 bit dropout）。"""
    def __init__(self, bit_len: int, hidden: int, p_dropout_bits: float = 0.0):
        super().__init__()
        self.bit_len = bit_len
        self.emb0 = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.emb1 = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.null = nn.Parameter(torch.zeros(1, 1, hidden))
        self.p_dropout = p_dropout_bits
        self.pos = nn.Parameter(build_sincos_1d(bit_len, hidden))

    def forward(self, bits: torch.Tensor):
        """bits: (B, bit_len) in {0,1} (torch.long 或 float)
        返回: (B, bit_len, hidden)
        """
        B, L = bits.shape
        bits = bits.float()
        if self.training and self.p_dropout > 0:
            drop = (torch.rand(B, L, device=bits.device) < self.p_dropout).float()
            # 置空位使用 null token
            emb = bits.unsqueeze(-1) * self.emb1 + (1 - bits).unsqueeze(-1) * self.emb0
            emb = emb + self.pos[None, :, :]
            emb = emb * (1 - drop.unsqueeze(-1)) + self.null * drop.unsqueeze(-1)
        else:
            emb = bits.unsqueeze(-1) * self.emb1 + (1 - bits).unsqueeze(-1) * self.emb0
            emb = emb + self.pos[None, :, :]
        return emb

