import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_1d_sincos_pos_embed(embed_dim, positions):
    """
    Args:
        embed_dim: int, 每个位置的编码维度 (必须为偶数)
        positions: (M,) 一维位置数组

    Returns:
        (M, embed_dim) 的正余弦编码
    """
    assert embed_dim % 2 == 0, "embed_dim 必须是偶数"

    # 频率：从 1 到 1/10000^(d/D)
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega = 1.0 / (10000 ** (omega / (embed_dim // 2)))

    # 外积：M × (D/2)
    np.einsum("m,d->md", positions, omega)
    angles = np.outer(positions, omega)

    # 拼接 sin 和 cos
    emb = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, base_size=16, scale=1.0, extra_tokens=0):
    """
    Args:
        embed_dim: int, 每个位置的编码维度
        grid_size: (H, W) 或 int，网格大小（patch 数量）
        base_size: int，基准 patch 大小，用于归一化
        scale: float，缩放因子（插值用）
        extra_tokens: int，前面额外的 token 数（如 cls_token）

    Returns:
        (H*W+extra_tokens, embed_dim)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)
    H, W = grid_size

    # 归一化的坐标
    grid_y = np.arange(H, dtype=np.float32) * (base_size / H) / scale
    grid_x = np.arange(W, dtype=np.float32) * (base_size / W) / scale

    # 构造网格
    grid = np.meshgrid(grid_x, grid_y)  # (x,y)
    grid = np.stack(grid, axis=0)       # (2, H, W)

    # 展平成一维
    grid = grid.reshape(2, -1)          # (2, H*W)

    # 分别对 x, y 方向做 1D sincos
    emb_x = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_y = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])

    # 拼接 (H*W, D)
    pos_embed = np.concatenate([emb_x, emb_y], axis=1)

    # 如果需要额外 token（比如 cls_token）
    if extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros((extra_tokens, embed_dim), dtype=np.float32), pos_embed], axis=0)

    return pos_embed



class PatchEmbed(nn.Module):
    """
    把图像切成 patch, 并加上 2D 正余弦位置编码
    """

    def __init__(self, width=512, height=512, patch_size=16, in_chans=3,
                 embed_dim=768, pos_embed_type="sincos", pos_embed_max_size=None, scale=1.0, extra_tokens=0):
        super().__init__()

        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.base_size = height // patch_size
        self.num_patches = (height // patch_size) * (width // patch_size)
        self.pos_embed_max_size = pos_embed_max_size

        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        else:
            grid_size = int(self.num_patches**0.5)
        
        # 用 Conv2d 做 patch 切分 (stride=patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)

        # 初始化最大 pos_embed
        if pos_embed_type == "sincos":
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim, grid_size, base_size=self.base_size, scale=scale, extra_tokens=extra_tokens
            )
            persistent = True if pos_embed_max_size else False
            self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=persistent)
        else:
            self.pos_embed = None
        
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)


    def cropped_pos_embed(self, height, width):
        """从最大表里裁剪出合适大小的 pos_embed"""
        h, w = height // self.patch_size, width // self.patch_size
        if h > self.pos_embed_max_size or w > self.pos_embed_max_size:
            raise ValueError(f"输入 ({h},{w}) 大于最大支持 {self.pos_embed_max_size}")

        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2

        pos = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        pos = pos[:, top: top + h, left: left + w, :]  # 裁剪
        return pos.reshape(1, -1, pos.shape[-1])

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 图像

        Returns:
            (B, N, D) patch embeddings (加上位置编码)
        """
        B, C, height, width = x.shape
        x = self.proj(x)                # (B, D, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)  # 裁剪
        else:     
            if self.height != height or self.width != width:
                h, w = height // self.patch_size, width // self.patch_size
                pos_embed = get_2d_sincos_pos_embed(x.shape[-1], (h, w), base_size=self.base_size)
                pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(x.device)
            else:
                pos_embed = self.pos_embed
        x = self.norm(x)
        x = x + pos_embed
        return x



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

class CondDiTBlock(nn.Module):
    def __init__(self, dim, cond_dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = Attention(dim, heads, attn_drop=dropout, proj_drop=dropout)
        self.mlp  = MLP(dim, ratio=mlp_ratio, drop=dropout)

        # gate & FiLM
        self.to_gate1 = nn.Linear(cond_dim, dim)
        self.to_mod2  = nn.Linear(cond_dim, dim * 3)

        nn.init.zeros_(self.to_gate1.weight)
        nn.init.zeros_(self.to_gate1.bias)
        nn.init.zeros_(self.to_mod2.weight)
        nn.init.zeros_(self.to_mod2.bias)

    def forward(self, x, cond=None):
        # Attention path
        h = self.norm1(x)
        attn_out = self.attn(h)
        if cond is not None:
            gate1 = self.to_gate1(cond)[:, None, :].tanh()
            x = x + gate1 * attn_out
        else:
            x = x + attn_out

        
        # MLP path
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


class UnpatchDecoder(nn.Module):
    def __init__(self, img_size, patch_size, in_dim, out_ch):
        super().__init__()
        self.grid = img_size // patch_size
        self.patch_size = patch_size

        # MOST STABLE: Linear unpatchify
        self.unpatch = nn.Linear(in_dim, patch_size * patch_size * out_ch)

        # Strong smoothing (UNet-like)
        self.smooth = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )

        self.out_gain = nn.Parameter(torch.zeros(1))  # zero-init residual

    def forward(self, tokens, img):
        B, N, D = tokens.shape
        g = int(math.sqrt(N))
        p = self.patch_size

        # Linear → patch reshape
        patch = self.unpatch(tokens)  # (B, N, p*p*C)
        patch = patch.reshape(B, g, g, p, p, -1)
        delta = patch.permute(0, 5, 1, 3, 2, 4).reshape(
            B, -1, g * p, g * p
        )

        # smoothing (UNet style)
        delta = self.smooth(delta)

        # stable residual
        x_tilde = img + self.out_gain * delta
        return x_tilde, delta

class WatermarkDiTEncoder1(nn.Module):
    def __init__(
        self,
        img_size: int = 512,
        width: int = 512,
        height: int = 512,
        patch_size: int = 16,
        num_patches: int = 1024,
        in_channels: int = 3,
        sec_p_dim: int = 0,
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
        self.patchemb = PatchEmbed(width=width, height=height, patch_size=patch_size, in_chans=in_channels, 
                 embed_dim=hidden, pos_embed_type=pos_embed_type, pos_embed_max_size=pos_embed_max_size, scale=scale)
        #self.unpatch = Unpatchify(img_size, patch_size, in_channels, hidden)
        self.N_img = num_patches

        # 条件嵌入
        self.patch_gate = nn.Parameter(torch.zeros(1))

        self.cond_dim = hidden
        self.sec_mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.SiLU(),
            nn.Linear(hidden * 4, hidden)
        )

        # Transformer 堆叠
        self.blocks = nn.ModuleList([
            CondDiTBlock(hidden, self.cond_dim, heads, mlp_ratio,  dropout)
            for _ in range(depth)
        ])
        self.final_norm = RMSNorm(hidden)
        self.final_proj = nn.Linear(hidden, hidden)

        # 输出融合：Conv2d 替代残差
        self.fuse = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.out_gain = nn.Parameter(torch.tensor(0.0))  # 仍可留作调节强度
        self.use_sec_tok = False
        self.decoder = UnpatchDecoder(img_size,patch_size,hidden,out_ch=in_channels)



        self.patch_cap_net = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)   # -> (B,N,1)
        )

        # ⭐ bit → 方向向量
        self.patch_dir_proj = nn.Linear(hidden, hidden)

    def _build_cond(self, sec_emb: torch.Tensor):
        # t: (B,)  strength: (B,) in [0,1] 控制嵌入强度
        
        sec_emb = F.layer_norm(sec_emb, (sec_emb.shape[-1],))
        #sec_emb = 0.1 * sec_emb

        #sec_emb = self.sec_mlp(sec_emb.float())
        return sec_emb

    def _build_patch_mod(self, img_tok, cond_emb):
        """
        img_tok: (B,N,D)
        cond_emb: (B,D)
        return modulation: (B,N,D)
        """

        # 1. 图像 patch 自己决定强度（capacity）
        patch_cap = torch.sigmoid(self.patch_cap_net(img_tok))  # (B,N,1)

        # 2. bits 决定方向（dir）
        bit_dir = self.patch_dir_proj(cond_emb)                 # (B,D)
        bit_dir = bit_dir / (bit_dir.norm(dim=-1, keepdim=True) + 1e-6)
        bit_dir = bit_dir[:, None, :]                           # (B,1,D)

        # 3. 最终 modulation
        sec_pos_emb = patch_cap * bit_dir                       # (B,N,D)
        return sec_pos_emb


    def forward(self, img: torch.Tensor, sec_pos_emb: torch.Tensor, sec_cond_emb: torch.Tensor, sec_tok_emb: torch.Tensor):
        """
        x: (B,C,H,W) 原图
        bits: (B, bit_len) in {0,1}
        t: (B,) 时间步（若不使用扩散，可填随机数）
        strength: (B,) [0,1] 水印注入强度（训练可随机抖动）
        返回 x_tilde: (B,C,H,W)
        """
        B, C, H, W = img.shape

        img_tok = self.patchemb(img)  #, sec_pix_emb

        img_tok = self._build_patch_mod(img_tok,sec_pos_emb)

        # 拼接（单流联合注意）
        if self.use_sec_tok:
            tokens = torch.cat([img_tok, sec_tok_emb], dim=1)  # (B, N_img+L_bits, D)
        else:
            tokens = img_tok

        cond = self._build_cond(sec_cond_emb)
        for blk in self.blocks:
            tokens = blk(tokens, cond)
        # 仅取回图像部分并重构
        img_tokens = tokens[:, : self.N_img, :]


        tokens = self.final_norm(img_tokens)
        x_tilde, delta = self.decoder(tokens, img)
        return x_tilde #, delta