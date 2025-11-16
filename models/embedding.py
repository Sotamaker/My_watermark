import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



##############################
#  pos embedding
##############################



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





    

##############################
#  sec embedding
##############################

    

class SecEmbed(nn.Module):
    def __init__(self, nbit=64, embed_dim=768, sec_p_dim=3, bit_emb_type="mlp"):
        super().__init__()
        self.embed_dim = embed_dim
        self.sec_p_dim = sec_p_dim
        self.sec_linear = nn.Sequential(
                nn.Linear(nbit, nbit * 16),
                nn.ReLU(),
                nn.Linear(nbit * 16, nbit * 16),
                nn.ReLU(),
                nn.Linear(nbit * 16, embed_dim)
            )
        
        #self.sec_pix_embedding =  torch.nn.Embedding(2, sec_p_dim)
        self.sec_tok_embedding =  torch.nn.Embedding(2*nbit, embed_dim)
        self.sec_tok_all_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, sec):
        # sec_pix_emb = self.sec_pix_embedding(sec)
        # sec_pix_emb = sec_pix_emb.reshape(-1, 8, 8, self.sec_p_dim)
        # sec_pix_emb = sec_pix_emb.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

        indices = 2 * torch.arange(sec.shape[-1]).to(sec.device)  # k: 0 2 4 ... 2k
        indices = indices.repeat(sec.shape[0], 1)  # b k
        sec_w_ind = (indices + sec).long()
        sec_tok_emb = self.sec_tok_embedding(sec_w_ind)
        sec_pos_emb = self.sec_tok_all_linear(sec_tok_emb.mean(dim=-2))


        sec = 2 * (sec - 0.5)
        sec_cond_emb = self.sec_linear(sec)
        return sec_pos_emb, sec_cond_emb, sec_tok_emb 
        
    # def get_bit_tok_emb(self, sec):
    #     indices = 2 * torch.arange(sec.shape[-1]).to(sec.device)  # k: 0 2 4 ... 2k
    #     indices = indices.repeat(sec.shape[0], 1)  # b k
    #     sec_w_ind = (indices + sec).long()
    #     sec_tok_emb = self.sec_embedding(sec_w_ind)
    #     return sec_tok_emb, sec_tok_emb.sum(dim=-2)

class PatchWithSecEmbed(nn.Module):
    
    def __init__(self, width=512, height=512, patch_size=16, in_chans=3, sec_p_dim=0,
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
        self.proj = nn.Conv2d(in_chans + sec_p_dim, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)

        #self.sec_emb = SecEmbed(nbit,embed_dim=embed_dim,sec_p_dim=sec_p_dim)

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

        
    def forward(self, x, sec_all_emb):
        """
        Args:
            x: (B, C, H, W) 图像

        Returns:
            (B, N, D) patch embeddings (加上位置编码)
        """
        B, C, height, width = x.shape
        h, w = height // self.patch_size, width // self.patch_size

        # sec_all_emb, sec_pix_emb = self.sec_emb(sec)
        # sec_pix_emb = sec_pix_emb.repeat(1, h, w, 1)
        
        x = self.proj(x)                # (B, D, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)  # 裁剪
        else:     
            if self.height != height or self.width != width:
                
                pos_embed = get_2d_sincos_pos_embed(x.shape[-1], (h, w), base_size=self.base_size)
                pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(x.device)
            else:
                pos_embed = self.pos_embed
        x = self.norm(x)
        x = x + pos_embed + sec_all_emb[:, None, :]
        return x
    



# patchembed = PatchWithSecEmbed()
# x = torch.randn(1,3,512,512)
# sec = torch.randint(0, 2, (1, 64))
# f = patchembed(x,sec)
# patchembed = PatchEmbed(pos_embed_max_size=48)

# 

# b = patchembed(x)
# print('end')

# Secemb = SecEmbed()

# 

# c,d = Secemb.get_sec_emb(sec)





import torch
import torch.nn as nn
import torch.nn.functional as F


class SecEmbeder(nn.Module):
    """
    Three modes:
      - "simple_mlp":        bits → MLP → LN → scaled cond
      - "positional_mlp":    bits → (bit_emb + pos_emb) seq → mean → MLP → LN
      - "lookup_2nbit":      bits → Embedding(2*nbit) → mean → Linear → LN  (noisy, for ablation)
    
    返回两个值:
      cond_emb: (B, D)   —— 给 DiT block 做 FiLM 调制用
      token_emb: (B, L, D)  —— 若你需要额外把 bit token 拼到 patch token（可选，用不到给 None）
    """

    def __init__(self, nbit=64,
                 embed_dim=768,
                 mode="simple_mlp",
                 return_patch_feat: bool = False,
                 ):
        super().__init__()
        self.nbit = nbit
        self.embed_dim = embed_dim
        self.mode = mode
        self.return_patch_feat = return_patch_feat

        
        
        # -------------------------
        # Mode 1: bits → simple MLP
        # -------------------------
        
        if mode == "simple_mlp":
            self.mlp = nn.Sequential(
            nn.Linear(nbit, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        # --------------------------------------
        # Mode 2: bits → (bit_emb + pos_emb) → MLP
        # --------------------------------------
        elif mode == "positional_mlp":
            self.bit_emb = nn.Embedding(2, embed_dim)      # bit=0 or 1
            self.pos_emb = nn.Embedding(nbit, embed_dim)   # position 0..L-1

            self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            )

        # --------------------------------------
        # Mode 3: lookup table Embedding(2*nbit)
        # --------------------------------------
        elif mode == "lookup_2nbit":
            self.lookup_emb = nn.Embedding(2 * nbit, embed_dim)  # 每个bit位置×取值=一个embedding
            self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        
        if return_patch_feat:
            # patch_feat 也用一个小 MLP refine 一下
            self.patch_refine = nn.Sequential(
                nn.Linear(nbit, embed_dim * 4),
                nn.SiLU(),
                nn.Linear(embed_dim * 4, embed_dim),
            )


    def _return(self, cond: torch.Tensor, token_emb: torch.Tensor, bits):
        if self.return_patch_feat:
            bits_f = 2 * bits.float() - 1.0
            patch_feat = self.patch_refine(bits_f)            # (B,D)
        else:
            patch_feat = None
        return patch_feat, cond, token_emb


    def forward(self, bits):
        """
        bits: (B, L) in {0,1}
        返回:
            cond_emb: (B, D)
            token_emb: (B, L, D)  or  None
        """

        B, L = bits.shape
        token_emb = None

        # ====================================================
        #                  Mode 1: simple MLP
        # ====================================================
        if self.mode == "simple_mlp":
            bits_f = 2 * bits.float() - 1.0
            cond = self.mlp(bits_f)               # (B,D)
            return self._return(cond, token_emb, bits)

        # ====================================================
        #         Mode 2: position-aware embedding → MLP
        # ====================================================
        elif self.mode == "positional_mlp":
            # bit token embedding
            bit_token = self.bit_emb(bits)              # (B,L,D)

            # position embedding
            positions = torch.arange(L, device=bits.device)
            pos_token = self.pos_emb(positions)         # (L,D)
            pos_token = pos_token.unsqueeze(0).expand(B, L, -1)  # (B,L,D)

            seq = bit_token + pos_token                 # (B,L,D)

            pooled = seq.mean(dim=1)                    # (B,D)
            cond = self.mlp(pooled)
            return self._return(cond, token_emb, bits)# 若需要把 bit token 拼进去，可用 seq

        # ====================================================
        #                Mode 3: lookup 2*nbit
        # ====================================================
        elif self.mode == "lookup_2nbit":
            # index trick: idx = 2*i + bit
            idx_base = 2 * torch.arange(L, device=bits.device)   # (L,)
            idx = idx_base.unsqueeze(0) + bits                   # (B,L)
            idx = idx.long()

            seq = self.lookup_emb(idx)                           # (B,L,D)
            pooled = seq.mean(dim=1)                             # (B,D)
            cond = self.mlp(pooled)
            return self._return(cond, token_emb, bits)

# Secember = SecEmbeder()



# sec = torch.rand(2,64)

# cond, token_emb, patch_feat = Secember(sec)