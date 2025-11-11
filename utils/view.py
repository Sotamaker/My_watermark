import torch
from torchvision.utils import save_image
import os

def save_val_grid(
    imgs,                # 原图 (B,3,H,W)
    imgs_wm,             # 带水印图 (B,3,H,W)
    imgs_diff,           # 差异图 (B,3,H,W)
    mask,                # 原mask (B,1,H,W)
    pred_mask,           # 预测mask (B,1,H,W)
    pred_mask_noise,     # 噪声mask (B,1,H,W)
    save_path,
    normalize=True,
):
    """
    将多个 batch 拼接为 6 行 (类型)，每列为一个样本。
    最后三行 (mask相关) 以黑白形式保存。
    """
    # ---- Step 1: 转为0~1范围，避免保存异常 ----
    imgs_01      = imgs.clamp(0, 1)
    imgs_wm_01   = imgs_wm.clamp(0, 1)
    imgs_diff_01 = (imgs_diff - imgs_diff.min()) / (imgs_diff.max() - imgs_diff.min() + 1e-8)

    # ---- Step 2: 将 mask 全部二值化并重复为3通道 ----
    mask_bw          = (mask > 0.5).float().repeat(1, 3, 1, 1)
    pred_mask_bw     = (pred_mask > 0.5).float().repeat(1, 3, 1, 1)
    pred_mask_noise_bw = (pred_mask_noise > 0.5).float().repeat(1, 3, 1, 1)

    # ---- Step 3: 构造每一行 ----
    row1 = imgs_01
    row2 = imgs_wm_01
    row3 = imgs_diff_01
    row4 = mask_bw
    row5 = pred_mask_bw
    row6 = pred_mask_noise_bw

    # ---- Step 4: 拼接所有行 (行方向concat) ----
    grid = torch.cat([row1, row2, row3, row4, row5, row6], dim=0)

    # ---- Step 5: 保存 ----
    save_image(
        grid,
        save_path,
        nrow=imgs.size(0),          # 每行显示多少列（=样本数量）
        normalize=normalize,
        scale_each=True,
    )

    #print(f"✅ 保存验证可视化图: {save_path}")


# from torchvision.utils import save_image
# import torch
# import os
# from tqdm import tqdm

# # ----------------- 辅助函数 -----------------
# def save_val_grid(
#     imgs, imgs_wm, imgs_diff,
#     mask, pred_mask, pred_mask_noise,
#     save_path, normalize=True
# ):
#     """按行类型、按列样本保存"""
#     imgs_01      = imgs.clamp(0, 1)
#     imgs_wm_01   = imgs_wm.clamp(0, 1)
#     imgs_diff_01 = (imgs_diff - imgs_diff.min()) / (imgs_diff.max() - imgs_diff.min() + 1e-8)

#     mask_bw          = (mask > 0.5).float().repeat(1, 3, 1, 1)
#     pred_mask_bw     = (pred_mask > 0.5).float().repeat(1, 3, 1, 1)
#     pred_mask_noise_bw = (pred_mask_noise > 0.5).float().repeat(1, 3, 1, 1)

#     grid = torch.cat([imgs_01, imgs_wm_01, imgs_diff_01,
#                       mask_bw, pred_mask_bw, pred_mask_noise_bw], dim=0)
    
#     save_image(grid, save_path, nrow=imgs.size(0),
#                normalize=normalize, scale_each=True)
#     print(f"✅ 保存验证可视化图: {save_path}")