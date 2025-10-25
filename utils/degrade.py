import torch
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import RandomPerspective
import math



import torch
import torch.nn.functional as F
import random
import math

# ==================== 基础退化操作 ====================

def _ensure_batch(x: torch.Tensor) -> (torch.Tensor, bool):
    """确保输入是 (N,C,H,W)，返回 (tensor, 是否单图像输入)"""
    if x.ndim == 3:  # (C,H,W)
        x = x.unsqueeze(0)
        return x, True
    elif x.ndim == 4:  # (N,C,H,W)
        return x, False
    else:
        raise ValueError(f"输入必须是 (C,H,W) 或 (N,C,H,W)，得到 {x.shape}")

def jpeg_compression(x: torch.Tensor, quality: int = 60) -> torch.Tensor:
    """模拟 JPEG 压缩：下采样 + 上采样近似"""
    x, single = _ensure_batch(x)
    N,C,H,W = x.shape
    scale = max(1, int(quality)) / 100.0
    new_H, new_W = max(1, int(H * scale)), max(1, int(W * scale))
    x_down = F.interpolate(x, size=(new_H, new_W), mode="bilinear", align_corners=False)
    x_up = F.interpolate(x_down, size=(H, W), mode="bilinear", align_corners=False)
    return x_up.squeeze(0) if single else x_up

def gaussian_filter(x: torch.Tensor, ksize: int = 3, sigma: float = 1.0) -> torch.Tensor:
    """高斯模糊"""
    x, single = _ensure_batch(x)
    ax = torch.arange(-ksize // 2 + 1., ksize // 2 + 1., device=x.device)
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1,1,ksize,ksize).to(x.dtype)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)  # C 通道共享
    out = F.conv2d(x, kernel, padding=ksize//2, groups=x.shape[1])
    return out.squeeze(0) if single else out

def gaussian_noise(x: torch.Tensor, mean=0.0, std=0.05) -> torch.Tensor:
    x, single = _ensure_batch(x)
    noise = torch.randn_like(x) * std + mean
    out = (x + noise).clamp(0,1)
    return out.squeeze(0) if single else out

def median_filter(x: torch.Tensor, ksize: int = 3) -> torch.Tensor:
    """中值滤波"""
    x, single = _ensure_batch(x)
    N,C,H,W = x.shape
    x_unf = F.unfold(x, kernel_size=ksize, padding=ksize//2)  # (N,C*ksize*ksize,L)
    x_unf = x_unf.view(N,C,ksize*ksize,-1)
    median, _ = x_unf.median(dim=2)
    out = median.view(N,C,H,W)
    return out.squeeze(0) if single else out

def salt_pepper_noise(x: torch.Tensor, ratio=0.05) -> torch.Tensor:
    x, single = _ensure_batch(x)
    mask = torch.rand_like(x)
    x = x.clone()
    x[mask < ratio/2] = 0.0
    x[mask > 1-ratio/2] = 1.0
    return x.squeeze(0) if single else x

def resize(x: torch.Tensor, scale=0.5) -> torch.Tensor:
    x, single = _ensure_batch(x)
    N,C,H,W = x.shape
    new_H, new_W = int(H*scale), int(W*scale)
    out = F.interpolate(x, size=(new_H,new_W), mode="bilinear", align_corners=False)
    return out.squeeze(0) if single else out

def adjust_brightness(x: torch.Tensor, factor: float) -> torch.Tensor:
    x, single = _ensure_batch(x)
    out = (x * factor).clamp(0,1)
    return out.squeeze(0) if single else out

def adjust_contrast(x: torch.Tensor, factor: float) -> torch.Tensor:
    x, single = _ensure_batch(x)
    mean = x.mean(dim=(2,3), keepdim=True)  # batch 版本
    out = ((x - mean) * factor + mean).clamp(0,1)
    return out.squeeze(0) if single else out

def adjust_saturation(x: torch.Tensor, factor: float) -> torch.Tensor:
    x, single = _ensure_batch(x)
    gray = x.mean(dim=1, keepdim=True)
    out = ((x - gray) * factor + gray).clamp(0,1)
    return out.squeeze(0) if single else out

def adjust_hue(x: torch.Tensor, factor: float) -> torch.Tensor:
    """色相调整，基于 YIQ 空间"""
    x, single = _ensure_batch(x)
    N,C,H,W = x.shape
    flat = x.view(N,C,-1)  # (N,3,H*W)

    u = math.cos(factor * math.pi)
    w = math.sin(factor * math.pi)
    M = torch.tensor([[0.299, 0.587, 0.114],
                      [0.596, -0.274, -0.321],
                      [0.211, -0.523, 0.312]], device=x.device, dtype=x.dtype)
    M_inv = torch.tensor([[1.0, 0.956, 0.621],
                          [1.0, -0.272, -0.647],
                          [1.0, -1.106, 1.703]], device=x.device, dtype=x.dtype)

    # 颜色空间变换
    y = torch.einsum('ij,nbj->nbi', M, flat)  # (N,3,H*W)

    # hue 旋转
    y1, y2 = y[:,1,:], y[:,2,:]  # (N,H*W)
    y[:,1,:] = u * y1 - w * y2
    y[:,2,:] = w * y1 + u * y2

    # 变换回 RGB
    out = torch.einsum('ij,nbj->nbi', M_inv, y).view(N,C,H,W)
    out = out.clamp(0,1)
    return out.squeeze(0) if single else out

# ==================== 随机退化入口 ====================

def apply_random_degradations(x: torch.Tensor, input_range="[-1,1]") -> torch.Tensor:
    """
    对图像或图像批次应用一个随机退化
    输入: (C,H,W) 或 (N,C,H,W)
    """
    if input_range == "[-1,1]":
        x = (x + 1) / 2  # -> [0,1]

    op_dict = {
        "clean": lambda x: x,
        "jpeg": lambda x: jpeg_compression(x, quality=50),
        "gaussian_filter": lambda x: gaussian_filter(x, ksize=3, sigma=1.5),
        "gaussian_noise": lambda x: gaussian_noise(x, mean=0, std=0.1),
        # "median": lambda x: median_filter(x, ksize=3),
        # "salt_pepper": lambda x: salt_pepper_noise(x, ratio=0.05),
        # "resize": lambda x: resize(x, scale=0.5),
        "brightness": lambda x: adjust_brightness(x, random.uniform(0.75, 1.25)),
        "contrast": lambda x: adjust_contrast(x, random.uniform(0.75, 1.25)),
        # "hue": lambda x: adjust_hue(x, random.uniform(-0.1, 0.1)),
        # "saturation": lambda x: adjust_saturation(x, random.uniform(0.75, 1.25)),
    }

    name = random.choice(list(op_dict.keys()))
    x = op_dict[name](x)

    if input_range == "[-1,1]":
        x = x * 2 - 1

    return x


def apply_random_degradations_no_clean(x: torch.Tensor, input_range="[-1,1]") -> torch.Tensor:
    """
    对图像或图像批次应用一个随机退化
    输入: (C,H,W) 或 (N,C,H,W)
    """
    if input_range == "[-1,1]":
        x = (x + 1) / 2  # -> [0,1]

    op_dict = {
        "jpeg": lambda x: jpeg_compression(x, quality=50),
        "gaussian_filter": lambda x: gaussian_filter(x, ksize=3, sigma=1.5),
        "gaussian_noise": lambda x: gaussian_noise(x, mean=0, std=0.1),
        # "median": lambda x: median_filter(x, ksize=3),
        # "salt_pepper": lambda x: salt_pepper_noise(x, ratio=0.05),
        # "resize": lambda x: resize(x, scale=0.5),
        "brightness": lambda x: adjust_brightness(x, random.uniform(0.75, 1.25)),
        "contrast": lambda x: adjust_contrast(x, random.uniform(0.75, 1.25)),
        # "hue": lambda x: adjust_hue(x, random.uniform(-0.1, 0.1)),
        # "saturation": lambda x: adjust_saturation(x, random.uniform(0.75, 1.25)),
    }

    name = random.choice(list(op_dict.keys()))
    x = op_dict[name](x)

    if input_range == "[-1,1]":
        x = x * 2 - 1

    return x



# def apply_random_degradations(x: torch.Tensor, input_range="[-1,1]") -> torch.Tensor:
#     """
#     对图像批次应用同一个随机退化
#     输入: (N,C,H,W) 或 (C,H,W)
#     输出: (N,C,H,W) 或 (C,H,W)
#     """
#     single_input = False
#     if x.ndim == 3:  # (C,H,W)
#         x = x.unsqueeze(0)  # 加 batch 维
#         single_input = True

#     if input_range == "[-1,1]":
#         x = (x + 1) / 2  # -> [0,1]

#     op_dict = {
#         "jpeg": lambda x: jpeg_compression(x, quality=50),
#         "gaussian_filter": lambda x: gaussian_filter(x, ksize=3, sigma=1.5),
#         "gaussian_noise": lambda x: gaussian_noise(x, mean=0, std=0.1),
#         "resize": lambda x: F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False),
#         "brightness": lambda x: adjust_brightness(x, random.uniform(0.75, 1.25)),
#         "contrast": lambda x: adjust_contrast(x, random.uniform(0.75, 1.25)),
#         "hue": lambda x: adjust_hue(x, random.uniform(-0.1, 0.1)),
#         "saturation": lambda x: adjust_saturation(x, random.uniform(0.75, 1.25)),
#     }

#     # 随机选择一个操作
#     name = random.choice(list(op_dict.keys()))
#     x = op_dict[name](x)  # 直接对整个 batch 操作

#     if input_range == "[-1,1]":
#         x = x * 2 - 1

#     if single_input:
#         x = x.squeeze(0)

#     return x



# a = torch.randn(1,3,512,512)
# b = apply_random_degradations(a)


def apply_specific_degradations(
    x: torch.Tensor,
    degradations: list,
    input_range="[-1,1]"
) -> torch.Tensor:
    """
    对图像应用指定的退化操作（依次执行）
    支持输入 (C,H,W) 或 (N,C,H,W)，其中 N 可以是 1
    """

    is_batched = False
    if x.ndim == 4:  # (N,C,H,W)
        assert x.shape[0] == 1, "当前只支持 batch=1，可以自己写循环扩展"
        x = x.squeeze(0)  # -> (C,H,W)
        is_batched = True
    elif x.ndim == 3:  # (C,H,W)
        pass
    else:
        raise ValueError(f"输入 shape {x.shape} 不支持，期望 (C,H,W) 或 (N,C,H,W)")

    # ---- Step 1: 归一化到 [0,1] ----
    if input_range == "[-1,1]":
        x = (x + 1) / 2  # [-1,1] -> [0,1]

    # ---- Step 2: 定义操作映射 ----
    op_dict = {
        "jpeg": lambda x: jpeg_compression(x, quality=60),
        "gaussian_filter": lambda x: gaussian_filter(x, ksize=3, sigma=1.0),
        "gaussian_noise": lambda x: gaussian_noise(x, mean=0, std=0.05),
        "median_filter": lambda x: median_filter(x, ksize=3),
        "salt_pepper": lambda x: salt_pepper_noise(x, ratio=0.05),
        "resize": lambda x: resize(x, scale=0.5),
        "brightness": lambda x: adjust_brightness(x, random.uniform(0.7, 1.3)),
        "contrast": lambda x: adjust_contrast(x, random.uniform(0.7, 1.3)),
        "hue": lambda x: adjust_hue(x, random.uniform(-0.1, 0.1)),
        "saturation": lambda x: adjust_saturation(x, random.uniform(0.7, 1.3)),
    }

    # ---- Step 3: 执行指定退化 ----
    for name in degradations:
        if name not in op_dict:
            raise ValueError(f"Unknown degradation: {name}")
        x = op_dict[name](x)

    # ---- Step 4: 转回 [-1,1] ----
    if input_range == "[-1,1]":
        x = x * 2 - 1

    # ---- Step 5: 如果原始是 batch 输入，加回 batch 维度 ----
    if is_batched:
        x = x.unsqueeze(0)

    return x






def apply_geometric_degradations(
    x: torch.Tensor,
    degradations: list,
    input_range="[-1,1]",
    with_temp = True,
) -> torch.Tensor:
    """
    对图像应用指定的几何退化操作（依次执行）

    参数:
        x: torch.Tensor, shape (C,H,W) 或 (N,C,H,W)，其中 N 可以是 1
        degradations: list[str], 例如 ["rotation", "perspective", "hflip"]
        input_range: '[-1,1]' 或 '[0,1]'
    """

    is_batched = False
    if x.ndim == 4:  # (N,C,H,W)
        assert x.shape[0] == 1, "当前版本只支持 batch=1，可以写循环扩展"
        x = x.squeeze(0)  # -> (C,H,W)
        is_batched = True
    elif x.ndim == 3:  # (C,H,W)
        pass
    else:
        raise ValueError(f"Unsupported shape {x.shape}, expect (C,H,W) or (N,C,H,W)")

    # ---- Step 1: 归一化到 [0,1] ----
    if input_range == "[-1,1]":
        x = (x + 1) / 2

    angle_temp = random.uniform(-30, 30)
    distortion_scale_temp = random.uniform(0.1, 0.3)

    startpoints, endpoints = RandomPerspective.get_params(512, 512, distortion_scale_temp)
    # ---- Step 2: 定义操作 ----
    op_dict = {
        "rotation": lambda x: TF.rotate(
            x, angle = angle_temp, expand=False,
            interpolation=InterpolationMode.BILINEAR
        ),
        "perspective": lambda x: TF.perspective(
            x, startpoints, endpoints,
            interpolation=InterpolationMode.BILINEAR, fill=0
        ),
        "hflip": lambda x: TF.hflip(x),
    }

    # ---- Step 3: 执行 ----
    for name in degradations:
        if name not in op_dict:
            raise ValueError(f"Unknown geometric degradation: {name}")
        x = op_dict[name](x)
    
    if name == "rotation":
        temp = angle_temp
    elif name == "perspective":
        temp = [startpoints, endpoints]
    else:
        temp = None
        

    # ---- Step 4: 转回 [-1,1] ----
    if input_range == "[-1,1]":
        x = x * 2 - 1

    # ---- Step 5: 如果原始是 batch 输入，加回 batch 维 ----
    if is_batched:
        x = x.unsqueeze(0)
    if with_temp:
        return x, temp
    else:
        return x


def apply_geometric_degradations_mask(
    x: torch.Tensor,
    degradations: list,
    temp,
    fill_value = 1
) -> torch.Tensor:
    """
    对 0/1 mask 应用几何退化（保持二值）

    支持输入:
        (H,W), (1,H,W), (C,H,W), (N,C,H,W)
    """
    is_batched = False

    # ---- Step 1: 调整维度到 (C,H,W) ----
    if x.ndim == 2:          # (H,W)
        x = x.unsqueeze(0)   # -> (1,H,W)
    elif x.ndim == 3:        # (C,H,W) OK
        pass
    elif x.ndim == 4:        # (N,C,H,W)
        assert x.shape[0] == 1, "只支持 batch=1，可以自己写循环处理 batch"
        x = x.squeeze(0)     # -> (C,H,W)
        is_batched = True
    else:
        raise ValueError(f"Unsupported shape {x.shape}, expect (H,W), (1,H,W), (C,H,W) or (N,C,H,W)")

    
    if temp is None:
        angle_temp = random.uniform(-30, 30)
        distortion_scale_temp = random.uniform(0.1, 0.3)
        startpoints, endpoints = RandomPerspective.get_params(512, 512, distortion_scale_temp)
    else:
        if isinstance (temp,list):
           startpoints, endpoints = temp[0],temp[1]
        else:
            angle_temp = temp


    # ---- Step 2: 定义操作 ----
    op_dict = {
        "rotation": lambda x: TF.rotate(
            x, angle=angle_temp, expand=False,
            interpolation=InterpolationMode.NEAREST, fill=fill_value
        ),
        "perspective": lambda x: TF.perspective(
            x, startpoints, endpoints,
            interpolation=InterpolationMode.NEAREST, fill=1
        ),
        "hflip": lambda x: TF.hflip(x),
    }

    # ---- Step 3: 应用变换 ----
    for name in degradations:
        if name not in op_dict:
            raise ValueError(f"Unknown geometric degradation: {name}")
        x =  op_dict[name](x)

    # ---- Step 4: 保证输出还是 0/1 ----
    x = (x > 0.5).float()

    # ---- Step 5: 还原回 (N,C,H,W) ----
    if is_batched:
        x = x.unsqueeze(0)  # 加回 batch 维

    return x