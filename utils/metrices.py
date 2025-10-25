import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_fn

def compute_psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target, reduction='mean')
    return 20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))

def compute_ssim(pred, target, max_val=1.0):
    # 使用 pytorch_msssim 里的 SSIM
    return ssim_fn(pred, target, data_range=max_val, size_average=True)




import torch
from torchmetrics.classification import BinaryAUROC

def compute_iou(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    二分类 IoU (batch 内平均)
    preds: (N,1,H,W) 概率 [0,1]
    targets: (N,1,H,W) 0/1
    """
    preds = (preds >= threshold).float()
    targets = (targets > 0.5).float()

    inter = (preds * targets).sum(dim=(1,2,3))
    union = (preds + targets - preds * targets).sum(dim=(1,2,3))

    iou = (inter + 1e-7) / (union + 1e-7)
    return iou.mean()   # 当前 batch 平均


def compute_f1(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    二分类 F1 (batch 内平均)
    """
    preds = (preds >= threshold).float()
    targets = (targets > 0.5).float()

    tp = (preds * targets).sum(dim=(1,2,3))
    fp = (preds * (1 - targets)).sum(dim=(1,2,3))
    fn = ((1 - preds) * targets).sum(dim=(1,2,3))

    f1 = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
    return f1.mean()

from torchmetrics.functional import auroc
def compute_auc(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    二分类 AUC (batch 内计算)
    preds: (N,1,H,W) 概率 [0,1]
    targets: (N,1,H,W) 0/1
    """
    preds = preds.view(-1)
    targets = targets.view(-1).long()

    # torchmetrics functional 版本是无状态的
    return auroc(preds, targets, task="binary")

# def compute_auc(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#     """
#     二分类 AUC (batch 内计算)
#     preds: (N,1,H,W) 概率
#     targets: (N,1,H,W) 0/1
#     """
#     preds = preds.view(-1)
#     targets = targets.view(-1).long()

#     auroc_fn = BinaryAUROC().to(preds.device)
#     auc = auroc_fn(preds, targets)
#     return auc


