import torch
import torch.nn as nn
import torch.nn.functional as F


def WbceLoss(pred, target, pos_weight=2.0, neg_weight=0.5):
    #pred = torch.sigmoid(pred)
    loss = - (pos_weight * target * torch.log(pred + 1e-4) +
              neg_weight * (1 - target) * torch.log(1 - pred + 1e-4))
    return loss.mean()




class EdgeLoss(nn.Module):
    def __init__(self, loss_type='L1'):
        super(EdgeLoss, self).__init__()
        if loss_type == 'L1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'L2':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError("loss_type must be 'L1' or 'L2'")

        # Sobel filters
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def sobel_filter(self, img):
        grad_x = F.conv2d(img, self.sobel_x, padding=1)
        grad_y = F.conv2d(img, self.sobel_y, padding=1)
        edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return edge

    def forward(self, pred, target):
        pred_edge = self.sobel_filter(pred)
        target_edge = self.sobel_filter(target)
        loss = self.loss_fn(pred_edge, target_edge)
        return loss


class LossWeight(nn.Module):
    def __init__(self, loss_config):
        super(LossWeight, self).__init__()
        self.loss_weight_dict = loss_config
    #     self.init_loss_weight(loss_config)

    # def init_loss_weight(self, loss_config):
    #     self.loss_weight_dict = {}
    #     self.loss_weight_dict['img_loss_weight'] = loss_config['img_loss_weight']
    #     self.loss_weight_dict['img_lpips_loss_weight'] = loss_config['img_lpips_loss_weight']
    #     self.loss_weight_dict['sec_loss_weight'] = loss_config['sec_loss_weight']
    #     self.loss_weight_dict['sec_patch_weight_loss_weight'] = loss_config['sec_patch_weight_loss_weight']
    #     self.loss_weight_dict['mask_loss_weight'] = loss_config['mask_loss_weight']
    #     self.loss_weight_dict['edge_loss_weight'] = loss_config['edge_loss_weight']

    def get_loss_weight(self, step: int):
        """
        根据当前 step 返回 loss 权重
        """
        curr_weight = {}
        for k, v in self.loss_weight_dict.items():
            start, end, decay_steps = v["start"], v["end"], v["decay_steps"]

            if decay_steps <= 0:
                # 不衰减，直接固定值
                weight = end
            else:
                ratio = min(step / decay_steps, 1.0)
                weight = start + (end - start) * ratio  # 线性插值

            curr_weight[k] = weight
        return curr_weight
    


import torch
import torch.nn.functional as F

def patch_wise_loss(
    bit_pred_patch: torch.Tensor,      # (B,N,nbit)
    sec: torch.Tensor,                 # (B,nbit)
    patch_weight: torch.Tensor,        # (B,N,1)
    threshold: float = 0.75,
    max_patches: int = 64,
):
    """
    返回 patch-wise BCE loss（只在高置信 patch 里随机选 max_patches 个）
    """
    B, N, nbit = bit_pred_patch.shape

    # (B,N)
    pw = patch_weight.squeeze(-1)  

    # mask：只选择 weight >= threshold 的 patch
    mask = (pw >= threshold)    # (B,N)

    # GT bits: expand to (B,N,nbit) for BCE
    target = sec[:, None, :].expand_as(bit_pred_patch).float()  # (B,N,nbit)

    total_loss = 0.0
    count = 0

    bit_prob = torch.sigmoid(bit_pred_patch)            # (B, N, nbit)
    bit_binary = (bit_prob >= 0.5).float() 
    bit_final = torch.zeros(B, nbit, device=bit_pred_patch.device)

    for b in range(B):
        valid_idx = mask[b].nonzero(as_tuple=False).squeeze(-1)  # (num_valid,)

        if valid_idx.numel() == 0:
            # 随机选 K 个 patch，不考虑权重
            K = min(max_patches, N)
            rand_idx = torch.randperm(N, device=bit_pred_patch.device)[:K]
            selected = rand_idx
        else:
            # 正常路径：从满足阈值的中随机选
            K = min(max_patches, valid_idx.numel())
            selected = valid_idx[torch.randperm(valid_idx.numel())[:K]]

        # 随机选 K 个 patch（K <= num_valid）
        # K = min(max_patches, valid_idx.numel())
        # selected = valid_idx[torch.randperm(valid_idx.numel())[:K]]  # (K,)

        # 选 patch logits 和 target
        pred_sel = bit_pred_patch[b, selected]   # (K,nbit)
        tgt_sel = target[b, selected]            # (K,nbit)

        # 求 BCE loss
        loss_b = F.binary_cross_entropy_with_logits(pred_sel, tgt_sel)
        total_loss += loss_b
        count += 1

        sel_bits = bit_binary[b, selected]                            # (K, nbit)

        # 4. 多数投票（0/1 投票）
        #    sum over selected patches: 得到每个bit有多少票 = 1
        votes = sel_bits.sum(dim=0)
        bit_final[b] = (votes >= (K / 2)).float()              

    if count == 0:
        return torch.tensor(0.0, device=bit_pred_patch.device)

    return total_loss / count, bit_final




import torch

def patch_vote_bits(
    bit_pred_patch_logits: torch.Tensor,   # (B, N, nbit)
    patch_weight: torch.Tensor,            # (B, N, 1)
    threshold: float = 0.75,
    max_patches: int = 64,
):
    """
    对选择的patch解码bit进行多数投票，得到最终bit。
    返回:
        bit_final: (B, nbit)   # 0/1 的最终预测
    """
    B, N, nbit = bit_pred_patch_logits.shape

    # (B, N)
    pw = patch_weight.squeeze(-1)

    # logits → prob → 0/1
    # 先变成概率，再 threshold=0.5 得到每个patch的binary bit
    bit_prob = torch.sigmoid(bit_pred_patch_logits)            # (B, N, nbit)
    bit_binary = (bit_prob >= 0.5).float()                     # (B, N, nbit)

    bit_final = torch.zeros(B, nbit, device=bit_pred_patch_logits.device)

    for b in range(B):

        # 1. 得到高可信 patch 的索引
        valid_idx = (pw[b] >= threshold).nonzero(as_tuple=False).squeeze(-1)  # (Nv,)

        # 若没有任何通过阈值的patch，则退回到所有patch
        if valid_idx.numel() == 0:
            valid_idx = torch.arange(N, device=pw.device)

        # 2. 随机选 K 个 patch
        K = min(max_patches, valid_idx.numel())
        selected = valid_idx[torch.randperm(valid_idx.numel())[:K]]   # (K,)

        # 3. 取所选 patch 的 bit binary
        #    shape: (K, nbit)
        sel_bits = bit_binary[b, selected]                            # (K, nbit)

        # 4. 多数投票（0/1 投票）
        #    sum over selected patches: 得到每个bit有多少票 = 1
        votes = sel_bits.sum(dim=0)                                   # (nbit)

        # 若 >K/2 → bit=1，否则 bit=0
        bit_final[b] = (votes >= (K / 2)).float()                     # (nbit)

    return bit_final



