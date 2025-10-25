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