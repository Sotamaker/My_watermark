import torch
import torch.nn as nn

from .embedding import SecEmbed,SecEmbeder
from .wmdit import WatermarkDiTEncoder,WatermarkDiTEncoder1,WatermarkDiTDecoder,WatermarkDiTDecoder1
from .sparsevit import SparseViT_Mul
from .jnd import JND
import torch.nn.functional as F





class WatermarkModel(nn.Module):
    def __init__(
        self,
        model_config,
    ):
        super(WatermarkModel, self).__init__()
        sec_emb_config = model_config['sec_emb_config']
        wm_enc_config = model_config['wm_enc_config']
        mask_decoder_config = model_config['mask_dec_config']
        sec_decoder_config = model_config['sec_dec_config']
        self.sec_emb = SecEmbed(**sec_emb_config)
        self.encoder = WatermarkDiTEncoder(**wm_enc_config)
        self.mask_decoder = SparseViT_Mul(**mask_decoder_config)
        self.sec_decoder = WatermarkDiTDecoder(**sec_decoder_config)
        
        self.jnd = JND()
    

    def mask_to_patch_ratio(self, mask, patch_size=16):
        """
        Args:
            mask: (B, 1, H, W)  0/1 mask
            patch_size: int, patch 大小 (e.g. 16)
        Returns:
            patch_ratio: (B, H/patch, W/patch) 每个 patch 内 1 的比例
        """
        B, C, H, W = mask.shape
        # 用 avg_pool 把每个 patch 平均化
        pooled = F.avg_pool2d(mask.float(), kernel_size=patch_size, stride=patch_size)
        return pooled.flatten(2).transpose(1, 2)#.squeeze(-1)  # (B, H/patch, W/patch)
        

    def apply_jnd(self, img, img_wm):
        img_wm = self.jnd(img, img_wm)
        return img_wm
    
    def hide(self, img, sec):

        patch_feat, cond, token_emb = self.sec_emb(sec)
        img_wm = self.encoder(img, patch_feat, cond, token_emb)

        return img_wm
    
    def extract(self, img, mask):
        pred_mask = self.mask_decoder(img)
        patch_weight = 1 - self.mask_to_patch_ratio(mask)
        pred_sec = self.sec_decoder(img, patch_weight)
        return pred_mask, pred_sec
    
    def test(self,img):

        pred_mask = self.mask_decoder(img)
        binary_pred_mask = (pred_mask > 0).float()
        patch_weight = 1 - self.mask_to_patch_ratio(binary_pred_mask)
        pred_sec = self.sec_decoder(img, patch_weight)
        return pred_mask, pred_sec
    






class WatermarkModel1(nn.Module):
    def __init__(
        self,
        model_config,
    ):
        super(WatermarkModel1, self).__init__()
        sec_emb_config = model_config['sec_emb_config']
        wm_enc_config = model_config['wm_enc_config']
        mask_decoder_config = model_config['mask_dec_config']
        sec_decoder_config = model_config['sec_dec_config']
        self.sec_emb = SecEmbeder(**sec_emb_config)
        self.encoder = WatermarkDiTEncoder1(**wm_enc_config)
        self.mask_decoder = SparseViT_Mul(**mask_decoder_config)
        self.sec_decoder = WatermarkDiTDecoder1(**sec_decoder_config)
        
        self.jnd = JND()
    

    def mask_to_patch_ratio(self, mask, patch_size=16):
        """
        Args:
            mask: (B, 1, H, W)  0/1 mask
            patch_size: int, patch 大小 (e.g. 16)
        Returns:
            patch_ratio: (B, H/patch, W/patch) 每个 patch 内 1 的比例
        """
        B, C, H, W = mask.shape
        # 用 avg_pool 把每个 patch 平均化
        pooled = F.avg_pool2d(mask.float(), kernel_size=patch_size, stride=patch_size)
        return pooled.flatten(2).transpose(1, 2)#.squeeze(-1)  # (B, H/patch, W/patch)
        

    def apply_jnd(self, img, img_wm):
        img_wm = self.jnd(img, img_wm)
        return img_wm
    
    def hide(self, img, sec):

        patch_feat, cond, token_emb = self.sec_emb(sec)
        img_wm = self.encoder(img, patch_feat, cond, token_emb)

        return img_wm
    
    def extract(self, img, mask):
        pred_mask = self.mask_decoder(img)
        patch_weight = 1 - self.mask_to_patch_ratio(mask)
        bit_pred_global, bit_pred_patch, s, w = self.sec_decoder(img, patch_weight)
        return pred_mask, bit_pred_global,bit_pred_patch, patch_weight, w
    
    def test(self,img):

        pred_mask = self.mask_decoder(img)
        binary_pred_mask = (pred_mask > 0).float()
        patch_weight = 1 - self.mask_to_patch_ratio(binary_pred_mask)
        bit_pred_global, bit_pred_patch, s, w = self.sec_decoder(img, patch_weight)
        return pred_mask, bit_pred_global,bit_pred_patch, patch_weight, w

# wmmodel = WatermarkModel({},{},None,None)


# x = torch.randn(2,3,512,512)
# sec = torch.randint(0, 2, (2, 64))

# wmmodel.encode(x,sec)



# self.decoder = DecoderResnet(**wm_dec_config)
# self.wm_enc_config = wm_enc_config
# self.wm_dec_config = wm_dec_config
# self.weight_dtype = weight_dtype
# self.device = device
