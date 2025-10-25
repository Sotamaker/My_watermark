import torch
import torch.nn as nn

from .embedding import SecEmbed
from .wmdit import WatermarkDiTEncoder,WatermarkDiTDecoder
from .jnd import JND






class WatermarkModel(nn.Module):
    def __init__(
        self,
        model_config,
    ):
        super(WatermarkModel, self).__init__()
        sec_emb_config = model_config['sec_emb_config']
        wm_enc_config = model_config['wm_enc_config']
        wm_dec_config = model_config['wm_dec_config']
        self.encoder = WatermarkDiTEncoder(**wm_enc_config)
        self.decoder = WatermarkDiTDecoder(**wm_dec_config)
        self.sec_emb = SecEmbed(**sec_emb_config)
        self.jnd = JND()

    def apply_jnd(self, img, img_wm):
        img_wm = self.jnd(img, img_wm)
        return img
    
    def hide(self, img, sec):

        sec_all_emb, sec_pix_emb, sec_tok_emb, sec_tok_sum_emb = self.sec_emb(sec)
        img_wm = self.encoder(img, sec_all_emb, sec_pix_emb, sec_tok_emb, sec_tok_sum_emb)

        return img_wm
    
    def extract(self, img):
        decode_sec_patch_weight, decode_sec_patch, decode_sec_img, decode_mask  = self.decoder(img)
        return decode_sec_patch_weight, decode_sec_patch, decode_sec_img, decode_mask
        


# wmmodel = WatermarkModel({},{},None,None)


# x = torch.randn(2,3,512,512)
# sec = torch.randint(0, 2, (2, 64))

# wmmodel.encode(x,sec)



# self.decoder = DecoderResnet(**wm_dec_config)
# self.wm_enc_config = wm_enc_config
# self.wm_dec_config = wm_dec_config
# self.weight_dtype = weight_dtype
# self.device = device
