import os
import random
import argparse
from pathlib import Path
import itertools
import random
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL
from diffusers.training_utils import cast_training_params
from tqdm import tqdm
from torchvision.utils import save_image
from diffusers.optimization import get_scheduler

from torch.utils.tensorboard import SummaryWriter
import logging
import lpips

from data import COCODataset, get_mask_embedder, mask_to_patch_ratio

from omegaconf import OmegaConf
from models import WatermarkModel,WatermarkModel1
import numpy as np
from utils.degrade import apply_random_degradations, apply_random_degradations_no_clean

import torch.nn as nn
from utils.loss import *
from utils.metrices import *
from utils.view import *
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config_path",
        type=str,
        default='configs/trainv1_small_g.yaml',
        help='Path to Logger YMAL file.',
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--log_type",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--sec_type",
        type=str,
        default="only_global",#only_global only_patch
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args




def main():
    args = parse_args()

    setting_config = OmegaConf.load(args.config_path)
    traing_config = setting_config['train_config']
    data_config = setting_config['datasets']
    mask_config = setting_config['masks']
    mask2_config = setting_config['mask2s']

    loss_config = setting_config['loss']
    model_config = setting_config['model']
    log_config = setting_config['log']
    stage_config = setting_config['stage']

    logging_dir = Path(traing_config['output_dir'], traing_config['logging_dir'])
    accelerator_project_config = ProjectConfiguration(project_dir=traing_config['output_dir'], logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.log_type,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=args.grad_accum_steps
    )
    if accelerator.is_main_process:
        if traing_config['output_dir'] is not None:
            os.makedirs(os.path.join(traing_config['output_dir'], 'images/train'), exist_ok=True)
            os.makedirs(os.path.join(traing_config['output_dir'], 'images/test'), exist_ok=True)
            
    writer = SummaryWriter(traing_config['output_dir'])
    logger = get_logger(__name__)
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            filename=os.path.join(traing_config['output_dir'], 'log.log'))
    
    
    wmmodel = WatermarkModel1(model_config)
    

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16": # may result in ``Nan`` error 
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16": # bf16 is recommended
        weight_dtype = torch.bfloat16

    
    params_to_opt = itertools.chain(wmmodel.parameters())


    def count_trainable_params(module):
        total = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total

    # 在 prepare 之前统计
    print("模型参数统计:")
    for name, submodule in [
        ("Encoder", wmmodel.encoder),
        ("Sec_decoder", wmmodel.sec_decoder),
        ("SecretEmbedder", wmmodel.sec_emb),
        ("Mask_decoder", wmmodel.mask_decoder)
    ]:
        params = count_trainable_params(submodule)
        print(f"  {name:15s}: {params/1e6:.3f} M trainable params")
        logger.info(f"  {name:15s}: {params/1e6:.3f} M trainable params")

    # 统计总参数
    total_params = sum(p.numel() for p in wmmodel.parameters() if p.requires_grad)
    print(f"  {'Total':15s}: {total_params/1e6:.3f} M trainable params")
    logger.info(f"  {'Total':15s}: {total_params/1e6:.3f} M trainable params")

    

    

    optimizer = torch.optim.AdamW(params_to_opt, lr=traing_config['learning_rate'], weight_decay=traing_config['weight_decay'])

    train_dataset = COCODataset(data_config['train'])   
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=data_config['train']['batchsize'],
        num_workers=data_config['train']['n_workers'],
        drop_last=True,
    )

    val_dataset = COCODataset(data_config['val'])
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=data_config['val']['batchsize'],
        num_workers=data_config['val']['n_workers'],
        drop_last=True,
    )

    lr_scheduler = get_scheduler(
        traing_config['lr_scheduler'],
        optimizer=optimizer,
        num_warmup_steps=traing_config['lr_warmup_steps'],
        num_training_steps=traing_config['num_training_steps'],
        #num_cycles = int(args.num_train_epochs // args.cosine_cycle_epoch),
    )

    
    
    all_step = traing_config['all_step']
    step = 0
    losses = 0.0
    mse_losses = 0.0

    log_step = 500
    sample_step = 500
    mask_embedder = get_mask_embedder(**mask_config)
    mask_embedder2 = get_mask_embedder(**mask2_config)

    lossweight_updater = LossWeight(loss_config)

    #
    BCE_loss = nn.BCEWithLogitsLoss()
    MSE_loss= nn.MSELoss()
    LPIPS_loss = lpips.LPIPS(net='vgg').to(accelerator.device)
    Edge_loss = EdgeLoss()

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae",torch_dtype=torch.float16)

    # Prepare everything with our `accelerator`.
    wmmodel,  optimizer, lr_scheduler, train_dataloader, val_dataloader, Edge_loss= \
        accelerator.prepare(wmmodel,  optimizer, lr_scheduler, train_dataloader, val_dataloader, Edge_loss)
    wmmodel = accelerator.unwrap_model(wmmodel)
    vae = vae.to(accelerator.device)

    train_batchsize = data_config['train']['batchsize']
    for current_epoch in range(1000):
        for batch_data in tqdm(train_dataloader):
            lr = lr_scheduler.get_last_lr()[0]
            img = batch_data['img']
            mask = batch_data['mask']
            bsz, C, H, W = img.shape
                        
            sec = torch.randint(0, 2, (bsz, setting_config['nbit'])).to(accelerator.device) # torch.Tensor(np.random.choice([0, 1], (bsz, setting_config['nbit'])))

            img_wm = wmmodel.hide(img, sec)


            if step <=  stage_config['stage1']:
                mask_input = torch.zeros(bsz, 1, H, W)
                is_tamaper = True
            elif stage_config['stage1'] < step <= stage_config['stage3']:
                mask_input, is_tamper = mask_embedder(mask)
            else: # step > stage_config['stage3']:
                mask_input, is_tamaper = mask_embedder2(mask)

            mask_input = mask_input.to(accelerator.device)

            # usr jnd
            if step > traing_config['jnd_step'] and traing_config['use_jnd']:
                img_wm = wmmodel.apply_jnd(img, img_wm)

            # tamper and vae
            if is_tamaper:
                img_wm = img_wm * (1 - mask) + mask * img
            else:
                latents = vae.encode(img_wm.half()).latent_dist.sample()
                img_wm = vae.decode(latents, return_dict=False)[0].float()
            
            # use degradations
            if step > stage_config['stage2'] and is_tamper:
                img_wm = apply_random_degradations(img_wm)


            # extract            
            pred_mask, bit_pred_global, bit_pred_patch, patch_weight, w= wmmodel.extract(img_wm, mask_input)
            
            


            loss_weight_dict = lossweight_updater.get_loss_weight(step)          

            pred_mask = torch.sigmoid(pred_mask)

            

            ## img loss 
            img_lpips_loss = LPIPS_loss(img_wm, img.float().detach().clone()).mean()
            img_mse_loss = MSE_loss(img_wm, img.float().detach().clone())

            img_loss = img_mse_loss + loss_weight_dict['img_lpips_loss_weight'] * img_lpips_loss

            ## bit loss

            sec_loss_g = BCE_loss(bit_pred_global, sec.float())
            sec_loss_p, bit_final_patch = patch_wise_loss(bit_pred_patch, sec.float(), patch_weight)

            if args.sec_type == "only_global":
                sec_loss = sec_loss_g
            elif args.sec_type == "only_patch":
                sec_loss = sec_loss_p
            elif args.sec_type == "all":
                sec_loss = sec_loss_g + 0.5 * sec_loss_p
            else:
                ValueError(f"Invalid mode: {args.sec_type}")


            ## mask loss
            mask_bce_loss = WbceLoss(pred_mask, mask)
            mask_edge_loss = Edge_loss(pred_mask, mask)

            mask_loss = mask_bce_loss + loss_weight_dict['edge_loss_weight'] * mask_edge_loss



            loss = loss_weight_dict['img_loss_weight'] * img_loss + loss_weight_dict['sec_loss_weight'] * sec_loss + loss_weight_dict['mask_loss_weight'] * mask_loss 


            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(list(wmmodel.parameters()), 10.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()


            if step % log_config['log_step'] == 0: # log
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batchsize)).mean().item()

                avg_img_loss = accelerator.gather(img_loss.repeat(train_batchsize)).mean().item()
                avg_img_mse_loss = accelerator.gather(img_mse_loss.repeat(train_batchsize)).mean().item()
                avg_img_lpips_loss = accelerator.gather(img_lpips_loss.repeat(train_batchsize)).mean().item()
                avg_mask_loss = accelerator.gather(mask_loss.repeat(train_batchsize)).mean().item()
                avg_mask_bce_loss = accelerator.gather(mask_bce_loss.repeat(train_batchsize)).mean().item()
                avg_mask_edge_loss = accelerator.gather(mask_edge_loss.repeat(train_batchsize)).mean().item()

                avg_sec_loss = accelerator.gather(sec_loss.repeat(train_batchsize)).mean().item()
                
                pred_sec_bit_g = torch.round(torch.sigmoid(bit_pred_global))
                

                avg_bit_acc_g = accelerator.gather(((pred_sec_bit_g.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()
                avg_bit_acc_p = accelerator.gather(((bit_final_patch.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()

                if accelerator.is_main_process:
                    # config
                    writer.add_scalar("LR", lr, step)
                    for k, v in loss_weight_dict.items():
                        writer.add_scalar(f"Loss_weight/ {k}" , v, step)

                    # loss 
                    writer.add_scalar("Loss/total_loss", avg_loss, step)

                    writer.add_scalar("Loss_img/img_all_loss", avg_img_loss, step)
                    writer.add_scalar("Loss_img/img_mse_loss", avg_img_mse_loss, step)
                    writer.add_scalar("Loss_img/img_lpips_loss", avg_img_lpips_loss, step)

                    writer.add_scalar("Loss_sec/sec_all_loss", avg_sec_loss, step)
                   

                    writer.add_scalar("Loss_mask/mask_all_loss", avg_mask_loss, step)
                    writer.add_scalar("Loss_mask/mask_bce_loss", avg_mask_bce_loss, step)
                    writer.add_scalar("Loss_mask/mask_edge_loss", avg_mask_edge_loss, step)


                    writer.add_scalar("Train_Bit_acc/avg_bit_acc_g", avg_bit_acc_g, step)
                    writer.add_scalar("Train_Bit_acc/avg_bit_acc_p", avg_bit_acc_p, step)

                    # log
                    msg = (
                        f"Step: {step:05d}/{all_step:07d} | "
                        f"{'LR:'}{lr:6.6f} | "
                        f"{'Step Loss:'}{avg_loss:6.3f} | "
                        f"{'Img_mse Loss:'}{avg_img_mse_loss:6.3f} | "
                        f"{'Img_lpips Loss:'}{avg_img_lpips_loss:6.3f} | "
                        f"{'Sec_Loss:'}{avg_sec_loss:6.3f} | "
                        f"{'Mask_bce Loss:'}{avg_mask_bce_loss:6.3f} | "
                        f"{'Mask_edge Loss:'}{avg_mask_edge_loss:6.3f} | "
                        f"{'Bit acc_g:'}{avg_bit_acc_g:6.3f}"
                        f"{'Bit acc_p:'}{avg_bit_acc_p:6.3f}"
                    )
                    print(msg)
                    logger.info(msg)
                    if step % log_config['view_step'] == 0: # visualization
                        result_images = torch.cat([img[:train_batchsize], 
                                                img_wm[:train_batchsize], 
                                                ((img_wm - img) *10)[:train_batchsize],
                                                mask.repeat(1, 3, 1, 1)[:train_batchsize], 
                                                pred_mask.repeat(1, 3, 1, 1)[:train_batchsize]],
                                                dim=0).detach().clone()
                        save_image(result_images, os.path.join(traing_config['output_dir'], 'images/train', '%s.jpg' % (step)), normalize=True, scale_each=True, nrow=train_batchsize)

            if step % log_config['val_step'] == 0:
                avg_psnr = 0.0
                avg_ssim = 0.0

                avg_bit_acc_clean_p = 0.0
                avg_bit_acc_noise_p = 0.0
                avg_bit_acc_vae_p = 0.0
                avg_bit_acc_fuse_p = 0.0
                avg_bit_acc_fuse_noise_p = 0.0

                avg_bit_acc_clean_g = 0.0
                avg_bit_acc_noise_g = 0.0
                avg_bit_acc_vae_g = 0.0
                avg_bit_acc_fuse_g = 0.0
                avg_bit_acc_fuse_noise_g = 0.0
                
                avg_iou = 0.0 
                avg_f1 = 0.0
                avg_auc = 0.0

                avg_iou_noise = 0.0 
                avg_f1_noise = 0.0
                avg_auc_noise = 0.0

                # 用来收集多个 batch 可视化结果
                save_count = 0 
                collected_imgs = []
                collected_imgs_wm = []
                collected_diffs = []
                collected_mask = []
                collected_pred_mask = []
                collected_pred_mask_noise = []


                with torch.no_grad():
                    for val_step, val_batch_data in enumerate(tqdm(val_dataloader)):
                        with accelerator.autocast():
                            img = val_batch_data['img']
                            mask = val_batch_data['mask']
                            mask, _ = mask_embedder(mask)
                            mask = mask.to(accelerator.device)
                            

                            bsz, C, H, W = img.shape
                            sec = torch.randint(0, 2, (bsz, setting_config['nbit'])).to(accelerator.device) # torch.Tensor(np.random.choice([0, 1], (bsz, setting_config['nbit'])))

                            img_wm_clean = wmmodel.hide(img,sec)

                            img_wm_clean_01   = (img_wm_clean.clamp(-1, 1) + 1) / 2
                            img_01 = (img + 1) / 2

                            psnr_val = compute_psnr(img_wm_clean_01, img_01, max_val=1.0)
                            avg_psnr += accelerator.gather(psnr_val).mean().item()

                            # -------- SSIM --------
                            ssim_val = compute_ssim(img_wm_clean_01, img_01, max_val=1.0)
                            avg_ssim += accelerator.gather(ssim_val).mean().item()


                            pred_mask, bit_pred_global, bit_pred_patch, patch_weight,w = wmmodel.test(img_wm_clean) #extract(img_wm_clean, torch.zeros_like(mask).to(accelerator.device)) #


                            pred_sec_bit_clean_g = torch.round(torch.sigmoid(bit_pred_global))
                            
                            pred_sec_bit_clean_p =  patch_vote_bits(bit_pred_patch, patch_weight)
                            avg_bit_acc_clean_g += accelerator.gather(((pred_sec_bit_clean_g.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()
                            avg_bit_acc_clean_p += accelerator.gather(((pred_sec_bit_clean_p.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()
                            


                            img_wm_noise = apply_random_degradations_no_clean(img_wm_clean)
                            pred_mask_noise, bit_pred_global_noise, bit_pred_patch_noise, patch_weight_noise ,w_noise = wmmodel.test(img_wm_noise) #extract(img_wm_noise, torch.zeros_like(mask).to(accelerator.device)) #

                            pred_sec_bit_noise_g = torch.round(torch.sigmoid(bit_pred_global_noise))
                            pred_sec_bit_noise_p =  patch_vote_bits(bit_pred_patch_noise, patch_weight_noise)
                            
                            avg_bit_acc_noise_g += accelerator.gather(((pred_sec_bit_noise_g.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()
                            avg_bit_acc_noise_p += accelerator.gather(((pred_sec_bit_noise_p.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()
                            

                            with torch.no_grad():
                                latents = vae.encode(img_wm_clean.half()).latent_dist.sample()
                                img_wm_vae = vae.decode(latents, return_dict=False)[0].float()
                            pred_mask_vae, bit_pred_global_vae, bit_pred_patch_vae, patch_weight_vae ,w_vae= wmmodel.test(img_wm_vae) #extract(img_wm_vae, torch.zeros_like(mask).to(accelerator.device)) #

                            pred_sec_bit_vae_g = torch.round(torch.sigmoid(bit_pred_global_vae))
                            pred_sec_bit_vae_p =  patch_vote_bits(bit_pred_patch_vae, patch_weight_vae)

                            avg_bit_acc_vae_g += accelerator.gather(((pred_sec_bit_vae_g.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()
                            avg_bit_acc_vae_p += accelerator.gather(((pred_sec_bit_vae_p.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()
                                        

                            img_wm_fuse = img_wm_clean * (1 - mask) + img * mask

                            pred_mask_fuse, bit_pred_global_fuse, bit_pred_patch_fuse, patch_weight_fuse ,w_fuse= wmmodel.test(img_wm_fuse) #extract(img_wm_fuse, mask) #
                            pred_mask_fuse = torch.sigmoid(pred_mask_fuse)

                            pred_sec_bit_fuse_g = torch.round(torch.sigmoid(bit_pred_global_fuse))
                            pred_sec_bit_fuse_p =  patch_vote_bits(bit_pred_patch_fuse, patch_weight_fuse)


                            avg_bit_acc_fuse_g += accelerator.gather(((pred_sec_bit_fuse_g.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()
                            avg_bit_acc_fuse_p += accelerator.gather(((pred_sec_bit_fuse_p.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()
                            


                            iou_val = compute_iou(pred_mask_fuse, mask)
                            f1_val  = compute_f1(pred_mask_fuse, mask)
                            auc_val = compute_auc(pred_mask_fuse, mask)

                            avg_iou += accelerator.gather(iou_val).mean().item()
                            avg_f1 += accelerator.gather(f1_val).mean().item()
                            avg_auc += accelerator.gather(auc_val).mean().item()

                            img_wm_fuse_noise = apply_random_degradations_no_clean(img_wm_fuse)

                            pred_mask_fuse_noise, bit_pred_global_fuse_nose, bit_pred_patch_fuse_noise, patch_weight_fuse_noise ,w_fuse_noise= wmmodel.test(img_wm_fuse_noise) #.extract(img_wm_fuse_noise, mask) #
                            pred_mask_fuse_noise = torch.sigmoid(pred_mask_fuse_noise)

                            pred_sec_bit_fuse_noise_g = torch.round(torch.sigmoid(bit_pred_global_fuse_nose))
                            pred_sec_bit_fuse_noise_p =  patch_vote_bits(bit_pred_patch_fuse_noise, patch_weight_fuse_noise)

                            avg_bit_acc_fuse_noise_g += accelerator.gather(((pred_sec_bit_fuse_noise_g.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()
                            avg_bit_acc_fuse_noise_p += accelerator.gather(((pred_sec_bit_fuse_noise_p.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()


                            iou_val_noise = compute_iou(pred_mask_fuse_noise, mask)
                            f1_val_noise  = compute_f1(pred_mask_fuse_noise, mask)
                            auc_val_noise = compute_auc(pred_mask_fuse_noise, mask)

                            avg_iou_noise += accelerator.gather(iou_val_noise).mean().item()
                            avg_f1_noise += accelerator.gather(f1_val_noise).mean().item()
                            avg_auc_noise += accelerator.gather(auc_val_noise).mean().item()



                            # ---------- 收集图像 ----------
                            if accelerator.is_main_process and save_count < 6:
                                collected_imgs.append(img[:train_batchsize].detach().cpu())
                                collected_imgs_wm.append(img_wm_clean[:train_batchsize].detach().cpu())
                                collected_diffs.append(((img_wm_clean - img) * 10)[:train_batchsize].detach().cpu())
                                collected_mask.append(mask[:train_batchsize].detach().cpu())
                                collected_pred_mask.append(pred_mask_fuse[:train_batchsize].detach().cpu())
                                collected_pred_mask_noise.append(pred_mask_fuse_noise[:train_batchsize].detach().cpu())
                                save_count += 1

                    

                    avg_bit_acc_clean_p = avg_bit_acc_clean_p / (val_step + 1)
                    avg_bit_acc_noise_p = avg_bit_acc_noise_p / (val_step + 1)
                    avg_bit_acc_vae_p = avg_bit_acc_vae_p / (val_step + 1)
                    avg_bit_acc_fuse_p = avg_bit_acc_fuse_p / (val_step + 1)
                    avg_bit_acc_fuse_noise_p = avg_bit_acc_fuse_noise_p / (val_step + 1)

                    avg_bit_acc_clean_g = avg_bit_acc_clean_g / (val_step + 1)
                    avg_bit_acc_noise_g = avg_bit_acc_noise_g / (val_step + 1)
                    avg_bit_acc_vae_g = avg_bit_acc_vae_g / (val_step + 1)
                    avg_bit_acc_fuse_g = avg_bit_acc_fuse_g / (val_step + 1)
                    avg_bit_acc_fuse_noise_g = avg_bit_acc_fuse_noise_g / (val_step + 1)

                    avg_iou = avg_iou / (val_step + 1)
                    avg_f1 = avg_f1 / (val_step + 1)
                    avg_auc = avg_auc / (val_step + 1)

                    avg_iou_noise = avg_iou_noise / (val_step + 1)
                    avg_f1_noise = avg_f1_noise / (val_step + 1)
                    avg_auc_noise = avg_auc_noise / (val_step + 1)
                    avg_psnr = avg_psnr /  (val_step + 1)
                    avg_ssim = avg_ssim /  (val_step + 1)

                    if accelerator.is_main_process:

                        writer.add_scalar("Val image/psnr", avg_psnr, step)
                        writer.add_scalar("Val image/ssim", avg_ssim, step)

                        writer.add_scalar("Val Sec_g /avg_bit_acc_clean", avg_bit_acc_clean_g, step)
                        writer.add_scalar("Val Sec_g /avg_bit_acc_noise", avg_bit_acc_noise_g, step)
                        writer.add_scalar("Val Sec_g /avg_bit_acc_vae", avg_bit_acc_vae_g, step)
                        writer.add_scalar("Val Sec_g /avg_bit_acc_fuse", avg_bit_acc_fuse_g, step)
                        writer.add_scalar("Val Sec_g /avg_bit_acc_fuse_noise", avg_bit_acc_fuse_noise_g, step)

                        
                        writer.add_scalar("Val Sec_p /avg_bit_acc_clean", avg_bit_acc_clean_p, step)
                        writer.add_scalar("Val Sec_p /avg_bit_acc_noise", avg_bit_acc_noise_p, step)
                        writer.add_scalar("Val Sec_p /avg_bit_acc_vae", avg_bit_acc_vae_p, step)
                        writer.add_scalar("Val Sec_p /avg_bit_acc_fuse", avg_bit_acc_fuse_p, step)
                        writer.add_scalar("Val Sec_p /avg_bit_acc_fuse_noise", avg_bit_acc_fuse_noise_p, step)

                        writer.add_scalar("Val Mask F1/clean", avg_f1, step)
                        writer.add_scalar("Val Mask F1/noise", avg_f1_noise, step)

                        writer.add_scalar("Val Mask Auc/clean", avg_auc, step)
                        writer.add_scalar("Val Mask Auc/noise", avg_auc_noise, step)

                        writer.add_scalar("Val Mask Iou/clean", avg_iou, step)
                        writer.add_scalar("Val Mask Iou/noise", avg_iou_noise, step)

                        
                        msg1 = "Eval: " \
                            "Step {:05d}, img quality : {:.3f} {:.3f}  \n" \
                            "-------------------------------------------------------------------------------------------------------------------------".format(
                                step, avg_psnr, avg_ssim)
                        
                        msg2 = "Eval: " \
                            "Step {:05d}, bit correct_g: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} \n" \
                            "-------------------------------------------------------------------------------------------------------------------------".format(
                                step, avg_bit_acc_clean_g, avg_bit_acc_noise_g, avg_bit_acc_vae_g, avg_bit_acc_fuse_g, avg_bit_acc_fuse_noise_g)
                        msg3 = "Eval: " \
                            "Step {:05d}, bit correct_p: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} \n" \
                            "-------------------------------------------------------------------------------------------------------------------------".format(
                                step, avg_bit_acc_clean_p, avg_bit_acc_noise_p, avg_bit_acc_vae_p, avg_bit_acc_fuse_p, avg_bit_acc_fuse_noise_p)
                        

                        msg4 = "Eval: " \
                            "Step {:05d}, Iou F1 auc: {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f}\n" \
                            "-------------------------------------------------------------------------------------------------------------------------".format(
                                step, avg_iou, avg_f1, avg_auc, avg_iou_noise, avg_f1_noise,avg_auc_noise)
                        print(msg1)
                        print(msg2)
                        print(msg3)
                        print(msg4)
                        logger.info(msg1)
                        logger.info(msg2)
                        logger.info(msg3)
                        logger.info(msg4)

                        # ---------- 拼接保存图像 ----------
                    if len(collected_imgs) > 0:
                        imgs_all = torch.cat(collected_imgs, dim=0)
                        imgs_wm_all = torch.cat(collected_imgs_wm, dim=0)
                        diffs_all = torch.cat(collected_diffs, dim=0)
                        mask_all = torch.cat(collected_mask, dim=0)
                        pred_mask_all = torch.cat(collected_pred_mask, dim=0)
                        pred_mask_noise_all = torch.cat(collected_pred_mask_noise, dim=0)

                        save_dir = os.path.join(traing_config['output_dir'], 'images/test')
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f'{step}.jpg')

                        save_val_grid(
                            imgs_all, imgs_wm_all, diffs_all,
                            mask_all, pred_mask_all, pred_mask_noise_all,
                            save_path
                        )

                        # result_images = torch.cat([img[:train_batchsize], 
                        #                         img_wm_clean[:train_batchsize], 
                        #                         ((img_wm_clean - img) *10)[:train_batchsize],
                        #                         mask.repeat(1, 3, 1, 1)[:train_batchsize], 
                        #                         pred_mask_fuse.repeat(1, 3, 1, 1)[:train_batchsize],
                        #                         pred_mask_fuse_noise.repeat(1, 3, 1, 1)[:train_batchsize]],
                        #                         dim=0).detach().clone()
                        # save_image(result_images, os.path.join(traing_config['output_dir'], 'images/test', '%s.jpg' % step), normalize=True, scale_each=True, nrow=train_batchsize)  
            step += 1

            if step % log_config['save_step'] == 0:
                save_path = os.path.join(traing_config['output_dir'], f"checkpoint_{step}")
                accelerator.save_state(save_path, safe_serialization=False)



if __name__ == "__main__":
    main()    