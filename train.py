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
from models import WatermarkModel
import numpy as np
from utils.degrade import apply_random_degradations, apply_random_degradations_no_clean

import torch.nn as nn
from utils.loss import *
from utils.metrices import *
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config_path",
        type=str,
        default='configs/trainv1.yaml',
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
    
    
    wmmodel = WatermarkModel(model_config)
    

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16": # may result in ``Nan`` error 
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16": # bf16 is recommended
        weight_dtype = torch.bfloat16

    
    params_to_opt = itertools.chain(wmmodel.encoder.parameters(),
                                    wmmodel.decoder.parameters(),
                                    wmmodel.sec_emb.parameters())
    

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
            mask, _ = mask_embedder(mask)
            mask = mask.to(accelerator.device)
            patch_ratio = 1 - mask_to_patch_ratio(mask).to(accelerator.device)
            bsz, C, H, W = img.shape
            sec = torch.randint(0, 2, (bsz, setting_config['nbit'])).to(accelerator.device) # torch.Tensor(np.random.choice([0, 1], (bsz, setting_config['nbit'])))

            img_wm = wmmodel.hide(img,sec)

            if step > traing_config['jnd_step'] and traing_config['use_jnd']:
                img_wm = wmmodel.apply_jnd(img, img_wm)

            is_fuse = False
            if step > stage_config['stage2'] and random.random()< 0.8:
                is_fuse = True
                img_wm = img_wm * (1 - mask) + mask * img
            
            if traing_config['with_degrade'] and step > stage_config['stage1'] and step < stage_config['stage3']:
                img_wm = apply_random_degradations(img_wm)
            
            if  traing_config['with_degrade'] and step > stage_config['stage3']:
                if is_fuse:
                    img_wm = apply_random_degradations(img_wm)
                else:
                    latents = vae.encode(img_wm.half()).latent_dist.sample()
                    img_wm = vae.decode(latents, return_dict=False)[0].float()

            

            decode_sec_patch_weight, decode_sec_patch, decode_sec_img, decode_mask = wmmodel.extract(img_wm)

            

            decode_mask = torch.sigmoid(decode_mask)

            loss_weight_dict = lossweight_updater.get_loss_weight(step)

            ## img loss 
            img_lpips_loss = LPIPS_loss(img_wm, img.float().detach().clone()).mean()
            img_mse_loss = MSE_loss(img_wm, img.float().detach().clone())

            img_loss = img_mse_loss + loss_weight_dict['img_lpips_loss_weight'] * img_lpips_loss

            ## bit loss

            sec_patch_bec_loss = BCE_loss(decode_sec_patch, sec.float())
            sec_img_bec_loss = BCE_loss(decode_sec_img, sec.float())


            sec_patch_weight_loss = MSE_loss(decode_sec_patch_weight[:,1:,], patch_ratio) 

            sec_loss = sec_patch_bec_loss + sec_img_bec_loss + loss_weight_dict['sec_patch_weight_loss_weight'] * sec_patch_weight_loss

            ## mask loss
            mask_bce_loss = WbceLoss(decode_mask, mask)
            mask_edge_loss = Edge_loss(decode_mask, mask)

            mask_loss = mask_bce_loss + loss_weight_dict['edge_loss_weight'] * mask_edge_loss



            loss = loss_weight_dict['img_loss_weight'] * img_loss + loss_weight_dict['sec_loss_weight'] * sec_loss + loss_weight_dict['mask_loss_weight'] * mask_loss 


            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(list(wmmodel.parameters()), 5.0)
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
                avg_sec_patch_bec_loss = accelerator.gather(sec_patch_bec_loss.repeat(train_batchsize)).mean().item()
                avg_sec_img_bec_loss = accelerator.gather(sec_img_bec_loss.repeat(train_batchsize)).mean().item()
                avg_sec_patch_weight_loss = accelerator.gather(sec_patch_weight_loss.repeat(train_batchsize)).mean().item()

                pred_sec_patch_bit = torch.round(torch.sigmoid(decode_sec_patch))
                pred_sec_img_bit = torch.round(torch.sigmoid(decode_sec_img))

                avg_patch_bit_acc = accelerator.gather(((pred_sec_patch_bit.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()
                avg_img_bit_acc = accelerator.gather(((pred_sec_img_bit.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()


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
                    writer.add_scalar("Loss_sec/sec_patch_bec_loss", avg_sec_patch_bec_loss, step)
                    writer.add_scalar("Loss_sec/sec_img_bec_loss", avg_sec_img_bec_loss, step)
                    writer.add_scalar("Loss_sec/sec_patch_weight_loss", avg_sec_patch_weight_loss, step)

                    writer.add_scalar("Loss_mask/mask_all_loss", avg_mask_loss, step)
                    writer.add_scalar("Loss_mask/mask_bce_loss", avg_mask_bce_loss, step)
                    writer.add_scalar("Loss_mask/mask_edge_loss", avg_mask_edge_loss, step)


                    writer.add_scalar("Train_Bit_acc/patch_bit_acc", avg_patch_bit_acc, step)
                    writer.add_scalar("Train_Bit_acc/img_bit_acc", avg_img_bit_acc, step)

                    # log
                    msg = (
                        f"Step: {step:05d}/{all_step:07d} | "
                        f"{'LR:'}{lr:6.6f} | "
                        f"{'Step Loss:'}{avg_loss:6.3f} | "
                        f"{'Img_mse Loss:'}{avg_img_mse_loss:6.3f} | "
                        f"{'Img_lpips Loss:'}{avg_img_lpips_loss:6.3f} | "
                        f"{'Sec_patch_weight Loss:'}{avg_sec_patch_weight_loss:6.3f} | "
                        f"{'Sec_patch_bce Loss:'}{avg_sec_patch_bec_loss:6.3f} | "
                        f"{'Sec_img_bce Loss:'}{avg_sec_patch_weight_loss:6.3f} | "
                        f"{'Mask_bce Loss:'}{avg_mask_bce_loss:6.3f} | "
                        f"{'Mask_edge Loss:'}{avg_mask_edge_loss:6.3f} | "
                        f"{'Bit_patch acc:'}{avg_patch_bit_acc:6.3f}"
                        f"{'Bit_img acc:'}{avg_img_bit_acc:6.3f}"
                    )
                    print(msg)
                    logger.info(msg)
                    if step % log_config['view_step'] == 0: # visualization
                        result_images = torch.cat([img[:train_batchsize], 
                                                img_wm[:train_batchsize], 
                                                ((img_wm - img) *10)[:train_batchsize],
                                                mask.repeat(1, 3, 1, 1)[:train_batchsize], 
                                                decode_mask.repeat(1, 3, 1, 1)[:train_batchsize]],
                                                dim=0).detach().clone()
                        save_image(result_images, os.path.join(traing_config['output_dir'], 'images/train', '%s.jpg' % (step)), normalize=True, scale_each=True, nrow=train_batchsize)

            if step % log_config['val_step'] == 0:
                avg_psnr = 0.0
                avg_ssim = 0.0

                avg_patch_bit_acc_clean = 0.0
                avg_img_bit_acc_clean = 0.0
                avg_patch_bit_acc_noise = 0.0
                avg_img_bit_acc_noise = 0.0 
                avg_patch_bit_acc_vae = 0.0
                avg_img_bit_acc_vae = 0.0 
                avg_patch_bit_acc_fuse = 0.0
                avg_img_bit_acc_fuse = 0.0 
                avg_patch_bit_acc_fuse_noise = 0.0
                avg_img_bit_acc_fuse_noise = 0.0 

                avg_iou = 0.0 
                avg_f1 = 0.0
                avg_auc = 0.0

                avg_iou_noise = 0.0 
                avg_f1_noise = 0.0
                avg_auc_noise = 0.0


                with torch.no_grad():
                    for val_step, val_batch_data in enumerate(tqdm(val_dataloader)):
                        with accelerator.autocast():
                            img = val_batch_data['img']
                            mask = val_batch_data['mask']
                            mask, _ = mask_embedder(mask)
                            mask = mask.to(accelerator.device)
                            patch_ratio = 1 - mask_to_patch_ratio(mask).to(accelerator.device)

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


                            _, decode_sec_patch_clean, decode_sec_img_clean, _ = wmmodel.extract(img_wm_clean)

                            pred_sec_patch_bit_clean = torch.round(torch.sigmoid(decode_sec_patch_clean))
                            pred_sec_img_bit_clean = torch.round(torch.sigmoid(decode_sec_img_clean))

                            avg_patch_bit_acc_clean += accelerator.gather(((pred_sec_patch_bit_clean.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()
                            avg_img_bit_acc_clean += accelerator.gather(((pred_sec_img_bit_clean.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()


                            img_wm_noise = apply_random_degradations_no_clean(img_wm_clean)
                            _, decode_sec_patch_noise, decode_sec_img_noise, _ = wmmodel.extract(img_wm_noise)

                            pred_sec_patch_bit_noise = torch.round(torch.sigmoid(decode_sec_patch_noise))
                            pred_sec_img_bit_noise = torch.round(torch.sigmoid(decode_sec_img_noise))

                            avg_patch_bit_acc_noise += accelerator.gather(((pred_sec_patch_bit_noise.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()
                            avg_img_bit_acc_noise += accelerator.gather(((pred_sec_img_bit_noise.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()
                            

                            with torch.no_grad():
                                latents = vae.encode(img_wm.half()).latent_dist.sample()
                                img_wm_vae = vae.decode(latents, return_dict=False)[0].float()
                            _, decode_sec_patch_vae, decode_sec_img_vae, _ = wmmodel.extract(img_wm_vae)

                            pred_sec_patch_bit_vae = torch.round(torch.sigmoid(decode_sec_patch_vae))
                            pred_sec_img_bit_vae = torch.round(torch.sigmoid(decode_sec_img_vae))

                            avg_patch_bit_acc_vae += accelerator.gather(((pred_sec_patch_bit_vae.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()
                            avg_img_bit_acc_vae += accelerator.gather(((pred_sec_img_bit_vae.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()


                            img_wm_fuse = img_wm_clean * (1 - mask) + img * mask

                            decode_sec_patch_weight, decode_sec_patch_fuse, decode_sec_img_fuse, decode_mask = wmmodel.extract(img_wm_fuse)
                            decode_mask = torch.sigmoid(decode_mask)

                            pred_sec_patch_bit_fuse = torch.round(torch.sigmoid(decode_sec_patch_fuse))
                            pred_sec_img_bit_fuse = torch.round(torch.sigmoid(decode_sec_img_fuse))


                            avg_patch_bit_acc_fuse += accelerator.gather(((pred_sec_patch_bit_fuse.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()
                            avg_img_bit_acc_fuse += accelerator.gather(((pred_sec_img_bit_fuse.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()


                            iou_val = compute_iou(decode_mask, mask)
                            f1_val  = compute_f1(decode_mask, mask)
                            auc_val = compute_auc(decode_mask, mask)

                            avg_iou += accelerator.gather(iou_val).mean().item()
                            avg_f1 += accelerator.gather(f1_val).mean().item()
                            avg_auc += accelerator.gather(auc_val).mean().item()



                            img_wm_fuse_noise = apply_random_degradations_no_clean(img_wm_fuse)

                            decode_sec_patch_weight, decode_sec_patch_fuse_noise, decode_sec_img_fuse_noise, decode_mask_noise = wmmodel.extract(img_wm_fuse_noise)
                            decode_mask_noise = torch.sigmoid(decode_mask_noise)

                            pred_sec_patch_bit_fuse_noise = torch.round(torch.sigmoid(decode_sec_patch_fuse_noise))
                            pred_sec_img_bit_fuse_noise = torch.round(torch.sigmoid(decode_sec_img_fuse_noise))


                            avg_patch_bit_acc_fuse_noise += accelerator.gather(((pred_sec_patch_bit_fuse_noise.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()
                            avg_img_bit_acc_fuse_noise += accelerator.gather(((pred_sec_img_bit_fuse_noise.eq(sec.data)).sum()) / (train_batchsize * setting_config['nbit'])).mean().item()


                            iou_val_noise = compute_iou(decode_mask_noise, mask)
                            f1_val_noise  = compute_f1(decode_mask_noise, mask)
                            auc_val_noise = compute_auc(decode_mask_noise, mask)

                            avg_iou_noise += accelerator.gather(iou_val_noise).mean().item()
                            avg_f1_noise += accelerator.gather(f1_val_noise).mean().item()
                            avg_auc_noise += accelerator.gather(auc_val_noise).mean().item()

                    avg_psnr = avg_psnr / (val_step + 1)
                    avg_ssim = avg_ssim / (val_step + 1)

                    avg_patch_bit_acc_clean = avg_patch_bit_acc_clean / (val_step + 1)
                    avg_img_bit_acc_clean = avg_img_bit_acc_clean / (val_step + 1)
                    avg_patch_bit_acc_noise = avg_patch_bit_acc_noise / (val_step + 1)
                    avg_img_bit_acc_noise = avg_img_bit_acc_noise / (val_step + 1)
                    avg_patch_bit_acc_vae = avg_patch_bit_acc_vae / (val_step + 1)
                    avg_img_bit_acc_vae = avg_img_bit_acc_vae / (val_step + 1)
                    avg_patch_bit_acc_fuse = avg_patch_bit_acc_fuse / (val_step + 1)
                    avg_img_bit_acc_fuse = avg_img_bit_acc_fuse / (val_step + 1)
                    avg_patch_bit_acc_fuse_noise = avg_patch_bit_acc_fuse_noise / (val_step + 1)
                    avg_img_bit_acc_fuse_noise = avg_img_bit_acc_fuse_noise / (val_step + 1)

                    avg_iou = avg_iou / (val_step + 1)
                    avg_f1 = avg_f1 / (val_step + 1)
                    avg_auc = avg_auc / (val_step + 1)

                    avg_iou_noise = avg_iou_noise / (val_step + 1)
                    avg_f1_noise = avg_f1_noise / (val_step + 1)
                    avg_auc_noise = avg_auc_noise / (val_step + 1)

                    
                    if accelerator.is_main_process:

                        writer.add_scalar("Val image/psnr", avg_psnr, step)
                        writer.add_scalar("Val image/ssim", avg_psnr, step)

                        writer.add_scalar("Val Sec Patch/avg_patch_bit_acc_clean", avg_patch_bit_acc_clean, step)
                        writer.add_scalar("Val Sec Patch/avg_patch_bit_acc_noise", avg_patch_bit_acc_noise, step)
                        writer.add_scalar("Val Sec Patch/avg_patch_bit_acc_vae", avg_patch_bit_acc_vae, step)
                        writer.add_scalar("Val Sec Patch/avg_patch_bit_acc_fuse", avg_patch_bit_acc_fuse, step)
                        writer.add_scalar("Val Sec Patch/avg_patch_bit_acc_fuse_noise", avg_patch_bit_acc_fuse_noise, step)


                        writer.add_scalar("Val Sec img/avg_img_bit_acc_clean", avg_img_bit_acc_clean, step)
                        writer.add_scalar("Val Sec img/avg_img_bit_acc_noise", avg_img_bit_acc_noise, step)
                        writer.add_scalar("Val Sec img/avg_img_bit_acc_vae", avg_img_bit_acc_vae, step)
                        writer.add_scalar("Val Sec img/avg_img_bit_acc_fuse", avg_img_bit_acc_fuse, step)
                        writer.add_scalar("Val Sec img/avg_img_bit_acc_fuse_noise", avg_img_bit_acc_fuse_noise, step)

                        writer.add_scalar("Val Mask F1/clean", avg_f1, step)
                        writer.add_scalar("Val Mask F1/noise", avg_f1_noise, step)

                        writer.add_scalar("Val Mask Auc/clean", avg_auc, step)
                        writer.add_scalar("Val Mask Auc/noise", avg_auc_noise, step)

                        writer.add_scalar("Val Mask Iou/clean", avg_iou, step)
                        writer.add_scalar("Val Mask Iou/noise", avg_iou_noise, step)


                        msg = "Eval: " \
                            "Step {:05d}, patch bit correct: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} \n" \
                            "-------------------------------------------------------------------------------------------------------------------------".format(
                                step, avg_patch_bit_acc_clean, avg_patch_bit_acc_noise, avg_patch_bit_acc_vae, avg_patch_bit_acc_fuse, avg_patch_bit_acc_fuse_noise)
                        msg2 = "Eval: " \
                            "Step {:05d}, img bit correct: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} \n" \
                            "-------------------------------------------------------------------------------------------------------------------------".format(
                                step, avg_img_bit_acc_clean, avg_img_bit_acc_noise, avg_img_bit_acc_vae, avg_img_bit_acc_fuse, avg_img_bit_acc_fuse_noise)
                        msg3 = "Eval: " \
                            "Step {:05d}, Iou F1 auc: {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f}\n" \
                            "-------------------------------------------------------------------------------------------------------------------------".format(
                                step, avg_iou, avg_f1, avg_auc, avg_iou_noise, avg_f1_noise,avg_auc_noise)
                        print(msg)
                        print(msg2)
                        print(msg3)
                        logger.info(msg)
                        logger.info(msg2)
                        logger.info(msg3)

                        result_images = torch.cat([img[:train_batchsize], 
                                                img_wm[:train_batchsize], 
                                                ((img_wm - img) *10)[:train_batchsize],
                                                mask.repeat(1, 3, 1, 1)[:train_batchsize], 
                                                decode_mask.repeat(1, 3, 1, 1)[:train_batchsize],
                                                decode_mask_noise.repeat(1, 3, 1, 1)[:train_batchsize]],
                                                dim=0).detach().clone()
                        save_image(result_images, os.path.join(traing_config['output_dir'], 'images/test', '%s.jpg' % step), normalize=True, scale_each=True, nrow=train_batchsize)  
            step += 1

            if step % log_config['save_step'] == 0:
                save_path = os.path.join(traing_config['output_dir'], f"checkpoint")
                accelerator.save_state(save_path, safe_serialization=False)



if __name__ == "__main__":
    main()    