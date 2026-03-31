import numpy as np
import torch

from einops import rearrange, repeat
from torch import nn, einsum
from inspect import isfunction
import torch.nn.functional as F
import os
from diffusers.models.attention import BasicTransformerBlock
import copy
import albumentations as A
from PIL import Image
import torchvision.transforms as transforms
Dim_pro = 32

def max_min_norm(x):    
    x_max = torch.max(x,-1)[0].unsqueeze(-1)
    x_min = torch.min(x,-1)[0].unsqueeze(-1)
    return (x-x_min)/(x_max-x_min)

def replace_cross_attention(attn_base, att_replace, trainer):
    if trainer.num_time_replace[0] <= trainer.i <= trainer.num_time_replace[1]: 
        attn_base_replace = attn_base[:, :, trainer.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * trainer.alphas + att_replace * (1 - trainer.alphas)
    else:
        attn_replace =  att_replace       

    if trainer.equalizer != []:
       attn_replace = attn_replace * trainer.equalizer[:, :, None, :]

    if trainer.equalizer_pos != []:
        if trainer.i <= trainer.num_time_weight[0] : 
            attn_replace = attn_replace * trainer.equalizer_pos[0, :, :, None, :]
        if trainer.num_time_weight[1] <=  trainer.i:   
            attn_replace = attn_replace * trainer.equalizer_pos[1, :, :, None, :]
    
    return attn_replace

def replace_self_attention( attn_base, att_replace, trainer):
    if trainer.num_time_replace[0] <= trainer.i <= trainer.num_time_replace[1]: 
        if att_replace.shape[2] <= Dim_pro ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    else:
        return att_replace


def attn_replace(attn, trainer, h, attn_type_pro = ''):
    batch_size = attn.shape[0] // h
    attn = attn.reshape(batch_size, h, *attn.shape[1:])
    attn_base, attn_repalce = attn[0], attn[1:]
       
    if attn_type_pro=='self':
        attn_repalce_new = attn_repalce
    else:
        attn_repalce_new = replace_cross_attention(attn_base, attn_repalce,trainer)

    attn[1:] = attn_repalce_new
    attn = attn.reshape(batch_size * h, *attn.shape[2:])
    return attn

def attn_pro(attn_save):
    attn_save_pro = {'cross_attn':[],'self_attn':[],'attn_S':[],'self_sim':[],'cross_sim':[]}
    for key,value in attn_save.items():
        if value == {}:
            continue
        value_block_num = value['block_num'] 
        value_tensor = value['all'] / value_block_num
        attn_save_pro[key].append(value_tensor)    
    return attn_save_pro

@torch.no_grad()
def aggregate_cross_attn_map(cross_attns, idx,thres=0.8):
        attn_map = torch.stack(cross_attns, dim=1).mean(1) 
        B = attn_map.shape[0]
        res = int(np.sqrt(attn_map.shape[-2])) 
        attn_map = attn_map.reshape(-1, res, res, attn_map.shape[-1])
        if idx=='mean':
            image = attn_map.mean(-1)
        else:   
            image = [attn_map[ i_a, :, :, id_x].sum(-1) for i_a,id_x in enumerate(idx) ] 
            image = torch.stack(image,0)
        image_min = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        image_max = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        image = (image - image_min) / (image_max - image_min)
        spatial_mask = image.reshape(B,-1)
        spatial_mask[spatial_mask < thres] = 0
        spatial_mask[spatial_mask >= thres] = 1
        return 1-spatial_mask

def next_step(model_output, timestep, sample, scheduler):
        timestep, next_timestep = min(
            timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = scheduler.alphas_cumprod[timestep] if timestep >= 0 else   scheduler.final_alpha_cumprod
        alpha_prod_t_next = scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

@torch.no_grad()
def ddim_loop(latent, unet1, scheduler=None, prompt_embeds=None, pooled_prompt_embeds=None, add_time_ids=None, NUM_DDIM_STEPS=30,trainer=None):
        all_latent = [latent]
        latent = latent.clone().detach()
        if scheduler is None:
            timesteps = torch.arange(0, 999, NUM_DDIM_STEPS) 
        ddim_attn_save={}
        ddim_t_save = {}

        for i in range(NUM_DDIM_STEPS):       
            if scheduler is None:
                t = timesteps[i]
            else:
                t = scheduler.timesteps[len(scheduler.timesteps) - i - 1]
            trainer.uncond = False
            trainer.block=0
            trainer.attn_save = {'cross_attn':{},'self_attn':{}, 'attn_S':{}, 'self_sim':{}, 'cross_sim':{}}

            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            }
            noise_pred = unet1(
                latent,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample
                
            trainer.uncond = True
            latent = next_step(noise_pred, t, latent,scheduler)
            all_latent.append(latent)
            attn_save_i = attn_pro(trainer.attn_save)
            ddim_attn_save.update({i:attn_save_i})
            ddim_t_save.update({i:t})
            del attn_save_i, trainer.attn_save
            torch.cuda.empty_cache()
            if  trainer.DDIM_test:
                del attn_save_i, trainer.attn_save, all_latent
                torch.cuda.empty_cache()
                all_latent =[]
        return all_latent,ddim_attn_save,ddim_t_save

@torch.no_grad()
def ddim_inversion(trainer, latent, unet1, scheduler, prompt_embeds, pooled_prompt_embeds,add_time_ids, NUM_DDIM_STEPS):
    ddim_latents ,ddim_attn_save, ddim_t_save= ddim_loop(latent, unet1, scheduler,prompt_embeds, pooled_prompt_embeds,add_time_ids, NUM_DDIM_STEPS=NUM_DDIM_STEPS, trainer=trainer)
    return ddim_latents , ddim_attn_save, ddim_t_save

def ddim_mask_pro(Bs, NUM_DDIM_STEPS, ddim_attn_save, IDx = [4], prompt = [''],thres=0.7,outdir=''):
        mask_a_save=[]      
        ddim_mask = {}     
        for i in range(NUM_DDIM_STEPS):
            i_ddim = NUM_DDIM_STEPS -1 - i
            attn_all_save=[]            
            attn_k = ddim_attn_save[i_ddim]['cross_sim'][0]  
            attn_k = attn_k.reshape(Bs,-1,*attn_k.shape[-2:]).mean(1) 
            attn_all_save = [attn_k]
            mask_a = aggregate_cross_attn_map(attn_all_save,idx=IDx,thres=thres) 
            ddim_mask.update({i_ddim:mask_a})
        return ddim_mask

def save_model_parameters(trainer, unet, args, i_n, Img_Emb=False, NA_pro=False, FT_out=False):
    """
    Save model parameters based on configuration
    
    Args:
        trainer: Trainer object containing model components
        unet: UNet model
        args: Argument object with configuration
        i_n: Iteration or checkpoint number
        Img_Emb: Whether to save image embedding
        NA_pro: Whether to save Neg_Attn module
        FT_out: Whether to save output parameters
    """
    # Save FT module
    model_FT_path = os.path.join(args.output_dir, f"model-FT-{i_n}.pth")
    torch.save(trainer.FT_module, model_FT_path)
    
    # Save image embedding if enabled
    if Img_Emb:
        model_img_path = os.path.join(args.output_dir, f"model-img-{i_n}.pth")
        torch.save(trainer.img_embedding, model_img_path)
    
    # Save Neg_Attn module if enabled
    if NA_pro:
        model_ra_path = os.path.join(args.output_dir, f"model-Neg_Attn-{i_n}.pth")
        torch.save(trainer.Neg_Attn_module, model_ra_path)
    
    # Save UNet delta parameters
    delta_dict = {'unet': {}}
    for name, params in unet.named_parameters():
        if args.freeze_model == "crossattn":
            if 'attn2' in name:
                delta_dict['unet'][name] = params.cpu().clone()
        elif args.freeze_model == "crossattn_kv":
            if 'attn2.to_k' in name:
                delta_dict['unet'][name] = params.cpu().clone()
            elif 'attn2.to_v' in name:
                delta_dict['unet'][name] = params.cpu().clone()
            if FT_out and 'attn2.to_out' in name:
                delta_dict['unet'][name] = params.cpu().clone()
        else:
            raise ValueError(
                "freeze_model argument only supports crossattn_kv or crossattn"
            )
    torch.save(delta_dict, os.path.join(args.output_dir, f"model-unet-{i_n}.bin"))