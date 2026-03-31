import argparse
import itertools
import logging
import math
import os
from pathlib import Path
from typing import Optional
import torch
import json
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint
from packaging import version
from typing import Optional, Union, Tuple, List, Callable, Dict

import transformers
import diffusers
from accelerate.logging import get_logger
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from diffusers.utils.import_utils import is_xformers_available
from diffusers_data_pipeline_my_sdxl import DiffusionDataset, PromptDataset, collate_fn
from src.diffusers_data_pipeline_my_sdxl import encode_prompt_sdxl_for_ddim,make_add_time_ids_sdxl
from VGG import VGGPerceptualLoss        
import torch.nn.functional as nnf

logger = get_logger(__name__)
from diffusers import DDIMScheduler
from util_others  import (
    max_min_norm,
    ddim_mask_pro,
    ddim_inversion,
    next_step, 
    attn_pro, 
    save_model_parameters)          

from ADisen  import (
    create_A_Disen, 
    A_Disen_Trainer, 
    load_sdxl_text_encoders, 
    load_sdxl_tokenizers
    )   

from inspect import isfunction
from seed import seed_everything
def exists(x):
    return x is not None
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

Dim_pro = 32

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--real_prior",
        default=False,
        action="store_true",
        help="real images as prior.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="diffusion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--freeze_model",
        type=str,
        default='crossattn_kv',
        help="crossattn to enable fine-tuning of all key, value, query matrices",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
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
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--hflip", action="store_true", help="Apply horizontal flip data augmentation.")
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--inner_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--Seed", type=int, default=42, help="Seed for random numbers.",)

    # model control signals
    parser.add_argument("--Attn_loss_pro", action="store_true",  help="Use attention loss or not.",)
    parser.add_argument("--Img_Emb_pro", type=str, default='type2-2-vgg-sum',  help="Use attention loss or not.",)
    parser.add_argument("--FT_KV", action="store_true",  help="Fine-tune key-value projection layer or not.",)
    parser.add_argument("--FT_out", action="store_true",  help="Fine-tune output projection layer or not.",)
    parser.add_argument("--NA_pro", action="store_true",  help="Use negative attention or not.",)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def prev_step(model_output, timestep, sample, scheduler):
        prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        #print('prev_timestep',prev_timestep)
        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample



def get_noise_pred(scheduler, unet, latents, t, is_forward=True, context=None, added_cond_kwargs_all=None, trainer=None, GUIDANCE_SCALE=7.5):
        
        uncond_embeddings, cond_embeddings = context
        uncond_added_cond_kwargs, cond_added_cond_kwargs = added_cond_kwargs_all
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        trainer.uncond = True
        trainer.attn_save= {'cross_attn':{},'self_attn':{}, 'attn_S':{}}
        trainer.block =0
        noise_pred_uncond = unet(latents, t, encoder_hidden_states=uncond_embeddings, added_cond_kwargs=uncond_added_cond_kwargs).sample  
        trainer.uncond = False
        trainer.attn_save= {'cross_attn':{},'self_attn':{}, 'attn_S':{}}
        trainer.block =0
        noise_prediction_text = unet(latents, t, encoder_hidden_states=cond_embeddings, added_cond_kwargs=cond_added_cond_kwargs).sample  
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = next_step(noise_pred, t, latents, scheduler)
        else:
            latents = prev_step(noise_pred, t, latents, scheduler)
        return latents


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args):
    seed_everything(args.seed)
    logging_dir = Path(args.output_dir, args.logging_dir)
    print('logging_dir',logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # load model:
    tokenizer_one, tokenizer_two = load_sdxl_tokenizers(args.pretrained_model_name_or_path, args.revision)
    text_encoder_one, text_encoder_two = load_sdxl_text_encoders(args.pretrained_model_name_or_path, args.revision)
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
    VGG = VGGPerceptualLoss().to(accelerator.device)


    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    NUM_DDIM_STEPS = args.ddim_steps
    trainer = A_Disen_Trainer(args.Img_Emb_pro, NUM_DDIM_STEPS, NA_pro)    

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    train_dataset = DiffusionDataset(
        concepts_list=[{
            "instance_prompt": args.instance_prompt,
            "class_prompt": args.class_prompt,
            "instance_data_dir": args.instance_data_dir,
        }],
        size=args.resolution,
        center_crop=args.center_crop,
        hflip=args.hflip,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )

    num_epoch , num_inner_steps = 1, args.inner_steps 
    GUIDANCE_SCALE = 7.5
    
    x = np.linspace(0, NUM_DDIM_STEPS - 1, NUM_DDIM_STEPS)
    NUM_INNER_STEPS = np.ceil(num_inner_steps * np.exp(-.1 * x))

    noise_scheduler.set_timesteps(NUM_DDIM_STEPS)
    for step, batch_i in enumerate(train_dataloader):
        batch = batch_i
    
    
    trainer.is_train = False
    unet, layer_name = create_A_Disen(unet, args.freeze_model, trainer, DDIM_pro = True, FT_out = args.FT_out, FT_KV = args.FT_KV, DDIM_M = True, NA_pro=args.NA_pro)
    text_s = args.class_prompt
    prompt = [args.instance_prompt]
    
    print('ddim inversion before training')
    with torch.no_grad():
        Bs = len(prompt) 
        trainer.DDIM = True
        latents = vae.encode(batch["pixel_values"].to(accelerator.device).to(dtype=weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        prompt_embeds, pooled_prompt_embeds = encode_prompt_sdxl_for_ddim(prompt, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, device=accelerator.device, out_dtype=weight_dtype )
        original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        crops_coords_top_left = (0, 0)

        add_time_ids = make_add_time_ids_sdxl(
            unet=unet,
            text_encoder_2=text_encoder_two,
            original_size=original_size,
            crops_coords_top_left=crops_coords_top_left,
            target_size=target_size,
            batch_size=Bs,
            device=accelerator.device,
            dtype=weight_dtype,
            )
        latents_all = latents.repeat(Bs,1,1,1) 
        ddim_latents, ddim_attn_save, ddim_t_save = ddim_inversion(trainer, latents_all,unet, noise_scheduler,prompt_embeds, pooled_prompt_embeds,add_time_ids, NUM_DDIM_STEPS)     
        ddim_attn_save.update({int(len(ddim_attn_save)):ddim_attn_save[len(ddim_attn_save)-1]})
        trainer.DDIM = False
        del latents, latents_all
        torch.cuda.empty_cache()


    Id_prompt = [ list(np.arange(4, len(prompt_x.split(' '))+1)) for prompt_x in prompt]
    ddim_mask = ddim_mask_pro(Bs, NUM_DDIM_STEPS, ddim_attn_save, Id_prompt,prompt, thres = 0.5) 
    trainer.ID_c = Id_prompt[0]

    ddim_attn_save_1={}
    
    # processing for Token-prior disentangled contrastive loss
    for i_d in range(NUM_DDIM_STEPS):
        ddim_attn_save_1.update({NUM_DDIM_STEPS - i_d:{'cross_attn_c':[],'cross_attn_S':[], 'self_attn_c':[],'self_attn_S':[]}})
        mask_a = ddim_mask[NUM_DDIM_STEPS - 1 - i_d][0:1].detach().unsqueeze(-1)

        L_try = torch.tensor(.0).to(accelerator.device)
        sim_k = ddim_attn_save[NUM_DDIM_STEPS - i_d]['cross_sim'][0]
            
        attn_c, attn_S = sim_k.clone(), sim_k.clone()

        max_neg_value = torch.finfo(attn_c.dtype).min

        attn_c = attn_c.masked_fill_(mask_a==1, max_neg_value).softmax(-1)
        attn_S = attn_S.masked_fill_(mask_a==0, max_neg_value).softmax(-1)

                
        if  sim_k.shape[-1]==77: 
            ddim_attn_save_1[NUM_DDIM_STEPS - i_d]['cross_attn_c'].append(attn_c) 
            ddim_attn_save_1[NUM_DDIM_STEPS - i_d]['cross_attn_S'].append(attn_S) 
        else:
            ddim_attn_save_1[NUM_DDIM_STEPS - i_d]['self_attn_c'].append(attn_c) 
            ddim_attn_save_1[NUM_DDIM_STEPS - i_d]['self_attn_S'].append(attn_S)       

    del ddim_attn_save_1
    
    trainer.is_train = True
    unet, layer_name = create_A_Disen(unet, args.freeze_model, trainer, FT_out = args.FT_out,FT_KV = args.FT_KV ,DDIM_M = True, NA_pro=args.NA_pro)
    trainer.layer_name_all = layer_name['down'] + layer_name['mid'] + layer_name['up']
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer_class = torch.optim.AdamW
    params_all =[]
    if args.FT_KV:
        params_all.append(trainer.FT_module.parameters())
    if args.FT_out:
        params_all+= [x[1] for x in unet.named_parameters() if ('attn2.to_out' in x[0])]
    if args.Img_Emb_pro:
        params_all.append(trainer.img_embedding.parameters())
    if args.NA_pro:
        params_all.append(trainer.Neg_Attn_module.parameters())
    
    params_to_optimize = itertools.chain( params_all,  [x[1] for x in unet.named_parameters() if  (  'attn2.to_v.weight' in x[0]  or 'attn2.to_k.weight' in x[0]  )])
    params_to_optimize =[{"params":p} for p in params_to_optimize]
 
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    args.max_train_steps = int(np.sum(NUM_INNER_STEPS))
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("A-Disen")

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    bar = tqdm(total=int(np.sum(NUM_INNER_STEPS)), colour='red', ncols=100)
        
    for step, batch_i in enumerate(train_dataloader):
        batch = batch_i 

    unet.train()
    if args.Img_Emb_pro: trainer.img_embedding.train()
    if args.FT_KV: trainer.FT_module.train()  
    if args.NA_pro: trainer.Neg_Attn_module.train()

    for epoch in range(num_epoch): 
        latent_cur = ddim_latents[-1][0:1]
        for i_n in range(NUM_DDIM_STEPS):
            num_inner_steps = int(NUM_INNER_STEPS[i_n])
            latent_prev = ddim_latents[len(ddim_latents) - i_n - 2][0].unsqueeze(0)
            t = noise_scheduler.timesteps[i_n]

            mask_a = ddim_mask[NUM_DDIM_STEPS - 1 - i_n][0:1].detach() # [1, res * res]
            trainer.mask = mask_a.reshape(1, Dim_pro, Dim_pro)                
            trainer.i = i_n
            mask_a =mask_a.unsqueeze(-1)

            with torch.no_grad():
                # process uncondition prompt
                uncond_prompt_embeds, uncond_pooled_prompt_embeds = encode_prompt_sdxl_for_ddim("", tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, device=accelerator.device, out_dtype=weight_dtype )
                uncond_added_cond_kwargs = {
                    "text_embeds": uncond_pooled_prompt_embeds,
                    "time_ids": add_time_ids[:1],
                }
            
            train_module=[unet]
            if args.Img_Emb_pro: train_module.append(trainer.img_embedding)
            if args.FT_KV: train_module.append(trainer.FT_module)
            if args.NA_pro: train_module.append(trainer.Neg_Attn_module)
            train_module=tuple(train_module)
            for j in range(num_inner_steps):
                with accelerator.accumulate(train_module):
                    cond_prompt_embeds, cond_pooled_prompt_embeds = encode_prompt_sdxl_for_ddim(batch["prompts"], tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, device=accelerator.device, out_dtype=weight_dtype )
                    cond_added_cond_kwargs = {
                        "text_embeds": cond_pooled_prompt_embeds,
                        "time_ids": add_time_ids[:1],
                    }
                    
                    # IMG_Embedding processing
                    if 'vgg' in  args.Img_Emb_pro :
                        f=VGG(batch["image_gt"])
                        # f on 4 layers : [1, 64, 224, 224] , [1, 128, 112, 112] , [1, 256, 56, 56] , [1, 512, 28, 28]
                        if 'vgg-sum' in args.Img_Emb_pro or 'vgg-linear' in args.Img_Emb_pro:
                            trainer.image_embeddings=f
                        else:
                            trainer.image_embeddings=f[-1].reshape(*f[-1].shape[:2],-1).permute(0, 2, 1)

                    img_f = trainer.forward_embed_img(args.Img_Emb_pro)


                    if args.Img_Emb_pro :
                        con_f = (cond_prompt_embeds, img_f)
                        uncon_f = (uncond_prompt_embeds, img_f)

                    # uncondition process
                    trainer.uncond = True
                    trainer.block = 0
                    trainer.attn_save= {'cross_attn':{},'self_attn':{}, 'attn_S':{}}
                    noise_pred_uncond = unet(
                        latent_cur,
                        t,
                        encoder_hidden_states=uncon_f,
                        added_cond_kwargs=uncond_added_cond_kwargs,
                    ).sample   

                    # condition process
                    trainer.uncond = False
                    trainer.block=0
                    del trainer.attn_save, uncon_f, uncond_added_cond_kwargs
                    torch.cuda.empty_cache()
                    trainer.attn_save= {'cross_attn':{},'self_attn':{}, 'attn_S':{}}
                    noise_pred_cond = unet(
                        latent_cur,
                        t,
                        encoder_hidden_states=con_f,
                        added_cond_kwargs=cond_added_cond_kwargs,
                    ).sample   

                    

                    noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                    latents_prev_rec = prev_step(noise_pred, t, latent_cur, noise_scheduler)

                    loss_rec = F.mse_loss(latents_prev_rec.float(), latent_prev.float(), reduction="none")
                    loss_rec = loss_rec.sum([1, 2, 3]).mean()
                    
                    #### introduce attention-map supervision with Token-prior disentangled contrastive loss
                    attn_loss_c = torch.tensor(.0).to(accelerator.device)
                    attn_loss_S = torch.tensor(.0).to(accelerator.device)
                    attn_loss_nag = torch.tensor(.0).to(accelerator.device) 
                    if args.Attn_loss_pro: 
                        attn_save_pro = attn_pro(trainer.attn_save) # get attention map
                        
                        # foreground attention loss      
                        attn_gt_all = ddim_attn_save_1[NUM_DDIM_STEPS - i_n]['cross_attn_c'] 
                        for attn_gt, attn in zip(attn_gt_all, attn_save_pro['cross_attn']):
                            attn_gt = attn_gt.reshape(Bs,-1,*attn_gt.shape[-2:])[0].detach() 

                            attn_loss_c += nnf.mse_loss(attn_gt, attn)

                        # background attention loss and constrative loss
                        attn_gt1_all = ddim_attn_save_1[NUM_DDIM_STEPS - i_n]['cross_attn_S'] 
                        for attn_gt1, attn1, attn_c in zip(attn_gt1_all, attn_save_pro['attn_S'], attn_save_pro['cross_attn']):
                            attn_gt1 = attn_gt1.reshape(Bs,-1,*attn_gt1.shape[-2:])[0]
                            m=0.05
                            nagative_loss = nnf.mse_loss(max_min_norm(attn_c[:,:,trainer.ID_c]), max_min_norm(attn1))
                            attn_loss_nag += torch.max(torch.tensor(.0).cuda(), m - nagative_loss)
                            attn_loss_S += nnf.mse_loss(attn_gt1[:,:,0].unsqueeze(-1) / 5,attn1)

                    loss = loss_rec + attn_loss_nag + attn_loss_S + attn_loss_c      
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_x = [x[1] for x in unet.named_parameters() if   (  'attn2.to_v.weight' in x[0] or 'attn2.to_k.weight' in x[0] )]
    
                        if args.Img_Emb_pro:
                            params_x += trainer.img_embedding.parameters()
                        if args.FT_KV:
                            params_x += trainer.FT_module.parameters()
                        if args.NA_pro:
                            params_x += trainer.Neg_Attn_module.parameters()
                        if args.FT_out:
                            params_x+= [x[1] for x in unet.named_parameters() if ('attn2.to_out' in x[0])]
            
                        params_to_clip = itertools.chain(params_x)
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    bar.update()

                logs = {"loss_rec":  format(loss_rec.detach().item(),'.4f'), "attn_loss_c":format(attn_loss_c.detach().item(),'.4f'),  "attn_loss_S":format(attn_loss_S.detach().item(),'.4f'),  "attn_loss_nag":format(attn_loss_nag.detach().item(),'.4f'), "lr": lr_scheduler.get_last_lr()[0]}
                bar.set_postfix(**logs)
                accelerator.log(logs)

            with torch.no_grad():   
                context = (con_f, uncon_f)  
                added_cond_kwargs_all = (uncond_added_cond_kwargs, cond_added_cond_kwargs)

                latent_cur = get_noise_pred(noise_scheduler, unet, latent_cur, t, False, context, added_cond_kwargs_all, trainer)      
        

            save_model_parameters(trainer, unet, args, i_n, Img_Emb_pro=args.Img_Emb_pro, NA_pro=NA_pro, FT_out=FT_out)
            accelerator.wait_for_everyone()
    
if __name__ == "__main__":
    args = parse_args()
    main(args)





