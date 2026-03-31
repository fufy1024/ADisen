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


def load_sdxl_tokenizers(pretrained_model_name_or_path: str, revision: Optional[str] = None):
    tokenizer_one = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=revision,
        use_fast=False,
    )
    return tokenizer_one, tokenizer_two


def load_sdxl_text_encoders(pretrained_model_name_or_path: str, revision: Optional[str] = None):
    cls_one = import_text_encoder_class(pretrained_model_name_or_path, revision, "text_encoder")
    cls_two = import_text_encoder_class(pretrained_model_name_or_path, revision, "text_encoder_2")
    text_encoder_one = cls_one.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    text_encoder_two = cls_two.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=revision,
    )
    return text_encoder_one, text_encoder_two



class MLP(nn.Module):
    def __init__(self,input_dim,hid_dim,out_dim):
        super(MLP, self).__init__()
        self.lin_1 = nn.Linear(input_dim, hid_dim, bias=False)
        self.lin_2 = nn.Linear(hid_dim, out_dim, bias=False)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)
        return x

class Neg_Attn(nn.Module):
    def __init__(self, out_channel=256, type_pro=1, type_pro_out='LN'): 
        super(Neg_Attn, self).__init__()
        shape = [4096]*4 + [1024]*60 +  [4096]*6 
        Len_layer = len(shape)
        print(f'loading NA_type_pro:{type_pro} and NA_type_pro_out {type_pro_out}')
        self.type_pro = type_pro    
        self.type_pro_out = type_pro_out
        self.out_channel = out_channel

        self.convert_input = nn.ModuleList([nn.Conv1d(shape[i_layer], out_channel, 1) for i_layer in range(Len_layer)])  
        
        if type_pro_out=='mapper':  
            self.output = nn.ModuleList([nn.Sequential(nn.Linear(shape[i_layer], self.out_channel),
                                         nn.LayerNorm(self.out_channel),
                                         nn.LeakyReLU(),
                                         nn.Linear(self.out_channel, self.out_channel),
                                         nn.LayerNorm(self.out_channel),
                                         nn.LeakyReLU(),
                                         nn.Linear(self.out_channel, shape[i_layer])) for i_layer in range(Len_layer)])
        elif type_pro_out=='conv': # here
            self.output =  nn.ModuleList([nn.Sequential( nn.Conv1d(shape[i_layer], self.out_channel, kernel_size=1), 
                    nn.Conv1d(self.out_channel, self.out_channel, kernel_size=1),
                    nn.BatchNorm1d(self.out_channel, affine=True),
                    nn.LeakyReLU(),
                    nn.Conv1d(self.out_channel, shape[i_layer] , kernel_size=1)) for i_layer in range(Len_layer)])
        elif type_pro_out=='mlp':
            self.output = nn.ModuleList([MLP(shape[i_layer],self.out_channel,shape[i_layer]) for i_layer in range(Len_layer)])
        elif type_pro_out=='LN':
            self.output = nn.ModuleList([nn.Linear(shape[i_layer],shape[i_layer]) for i_layer in range(Len_layer)])

        self.convert_output = nn.ModuleList([nn.Conv1d(out_channel,shape[i_layer], 3, padding=1) for i_layer in range(Len_layer)])

    def forward(self, S_value, S_attn, y, a = None, h = 4, block = 0, test=False, test_use_all=False):
        if test:
            S_attn  = S_attn.reshape(-1,h,*S_attn.shape[-2:])[0:1,:,:,:].reshape(-1,*S_attn.shape[-2:])
        if  a is None:
            a = torch.sigmoid(-y)
        else:    
            if  test:
                a = a[0].unsqueeze(0).repeat(h,1,1)
            else:     
                a = a.unsqueeze(1).repeat(1,h,1,1).reshape(-1,*a.shape[-2:])

        x = self.convert_input[block](S_attn) 
        x = a.mul(x) 

        S_attn_out1 = self.convert_output[block](x) 
        S_attn_out = S_attn_out1.softmax(dim=-1) 

        if test:
            S_attn_out=S_attn_out.unsqueeze(0).repeat(S_value.shape[0]//h,1,1,1).reshape(-1,*S_attn_out.shape[-2:])

        hidden_states_S = torch.matmul(S_attn_out, S_value)
        
        if self.type_pro_out in ['mapper','mlp','LN']:
            hidden_states_S = hidden_states_S.transpose(2,1)
            hidden_states_S = self.output[block](hidden_states_S)
            hidden_states_S = hidden_states_S.transpose(2,1)
        elif self.type_pro_out in ['conv']:
            hidden_states_S = self.output[block](hidden_states_S)

        hidden_states_S= rearrange(hidden_states_S, '(b h) n d -> b n (h d)', h=h)

        return hidden_states_S, S_attn_out1 



     
def create_A_Disen(unet, freeze_model, trainer=None, path = None, DDIM_pro = False, FT_out= False, FT_KV = False, DDIM_M=False, NA_pro=False, ):
    if trainer.is_train and freeze_model == "crossattn_kv" :
        print('unet --> requires_grad = True')
        for x in unet.named_parameters():
            if 'transformer_blocks' not in x[0]:
                    x[1].requires_grad = False
            elif not ('attn2.to_k' in x[0] or 'attn2.to_v' in x[0]): 
                    x[1].requires_grad = False
            if  ( ('attn2.to_v' in x[0]  or 'attn2.to_k' in x[0] )  or ('attn2.to_out' in x[0] and FT_out)):
                    x[1].requires_grad = True
                    print('requires_grad = True:', x[0])

    def change_checkpoint(model):
            for layer in model.children():
                if type(layer) == BasicTransformerBlock:
                    layer.checkpoint = False
                else:
                    change_checkpoint(layer)

    if trainer.is_train:
        change_checkpoint(unet)

    def inj_forward_crossattention_train(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        self.counter = 0 
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
            context = encoder_hidden_states
            if context is not None:
                context_c, context_S = context
                crossattn = True  
            else:
                context_c = hidden_states
                crossattn = False
                    

            query = self.to_q(hidden_states)        
            key = self.to_k(context_c)
            value = self.to_v(context_c)
            h = self.heads

            query, key, value = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (query, key, value ))
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
            attention_scores = attention_scores * self.scale

            if crossattn:
                key_S =  getattr(trainer.FT_module, trainer.layer_name_all[trainer.block//2].replace('_v_','_k_'))(context_S)
                value_S = getattr(trainer.FT_module, trainer.layer_name_all[trainer.block//2])(context_S)

                modifier_S = torch.ones_like(key_S)
                modifier_S[:, :1, :] = modifier_S[:, :1, :]*0.    
                key_S = modifier_S * key_S + (1-modifier_S) * key_S.detach()
                value_S = modifier_S * value_S + (1-modifier_S) * value_S.detach()

                key_S, value_S = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (key_S, value_S ))
                attention_scores_S = torch.matmul(query, key_S.transpose(-1, -2))
                attention_scores_S = attention_scores_S * self.scale

                res = int(np.sqrt(attention_scores_S.shape[1]))
                mask = F.interpolate(trainer.mask.unsqueeze(0), (Dim_pro, Dim_pro)).detach()
                mask = mask.squeeze(0).reshape(-1,Dim_pro*Dim_pro) 
            attention_probs = attention_scores.softmax(dim=-1)

            if crossattn :
                if not trainer.test :
                    attention_probs_mask = mask.unsqueeze(1).reshape(-1,Dim_pro*Dim_pro).unsqueeze(-1) 
                else:
                    attention_probs_mask = mask[0].unsqueeze(1).reshape(-1,Dim_pro*Dim_pro).unsqueeze(-1).repeat(mask.shape[0],1,1) 

                attention_probs_mask = None    
                if not trainer.test :
                    y = attention_probs[:,:,trainer.ID_c].mean(-1).mean(0).unsqueeze(-1).unsqueeze(0) 
                else:
                    y = attention_probs.reshape(-1,h,*attention_probs.shape[-2:])[:,:,:,trainer.ID_c].mean(-1).mean(1).unsqueeze(-1)
                       
                attention_probs_S1 = attention_scores_S.softmax(dim=-1) 
                hidden_states_S, attention_probs_S = trainer.Neg_Attn_module(value_S, attention_probs_S1, y, attention_probs_mask, h=h, block=trainer.block//2, test=trainer.test)
                    
            if trainer.attn_replace and trainer.test  and not trainer.uncond :
                if crossattn:
                    attention_probs = attn_replace(attention_probs, trainer , h)
                else:
                    attention_probs = attn_replace(attention_probs, trainer , h, attn_type_pro = 'self')
            
            if not trainer.test :
              if attention_scores.shape[1]==Dim_pro * Dim_pro:  
                if crossattn:
                    if trainer.attn_save['cross_attn'] != {}:
                        attention_probs_before = trainer.attn_save['cross_attn']['all']
                        block_num = trainer.attn_save['cross_attn']['block_num'] + 1
                    else:
                        attention_probs_before = 0
                        block_num = 1
                    trainer.attn_save['cross_attn'].update({'all':attention_probs_before + attention_probs,'block_num':block_num})
                
                    if trainer.attn_save['attn_S'] != {}:
                        attention_probs_before = trainer.attn_save['attn_S']['all']
                        block_num = trainer.attn_save['attn_S']['block_num'] + 1
                    else:
                        attention_probs_before = 0
                        block_num = 1
                    trainer.attn_save['attn_S'].update({'all':attention_probs_before + attention_probs_S,'block_num':block_num})
                else:
                    if trainer.attn_save['self_attn'] != {}:
                        attention_probs_before = trainer.attn_save['self_attn']['all']
                        block_num = trainer.attn_save['self_attn']['block_num'] + 1
                    else:
                        attention_probs_before = 0
                        block_num = 1
                    trainer.attn_save['self_attn'].update({'all':attention_probs_before + attention_probs,'block_num':block_num})

            trainer.block+=1
            hidden_states_c = torch.matmul(attention_probs, value)
            hidden_states_c = rearrange(hidden_states_c, '(b h) n d -> b n (h d)', h=h)

            if crossattn:                
                hidden_states = hidden_states_c  +  hidden_states_S 
            else:
                hidden_states = hidden_states_c    


            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)
        
            return hidden_states
        return forward

    def inj_forward_crossattention_ddim(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        self.counter = 0 
        def forward (hidden_states, encoder_hidden_states=None, attention_mask=None):
            context = encoder_hidden_states
            if context is not None:
                context_tensor = context
                crossattn = True  
            else:
                context_tensor = hidden_states
                crossattn = False
                    
            query = self.to_q(hidden_states)
            key = self.to_k(context_tensor)
            value = self.to_v(context_tensor)
            h = self.heads
            query, key, value = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (query, key, value ))
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
            attention_scores = attention_scores * self.scale
            attention_probs = attention_scores.softmax(dim=-1)
            
            if not trainer.test :
                if crossattn:
                    if attention_scores.shape[1] == Dim_pro * Dim_pro:
                        if trainer.attn_save['cross_attn'] != {}:
                            attention_probs_before = trainer.attn_save['cross_attn']['all']
                            block_num = trainer.attn_save['cross_attn']['block_num'] + 1
                        else:
                            attention_probs_before = 0
                            block_num = 1
                        trainer.attn_save['cross_attn'].update({'all':attention_probs_before + attention_probs,'block_num':block_num})
                else:
                    if attention_scores.shape[1] == Dim_pro * Dim_pro:
                        if trainer.attn_save['self_attn'] != {}:
                            attention_probs_before = trainer.attn_save['self_attn']['all']
                            block_num = trainer.attn_save['self_attn']['block_num'] + 1
                        else:
                            attention_probs_before = 0
                            block_num = 1
                        trainer.attn_save['self_attn'].update({'all':attention_probs_before + attention_probs,'block_num':block_num})

            if trainer.DDIM or trainer.DDIM_test: 
                if attention_scores.shape[1] == Dim_pro * Dim_pro:
                    if crossattn:

                        if trainer.attn_save['cross_sim'] != {}:
                            attention_scores_before = trainer.attn_save['cross_sim']['all']
                            block_num = trainer.attn_save['cross_sim']['block_num'] + 1
                        else:
                            attention_scores_before =0
                            block_num = 1
                        trainer.attn_save['cross_sim'].update({'all':attention_scores_before + attention_scores,'block_num':block_num})
                    else:
                        if trainer.attn_save['self_sim'] != {}:
                            attention_scores_before = trainer.attn_save['self_sim']['all']
                            block_num = trainer.attn_save['self_sim']['block_num'] + 1
                        else:
                            attention_scores_before =0
                            block_num = 1
                        trainer.attn_save['self_sim'].update({'all':attention_scores_before + attention_scores,'block_num':block_num})    

            trainer.block+=1
            hidden_states = torch.matmul(attention_probs, value)
            hidden_states= rearrange(hidden_states, '(b h) n d -> b n (h d)', h=h)
            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states
        
        return forward
        
    layer_name = {'down':[],'up':[],'mid':[]}
    def change_forward(model,name='',place_in_unet=None, up_first=False,block=0):
        if model.__class__.__name__ == 'Attention':
            if not DDIM_pro:
                model.forward = inj_forward_crossattention_train(model,place_in_unet)
            else:    
                model.forward = inj_forward_crossattention_ddim(model,place_in_unet)

            if  not DDIM_pro and FT_KV and 'attn2' in name:
                shape = model.to_k.weight.shape
                to_k_global = nn.Linear(shape[1], shape[0], bias=False).to(model.to_k.weight.dtype)
                to_k_global.weight.data = model.to_k.weight.data.clone()
                trainer.FT_module.add_module(f'{place_in_unet}_{name}_to_kc_global', to_k_global)

                shape1 = model.to_v.weight.shape
                to_v_global = nn.Linear(shape1[1], shape1[0], bias=False).to(model.to_v.weight.dtype)
                to_v_global.weight.data = model.to_v.weight.data.clone()
                trainer.FT_module.add_module(f'{place_in_unet}_{name}_to_v_global', to_v_global)
                trainer.FT_module.add_module(f'{place_in_unet}_{name}_to_k_global', to_k_global)
                layer_name[place_in_unet].append(f'{place_in_unet}_{name}_to_v_global')

                if FT_KV and path is None:
                    model.add_module('to_k_global', to_k_global.to(model.to_k.weight.dtype))
                    model.add_module('to_v_global', to_v_global.to(model.to_k.weight.dtype))
                elif FT_KV:
                    trainer.FT_module=torch.load(path)
                    trainer.FT_module.eval()
                    model.add_module('to_k_global', getattr(trainer.FT_module, f'{place_in_unet}_{name}_to_k_global').to(model.to_k.weight.dtype))
                    model.add_module('to_v_global', getattr(trainer.FT_module, f'{place_in_unet}_{name}_to_v_global').to(model.to_k.weight.dtype))
    
            for child_name, child in model.named_children():
                change_forward(
                    child,
                    name=f'{name}_{child_name}' if name else child_name,
                    place_in_unet=place_in_unet,
                )

    sub_nets = unet.named_children()
    for net in sub_nets:
        
        if "down" in net[0]:
            change_forward(net[1],'',"down")
        elif "up" in net[0]:
            change_forward(net[1],'',"up")
        elif "mid" in net[0]:
            change_forward(net[1],'',"mid")

    return unet, layer_name



class A_Disen_Trainer(nn.Module):
    def __init__(self,Img_Emb, NUM_DDIM_STEPS , NA_pro):
        super(A_Disen_Trainer, self).__init__()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.block = 0
        self.attn_save = {'cross_attn':{},'self_attn':{}, 'attn_S':{}}
        self.test = False
        self.DDIM_test = False
        self.FT_module = nn.Module()  

        if 'vgg-sum' in Img_Emb:
            self.img_embedding = nn.ModuleDict({
                'mapper0':nn.Sequential(nn.Linear(64, 1024), nn.LayerNorm(1024), nn.LeakyReLU(), nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.LeakyReLU(), nn.Linear(1024, 2048)),
                'mapper1':nn.Sequential(nn.Linear(128, 1024), nn.LayerNorm(1024), nn.LeakyReLU(), nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.LeakyReLU(), nn.Linear(1024, 2048)),
                'mapper2':nn.Sequential(nn.Linear(256, 1024), nn.LayerNorm(1024), nn.LeakyReLU(), nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.LeakyReLU(), nn.Linear(1024, 2048)),                    
                'mapper3':nn.Sequential(nn.Linear(512, 1024), nn.LayerNorm(1024), nn.LeakyReLU(), nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.LeakyReLU(), nn.Linear(1024, 2048))}).train().requires_grad_(False).to(self.device)
        elif 'vgg-linear' in Img_Emb:
            self.img_embedding = nn.ModuleDict({
                'L0':nn.Sequential(nn.Linear(224*224, 512), nn.LayerNorm(512), nn.LeakyReLU(), nn.Linear(512, 256), nn.LayerNorm(256), nn.LeakyReLU(), nn.Linear(256, 1)),
                'L1':nn.Sequential(nn.Linear(112*112, 512), nn.LayerNorm(512), nn.LeakyReLU(), nn.Linear(512, 256), nn.LayerNorm(256), nn.LeakyReLU(), nn.Linear(256, 1)),
                'L2':nn.Sequential(nn.Linear(56*56, 512), nn.LayerNorm(512), nn.LeakyReLU(), nn.Linear(512, 256), nn.LayerNorm(256), nn.LeakyReLU(), nn.Linear(256, 1)),
                'L3':nn.Sequential(nn.Linear(28*28, 512), nn.LayerNorm(512), nn.LeakyReLU(), nn.Linear(512, 256), nn.LayerNorm(256), nn.LeakyReLU(), nn.Linear(256, 1)),
                'mapper0':nn.Sequential(nn.Linear(64, 1024), nn.LayerNorm(1024), nn.LeakyReLU(), nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.LeakyReLU(), nn.Linear(1024, 2048)),
                'mapper1':nn.Sequential(nn.Linear(128, 1024), nn.LayerNorm(1024), nn.LeakyReLU(), nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.LeakyReLU(), nn.Linear(1024, 2048)),
                'mapper2':nn.Sequential(nn.Linear(256, 1024), nn.LayerNorm(1024), nn.LeakyReLU(), nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.LeakyReLU(), nn.Linear(1024, 2048)),
                'mapper3':nn.Sequential(nn.Linear(512, 1024), nn.LayerNorm(1024), nn.LeakyReLU(), nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.LeakyReLU(), nn.Linear(1024, 2048))}).train().requires_grad_(False).to(self.device)
        if NA_pro:
            self.Neg_Attn_module = Neg_Attn(out_channel=1024, type_pro=NA_pro)

        self.attn_replace = False
        self.i = -1
        self.num_time_replace = (0.2,0.8)
        self.num_block_replace = None
        self.DDIM_steps = 0 
        self.equalizer = []
        self.mapper, self.alphas = None, None

        self.num_time_weight = (0.3,0.6)
        self.N_weight =  (0.8, 0.8) 
        self.V_weight =  (0.8, 0.8) 
        self.equalizer_pos = []

        self.Img_Emb = Img_Emb
        self.outdir_other =''
        self.outdir_other_train =''
    
    def forward_embed_img(self,pro_Img_Emb = ''):
        img_emb = self.image_embeddings            
        if 'vgg-sum' in pro_Img_Emb or 'vgg-linear' in pro_Img_Emb:
            img_f_all=[]
            for i_block, img_f in enumerate(img_emb):  
                if 'vgg-sum' in pro_Img_Emb:
                    img_f=img_f.sum(-1).sum(-1) 
                else:
                    img_f = (getattr(self.img_embedding, f'L{i_block}')(img_f.reshape(*img_f.shape[:2],-1))).squeeze(-1)        
                img_block = getattr(self.img_embedding, f'mapper{i_block}')
                img_f_all.append(img_block(img_f))

            img_f_all = torch.stack(img_f_all, 1)    
            return img_f_all

    def load_pretrained_img(self, pretrained_embedding):
        for i, pre_embedding in enumerate(pretrained_embedding):
            for pre_emb, emb in zip(pre_embedding.values(), self.img_embedding[i].values()):
                self.copy_params_and_buffers(pre_emb, emb)     
    
    def named_params_and_buffers(self, module):
        assert isinstance(module, torch.nn.Module)
        return list(module.named_parameters()) + list(module.named_buffers())            

    def copy_params_and_buffers(self, src_vae, dst_vae, require_all=False):
        assert isinstance(src_vae, torch.nn.Module)
        assert isinstance(dst_vae, torch.nn.Module)
        vae_tensors = dict(self.named_params_and_buffers(src_vae))
        for name, tensor in self.named_params_and_buffers(dst_vae):
            assert (name in vae_tensors) or not require_all
            if name in vae_tensors and tensor.shape == vae_tensors[name].shape:
                try:
                    tensor.copy_(vae_tensors[name].detach()).requires_grad_(tensor.requires_grad)
                except Exception as e:
                    print(f'Error loading: {name} {vae_tensors[name].shape} {tensor.shape}')
                    raise e            
