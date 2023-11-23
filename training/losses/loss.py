﻿import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from icecream import ic
# from .high_receptive_pl import HRFPL
from lpips import LPIPS, im2tensor

import os

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_encoder, G_mapping, G_synthesis, G_tuning_fn, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2,  is_recommand=False):
        super().__init__()
        self.device = device
        self.G_encoder = G_encoder
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.G_tuning_fn = G_tuning_fn
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        # self.run_hrfpl = HRFPL(weight=5, weights_path=os.getcwd())
        self.LPIPS = LPIPS(net='alex',version='0.1').cuda()
        self.is_recommand = is_recommand

    def run_G(self, r_img, z, c, sync):
        with misc.ddp_sync(self.G_encoder, sync):
            x_global, feats = self.G_encoder(r_img, c)
        with misc.ddp_sync(self.G_mapping, sync):
            if self.is_recommand:
                z_tune = self.G_tuning_fn(z[:, :-2].to(torch.float32)) ## 
                z = torch.cat((z_tune, z[:, -2:]), 1)
            ws = self.G_mapping(z, c)
            # if self.style_mixing_prob > 0:
            #     with torch.autograd.profiler.record_function('style_mixing'):
            #         cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            #         cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
            #         ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(x_global, None, feats, ws)
        return img, ws

    def run_D(self, img, c, sync):
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits
    

    def accumulate_gradients(self, phase, noisy_img, denoised_img, tuning_param, real_c, gen_c, sync, gain):
        # print('phase: ', phase)
        if self.is_recommand:
            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                # g_inputs = torch.cat([0.5 - mask, erased_img], dim=1)
                gen_img, _ = self.run_G(noisy_img, tuning_param, gen_c, sync=sync) # May get synced by Gpl.
                # gen_img = gen_img * mask + real_img * (1 - mask)
                # loss_rec = 10 * torch.nn.functional.l1_loss(gen_img, denoised_img)
                loss_pl = self.LPIPS.forward(gen_img, denoised_img).mean()
                
                if self.augment_pipe is not None:
                    gen_img = self.augment_pipe(gen_img)
                # d_inputs = torch.cat([0.5 - mask, gen_img], dim=1)
                # gen_logits = self.run_D(gen_img, gen_c, sync=False)
                # loss_G =  torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                
                # loss_Gmain = loss_G.mean() + loss_rec + loss_pl
                loss_Gmain = loss_pl
                # loss_Gmain = loss_pl
                # training_stats.report('Loss/G/loss', loss_G)
                # training_stats.report('Loss/G/rec_loss', loss_rec)
                training_stats.report('Loss/G/main_loss', loss_Gmain)
                training_stats.report('Loss/G/pl_loss', loss_pl)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mul(gain).backward()
                
        else:
        
            assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
            do_Gmain = (phase in ['Gmain', 'Gboth'])
            do_Dmain = (phase in ['Dmain', 'Dboth'])
            do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)
            
            # Gmain: Maximize logits for generated images.
            if do_Gmain:
                with torch.autograd.profiler.record_function('Gmain_forward'):
                    # g_inputs = torch.cat([0.5 - mask, erased_img], dim=1)
                    gen_img, _ = self.run_G(noisy_img, tuning_param, gen_c, sync=sync) # May get synced by Gpl.
                    # gen_img = gen_img * mask + real_img * (1 - mask)
                    loss_rec = 10 * torch.nn.functional.l1_loss(gen_img, denoised_img)
                    loss_pl = self.LPIPS.forward(gen_img, denoised_img).mean()
                    
                    # if self.augment_pipe is not None:
                    #     gen_img = self.augment_pipe(gen_img)
                    # d_inputs = torch.cat([0.5 - mask, gen_img], dim=1)
                    # gen_logits = self.run_D(gen_img, gen_c, sync=False)
                    # loss_G =  torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                    
                    # loss_Gmain = loss_G.mean() + loss_rec + loss_pl
                    loss_Gmain = loss_rec + loss_pl
                    # training_stats.report('Loss/G/loss', loss_G)
                    training_stats.report('Loss/G/rec_loss', loss_rec)
                    training_stats.report('Loss/G/main_loss', loss_Gmain)
                    training_stats.report('Loss/G/pl_loss', loss_pl)
                with torch.autograd.profiler.record_function('Gmain_backward'):
                    loss_Gmain.mul(gain).backward()

#----------------------------------------------------------------------------
