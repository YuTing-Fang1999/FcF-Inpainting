import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from icecream import ic
# from .high_receptive_pl import HRFPL
from lpips import LPIPS, im2tensor

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, Gs, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2,  is_recommand=False):
        super().__init__()
        self.device = device
        self.Gs = Gs
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

    def run_G(self, img, tuning_param, c, sync):
        dim = 0
        for i, G in enumerate(self.Gs):
            z = torch.cat((tuning_param[:, dim:dim+G.input_param_dim], tuning_param[:, -2:]), 1)
            with misc.ddp_sync(G.encoder, sync):
                x_global, feats = G.encoder(img, c)
            with misc.ddp_sync(G.mapping, sync):
                if self.is_recommand:
                    z_tune = G.tuning_fn(z[:, :-2].to(torch.float32))
                    z = torch.cat((z_tune, z[:, -2:]), 1)
                ws = G.mapping(z, c)
                
            with misc.ddp_sync(G.synthesis, sync):
                img = G.synthesis(x_global, None, feats, ws)
                
            dim += G.input_param_dim
                
        return img, ws
    

    def accumulate_gradients(self, phase, noisy_img, denoised_img, tuning_param, real_c, gen_c, sync, gain, state="train"):
        if state == "train":
            # print('phase: ', phase)
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _ = self.run_G(noisy_img, tuning_param, gen_c, sync=sync) # May get synced by Gpl.
                loss_pl = self.LPIPS.forward(gen_img, denoised_img).mean()
                if not self.is_recommand:
                    loss_rec = 10 * torch.nn.functional.l1_loss(gen_img, denoised_img)
                    loss_Gmain = loss_pl + loss_rec
                    training_stats.report('Loss/G/rec_loss', loss_rec)
                else:
                    loss_Gmain = loss_pl
                    
                # print('train loss_Gmain: ', loss_pl)
                training_stats.report('Loss/G/pl_loss', loss_pl)
                training_stats.report('Loss/G/main_loss', loss_Gmain)
                
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mul(gain).backward()
                
        elif state == "val":
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _ = self.run_G(noisy_img, tuning_param, gen_c, sync=sync) # May get synced by Gpl.
                loss_pl = self.LPIPS.forward(gen_img, denoised_img).mean()
                # print('val loss_Gmain: ', loss_pl)
                
                training_stats.report('Loss/val/pl_loss', loss_pl)

#----------------------------------------------------------------------------
