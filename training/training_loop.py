﻿import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
import warnings
warnings.filterwarnings("ignore")
from colorama import init
from colorama import Fore, Style
from icecream import ic
init(autoreset=True)
from etaprogress.progress import ProgressBar
import sys
import matplotlib.pyplot as plt
from evaluate import save_gen, create_folders

from metrics.evaluation.data import PrecomputedInpaintingResultsDataset
from metrics.evaluation.evaluator import InpaintingEvaluator
from metrics.evaluation.losses.base_loss import FIDScore
from metrics.evaluation.utils import load_yaml

from torch.nn.parameter import Parameter
#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set):
    grid_indices = [0, 0, 0, 0]
    # grid_indices = [i*199 for i in grid_indices]
    # Load data.
    noisy_image, denoised_image, tuning_param, labels = zip(*[training_set[i] for i in grid_indices])
    return np.stack(noisy_image), np.stack(denoised_image), np.stack(tuning_param), np.stack(labels)


#----------------------------------------------------------------------------

def save_image_grid(imgs, fname, label, drange):
    lo, hi = drange
    imgs = np.asarray(imgs, dtype=np.float32)
    imgs = (imgs - lo) * (255 / (hi - lo))
    imgs = np.rint(imgs).clip(0, 255).astype(np.uint8)

    _N, C, H, W = imgs.shape
    imgs = imgs.transpose(0, 2, 3, 1) # HWC

    assert C in [1, 3]
    
    for i in range(_N):
        if C == 1:
            PIL.Image.fromarray(imgs[i][:, :, 0], 'L').save("{}_{}_{}.png".format(fname, i, label))
        if C == 3:
            PIL.Image.fromarray(imgs[i], 'RGB').save("{}_{}_{}.png".format(fname, i, label))

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    eval_img_data           = None,     # Evaluation Image data
    resolution              = None,      # Resolution of evaluation image
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    input_param_dims         = 512,
    is_recommand            = False,
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = None,     # EMA ramp-up coefficient.
    G_reg_interval          = None,        # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 0.1,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkls              = None,     # Network pickle to resume training from.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    allow_tf32              = False,    # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
):
    no_updates_times = 0
    best_loss_Gmain = 1e8
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    eval_config = load_yaml('metrics/configs/eval2_gpu.yaml')

    # Load training set.
    if rank == 0:
        print(Fore.GREEN + 'Loading training set...')
        
    if is_recommand:
        training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
        training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
        training_loader = torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs)
        training_set_iterator = iter(training_loader)
        full_dataset=training_set
    else:
        full_dataset = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
        num_train = int(len(full_dataset) * 0.8)
        num_val = len(full_dataset) - num_train

        # Split the data set into training set and validation set
        training_set, val_set = torch.utils.data.random_split(full_dataset, [num_train, num_val])
        
        training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
        training_loader = torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs)
        training_set_iterator = iter(training_loader)
        
        val_set_sampler = misc.InfiniteSampler(dataset=val_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
        val_loader = torch.utils.data.DataLoader(dataset=val_set, sampler=val_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs)
        val_set_iterator = iter(val_loader)
    
    
    if rank == 0:
        print()
        print(Fore.GREEN + 'Num images: ', len(training_set))
        print(Fore.GREEN + 'Image shape:', full_dataset.image_shape)
        print(Fore.GREEN + 'Label shape:', full_dataset.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=full_dataset.label_dim, img_resolution=full_dataset.resolution, img_channels=full_dataset.num_channels)

    Gs = []
    # Resume from existing pickle.
    if (resume_pkls is not None) and (rank == 0):
        for resume_pkl, input_param_dim in zip(resume_pkls, input_param_dims):
            print(f'Resuming from "{resume_pkl}"')
            G_kwargs.input_param_dim = input_param_dim
            G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
            
            with dnnlib.util.open_url(resume_pkl) as f:
                resume_data = legacy.load_network_pkl(f)
                
            # perform model surgery
            with torch.no_grad():
                getattr(resume_data['G'].encoder, f'b{resolution}').fromrgb.weight=Parameter(getattr(resume_data['G'].encoder, f'b{resolution}').fromrgb.weight[:,:3,...])
                # resume_data['G'].encoder.b256.fromrgb.weight=Parameter(resume_data['G'].encoder.b256.fromrgb.weight[:,:3,...])
                resume_data['G'].mapping.fc0.weight=Parameter(resume_data['G'].mapping.fc0.weight[:, :input_param_dim+2]) # param dim contains position(2 more dim)
                resume_data['G'].tuning_fn.fc1.weight=Parameter(resume_data['G'].tuning_fn.fc1.weight[:input_param_dim, :input_param_dim]) # param dim contains position(2 more dim)

            for name, module in [('G', G)]:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
            
            Gs.append(G)
    assert is_recommand or  len(Gs) == 1
    # Print network parameters
    if rank == 0:
        for G in Gs:
            netG_params = sum(p.numel() for p in G.parameters())
            print(Fore.GREEN +"Generator Params: {} M".format(netG_params/1e6))

    # Distribute across GPUs.
    if rank == 0:
        print(Fore.CYAN + f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    for i, G in enumerate(Gs):
        for name, module in [(f'G{i}_encoder', G.encoder), (f'G{i}_mapping', G.mapping), (f'G{i}_synthesis', G.synthesis), (f'G{i}_tuning_fn', G.tuning_fn)]:
            if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
                module.requires_grad_(True)
                module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False, find_unused_parameters=True)
                module.requires_grad_(False)
            if name is not None:
                ddp_modules[name] = module

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, Gs=Gs, **loss_kwargs) # subclass of training.losses.loss.Loss
    phases = []
    for i, G in enumerate(Gs):
        if is_recommand:
            for name, module, opt_kwargs, reg_interval in [(f'G{i}_tuning_fn', G.tuning_fn, G_opt_kwargs, G_reg_interval)]:
                if reg_interval is None:
                    opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
                    phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
                else: # Lazy regularization.
                    mb_ratio = reg_interval / (reg_interval + 1)
                    opt_kwargs = dnnlib.EasyDict(opt_kwargs)
                    opt_kwargs.lr = opt_kwargs.lr * mb_ratio
                    opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
                    opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
                    phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
                    phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
        else:
            for name, module, opt_kwargs, reg_interval in [(f'G{i}', G, G_opt_kwargs, G_reg_interval)]: #, ('D', D, D_opt_kwargs, D_reg_interval)]:
                if reg_interval is None:
                    opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
                    phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
                else: # Lazy regularization.
                    mb_ratio = reg_interval / (reg_interval + 1)
                    opt_kwargs = dnnlib.EasyDict(opt_kwargs)
                    opt_kwargs.lr = opt_kwargs.lr * mb_ratio
                    opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
                    opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
                    phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
                    phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        noisy_image, denoised_image, tuning_param, labels = setup_snapshot_image_grid(training_set=training_set)
        print(noisy_image.shape, denoised_image.shape, tuning_param.shape)
        sample_noisy_img = (torch.from_numpy(noisy_image).to(torch.float32) / 127.5 - 1).to(device)
        sample_denoised_img = (torch.from_numpy(denoised_image).to(torch.float32) / 127.5 - 1).to(device)
        sample_tuning_param = torch.from_numpy(tuning_param).to(torch.float32).to(device)
        
        sample_noisy_img = sample_noisy_img.split(batch_gpu)
        sample_denoised_img = sample_denoised_img.split(batch_gpu)
        sample_tuning_param = sample_tuning_param.split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(torch.float32).to(device).split(batch_gpu)
        
        pred_images = []
        for noisy_img, tuning_param, c in zip(sample_noisy_img, sample_tuning_param, grid_c):
            img = noisy_img
            dim = 0
            for i, G in enumerate(Gs):
                z = torch.cat((tuning_param[:, dim:dim+G.input_param_dim], tuning_param[:, -2:]), 1)
                img = G(img=img, z=z, c=c, noise_mode='const')
                dim += G.input_param_dim
                pred_images.append(img.cpu())
                
        pred_images = torch.cat(pred_images)
        # pred_images = torch.cat([G(img=noisy_img, z=tuning_param, c=c, noise_mode='const').cpu() for noisy_img, tuning_param, c in zip(sample_noisy_img, sample_tuning_param, grid_c)])
        save_image_grid(noisy_image, os.path.join(run_dir, 'run_init'), label='NOISY', drange=[0,255])
        save_image_grid(denoised_image, os.path.join(run_dir, 'run_init'),  label='GT', drange=[0,255])
        save_image_grid(pred_images.detach().numpy(), os.path.join(run_dir, 'run_init'),  label='PRED', drange=[-1,1])
        
        if is_recommand:
            print("sample_tuning_param", sample_tuning_param)
    
    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(Fore.GREEN + Style.BRIGHT + f'Training for {total_kimg} kimg...')
        print()
        total = total_kimg * 1000
        bar = ProgressBar(total, max_width=80)

    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    
    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_noisy_imgs, phase_denoised_imgs, phase_tuning_param, phase_real_cs = next(training_set_iterator)
            phase_noisy_imgs = (phase_noisy_imgs.to(device).to(torch.float32) / 127.5 - 1)
            phase_denoised_imgs = (phase_denoised_imgs.to(device).to(torch.float32) / 127.5 - 1)
            phase_noisy_imgs = phase_noisy_imgs.split(batch_gpu)
            phase_denoised_imgs = phase_denoised_imgs.split(batch_gpu)
            phase_tuning_param = phase_tuning_param.to(device).split(batch_gpu)
            phase_real_c = phase_real_cs.to(device).split(batch_gpu)
            all_gen_c = [full_dataset.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_c in zip(phases, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for round_idx, (noisy_img, denoised_img, tuning_param, real_c, gen_c) in enumerate(zip(phase_noisy_imgs, phase_denoised_imgs, phase_tuning_param, phase_real_c, phase_gen_c)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase.interval
                loss.accumulate_gradients(phase=phase.name, noisy_img=noisy_img, denoised_img=denoised_img, tuning_param=tuning_param, real_c=real_c, gen_c=gen_c, sync=sync, gain=gain)

            # Update weights.
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param_tuned in phase.module.parameters():
                    if param_tuned.grad is not None:
                        misc.nan_to_num(param_tuned.grad, nan=0, posinf=1e5, neginf=-1e5, out=param_tuned.grad)
                phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        if rank == 0:
            bar.numerator = cur_nimg
            print(bar, end='\r')

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000 * image_snapshot_ticks):
            continue
        
        # validate
        if not is_recommand:
            for val_num in range(20):
                # Fetch val data.
                with torch.autograd.profiler.record_function('data_fetch'):
                    phase_noisy_imgs, phase_denoised_imgs, phase_tuning_param, phase_real_cs = next(val_set_iterator)
                    phase_noisy_imgs = (phase_noisy_imgs.to(device).to(torch.float32) / 127.5 - 1)
                    phase_denoised_imgs = (phase_denoised_imgs.to(device).to(torch.float32) / 127.5 - 1)
                    phase_noisy_imgs = phase_noisy_imgs.split(batch_gpu)
                    phase_denoised_imgs = phase_denoised_imgs.split(batch_gpu)
                    phase_tuning_param = phase_tuning_param.to(device).split(batch_gpu)
                    phase_real_c = phase_real_cs.to(device).split(batch_gpu)
                    all_gen_c = [full_dataset.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
                    all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
                    all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

                # Execute val phases.
                for phase, phase_gen_c in zip(phases, all_gen_c):
                    phase.module.requires_grad_(False)

                    # Accumulate gradients over multiple rounds.
                    for round_idx, (noisy_img, denoised_img, tuning_param, real_c, gen_c) in enumerate(zip(phase_noisy_imgs, phase_denoised_imgs, phase_tuning_param, phase_real_c, phase_gen_c)):
                        sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                        gain = phase.interval
                        loss.accumulate_gradients(phase=phase.name, noisy_img=noisy_img, denoised_img=denoised_img, tuning_param=tuning_param, real_c=real_c, gen_c=gen_c, sync=sync, gain=gain, state="val")

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem GB {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem GB {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(Fore.CYAN + Style.BRIGHT + ' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print(Fore.RED + 'Aborting...')
            
        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        if rank == 0:
            losses = []
            for key in stats_dict.keys():
                if 'Loss/D' in key or 'Loss/G' in key:
                    losses += [f"{key}: {(stats_dict[key]['mean']):<.4f}"]
            print(Fore.MAGENTA + Style.BRIGHT + ' '.join(losses))
            
        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        
        if (rank == 0):
            if not is_recommand:
                print('train:', stats_dict["Loss/G/main_loss"])
                print('val:', stats_dict["Loss/val/pl_loss"])
                if (stats_dict["Loss/val/pl_loss"]['mean']<best_loss_Gmain): # or (stats_dict["Loss/G/rec_loss"]['mean']<best_rec_loss):
                    # Save image snapshot.
                    if cur_tick>0: best_loss_Gmain = stats_dict["Loss/val/pl_loss"]['mean']
                    no_updates_times=0
                    pred_images = []
                    for noisy_img, tuning_param, c in zip(sample_noisy_img, sample_tuning_param, grid_c):
                        img = noisy_img
                        dim = 0
                        for i, G in enumerate(Gs):
                            z = torch.cat((tuning_param[:, dim:dim+G.input_param_dim], tuning_param[:, -2:]), 1)
                            img = G(img=img, z=z, c=c, noise_mode='const')
                            dim += G.input_param_dim
                            pred_images.append(img.cpu())
                            
                    pred_images = torch.cat(pred_images)
                    # pred_images = torch.cat([G(img=noisy_img, z=tuning_param, c=c, noise_mode='const').cpu() for noisy_img, tuning_param, c in zip(sample_noisy_img, sample_tuning_param, grid_c)])
                    save_image_grid(pred_images.detach().numpy(), os.path.join(run_dir, f'run_{cur_nimg//1000:06d}'),  label='PRED', drange=[-1,1])

                    print('save best model', best_loss_Gmain)
                    snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
                    
                    assert len(Gs) == 1
                    for name, module in [('G', Gs[0])]:
                        if module is not None:
                            if num_gpus > 1:
                                misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                            module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                        snapshot_data[name] = module
                        del module # conserve memory
                    snapshot_pkl = os.path.join(run_dir, f'Model.pkl')
                
                    with open(snapshot_pkl, 'wb') as f:
                        pickle.dump(snapshot_data, f)
                else:
                    no_updates_times += 1
            else:
                if (stats_dict["Loss/G/main_loss"]['mean']<best_loss_Gmain):
                    if cur_tick>0: best_loss_Gmain = stats_dict["Loss/G/main_loss"]['mean']
                    pred_images = []
                    for noisy_img, tuning_param, c in zip(sample_noisy_img, sample_tuning_param, grid_c):
                        img = noisy_img
                        dim = 0
                        for i, G in enumerate(Gs):
                            z = torch.cat((tuning_param[:, dim:dim+G.input_param_dim], tuning_param[:, -2:]), 1)
                            img = G(img=img, z=z, c=c, noise_mode='const')
                            dim += G.input_param_dim
                            pred_images.append(img.cpu())
                            
                    pred_images = torch.cat(pred_images)
                    # pred_images = torch.cat([G(img=noisy_img, z=tuning_param, c=c, noise_mode='const').cpu() for noisy_img, tuning_param, c in zip(sample_noisy_img, sample_tuning_param, grid_c)])
                    save_image_grid(pred_images.detach().numpy(), os.path.join(run_dir, f'run_{cur_nimg:06d}'), label='PRED', drange=[-1,1])
                    print('save best txt', best_loss_Gmain)
                    with open(os.path.join(run_dir, run_dir.split("/")[-1]+"_"+str(cur_tick)+'.txt'), 'w') as f:
                        dim = 0
                        param_tuned = []
                        for G in Gs:
                            param_tuned.append(G.tuning_fn(sample_tuning_param[0][:, dim:dim+G.input_param_dim]).cpu())
                            dim += G.input_param_dim
                        param_tuned = np.concatenate(param_tuned, axis=1)
                        param_tuned = [round(num, 4) for num in param_tuned[0].tolist()]
                        f.write(str(param_tuned))
                        
                    if len(Gs) == 1:
                        print('save best model', best_loss_Gmain)
                        snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
                        for name, module in [('G', Gs[0])]:
                            if module is not None:
                                if num_gpus > 1:
                                    misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                                module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                            snapshot_data[name] = module
                            del module # conserve memory
                        snapshot_pkl = os.path.join(run_dir, f'Model.pkl')
        
        del snapshot_data # conserve memory

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if rank == 0:
            sys.stdout.flush()
        if done:
            break
        if no_updates_times >= 4:
            print('no updates, so break')
            break

    # Done.
    if rank == 0:
        print()
        print(Fore.YELLOW + 'Exiting...')

#----------------------------------------------------------------------------
