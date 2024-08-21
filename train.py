import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from gaussian_diffusion import GaussianDiffusion, generate_linear_schedule
from unet import UNet
from train_dataset import DiffusionDataset
from utils.utils import  set_optimizer_lr, get_lr_scheduler
from utils.utils_fit import fit_one_epoch
import os

if __name__ == "__main__":

    device          = torch.device('cuda:0')

    #   betas相关参数
    num_timesteps   = 50
    schedule_low    = 1e-4
    schedule_high   = 0.02

    input_shape     = (160, 90)

    Epoch           = 100
    batch_size      = 16
    
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.001
    
    momentum            = 0.9
    lr_decay_type       = "cos"

    save_period         = 10
    save_dir            = "PATH"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    betas = generate_linear_schedule(
        num_timesteps,
        schedule_low * 1000 / num_timesteps,
        schedule_high * 1000 / num_timesteps,
    )
    
    diffusion_model = GaussianDiffusion(UNet(), input_shape, img_channels = 1, betas=betas)
    cudnn.benchmark = True
    diffusion_model = diffusion_model.train()
    diffusion_model = diffusion_model.to(device)

    optimizer = optim.AdamW(diffusion_model.parameters(), lr=Init_lr, betas=(momentum, 0.999))
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, Epoch)

    train_dataset   = DiffusionDataset()
    dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=1, pin_memory=True, drop_last=True)

    for epoch in range(Epoch):   
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(diffusion_model, optimizer, 
                    epoch, train_dataset, dataloader, Epoch, device, save_period, save_dir)