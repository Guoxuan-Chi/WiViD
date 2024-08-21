import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from unet import UNet


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
        self, model, img_size, img_channels, betas=[]
    ):
        super().__init__()
        self.model      = model
        
        self.step               = 0

        self.img_size       = img_size
        self.img_channels   = img_channels

        self.num_timesteps  = len(betas)

        alphas              = 1.0 - betas
        alphas_cumprod      = np.cumprod(alphas)

        # 转换成torch.tensor来处理
        to_torch = partial(torch.tensor, dtype=torch.float32)

        # betas             [0.0001, 0.00011992, 0.00013984 ... , 0.02]
        self.register_buffer("betas", to_torch(betas))
        # alphas            [0.9999, 0.99988008, 0.99986016 ... , 0.98]
        self.register_buffer("alphas", to_torch(alphas))
        # alphas_cumprod    [9.99900000e-01, 9.99780092e-01, 9.99640283e-01 ... , 4.03582977e-05]
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        # sqrt(alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        # sqrt(1 - alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        # sqrt(1 / alphas)
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))


    @torch.no_grad()
    def remove_noise(self, x, t, rgb_condition, csi_condition):
        return (
            (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, rgb_condition, csi_condition)) *
            extract(self.reciprocal_sqrt_alphas, t, x.shape)
        )

    @torch.no_grad()
    def sample(self, batch_size, device, rgb_condition, csi_condition):
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, rgb_condition, csi_condition)
        return x.cpu().detach()
    
    @torch.no_grad()
    def predict_noise(self, x, t, rgb_condition, csi_condition):
        noise_predicted = self.model(x, t, rgb_condition, csi_condition)
        return noise_predicted


    def perturb_x(self, x, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t,  x.shape) * x +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )   

    def get_losses(self, x, t, rgb_condition, csi_condition):
        # x, noise [batch_size, 3, H, W]
        noise           = torch.randn_like(x)
        perturbed_x     = self.perturb_x(x, t, noise)
        estimated_noise = self.model(perturbed_x, t, rgb_condition, csi_condition)
        loss = F.mse_loss(estimated_noise, noise)
        return loss

    def forward(self, x, rgb_condition, csi_condition):
        b, c, h, w  = x.shape
        device      = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self.get_losses(x, t, rgb_condition, csi_condition)

def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)