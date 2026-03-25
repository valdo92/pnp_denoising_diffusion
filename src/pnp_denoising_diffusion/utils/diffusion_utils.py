"""Utils functions for the diffusion process"""
from box import Box
import torch
import numpy as np
from pnp_denoising_diffusion.utils import utils_model 

def get_params_diffusion(config):
    """return the params for the diffusion"""
    skip = config.num_train_timesteps // config.iter_num
    betas = np.linspace(config.beta_start, config.beta_end, config.num_train_timesteps, dtype=np.float32)
    betas = torch.from_numpy(betas).to(config.device)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)  
    t_start = config.num_train_timesteps - 1 
    params = {
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_1m_alphas_cumprod": sqrt_1m_alphas_cumprod,
        "reduced_alpha_cumprod": reduced_alpha_cumprod,
        "t_start": t_start       
    }
    return Box(params)


def transfer_model_shape(image, image_transformed, mask, device):
    """Transfer the images to the model shape"""
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device)
    image = image * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    
    mask = torch.from_numpy(mask).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    image_transformed = torch.from_numpy(
        image_transformed
        ).permute(2, 0, 1).float().unsqueeze(0).to(device)
    image_transformed = image_transformed * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    return image, image_transformed, mask


def initialize_x(params, config, y):
    """Change the transformed image y in initial state for diffusion"""
    t_y = utils_model.find_nearest(
        params.reduced_alpha_cumprod, 2 * config.noise_level_img
        )
    sqrt_alpha_effective = params.sqrt_alphas_cumprod[params.t_start] \
        / params.sqrt_alphas_cumprod[t_y]
    x = sqrt_alpha_effective * y + torch.sqrt(params.sqrt_1m_alphas_cumprod[params.t_start]**2 - \
                    sqrt_alpha_effective**2 * params.sqrt_1m_alphas_cumprod[t_y]**2) \
                        * torch.randn_like(y)
    return x