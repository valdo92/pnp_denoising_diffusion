"""Utils functions for the diffusion process"""
from box import Box
import torch
import numpy as np
from pnp_denoising_diffusion.utils import utils_model 
from pnp_denoising_diffusion.guided_diffusion.script_util import create_model_and_diffusion
import lpips
from pnp_denoising_diffusion.utils.score import calculate_psnr, calculate_fid_process
from pnp_denoising_diffusion.utils.read_image import read_and_save
import os
import csv

def get_params_diffusion(config):
    """return the params for the diffusion"""
    betas = np.linspace(
        config.beta_start, config.beta_end, config.num_train_timesteps, dtype=np.float32
        )
    betas = torch.from_numpy(betas).to(config.device)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)  
    t_start = config.num_train_timesteps - 1 
    
        # create sequence of timestep for sampling
    seq = np.sqrt(np.linspace(0, config.num_train_timesteps**2, config.iter_num))
    seq = [int(s) for s in list(seq)]
    seq[-1] = seq[-1] - 1
    progress_seq = seq[::(len(seq)//10)]
    progress_seq.append(seq[-1])

    sigmas = []
    sigma_ks = []
    rhos = []
    for i in range(config.num_train_timesteps):
        sigmas.append(reduced_alpha_cumprod[config.num_train_timesteps-1-i])
        sigma_ks.append((sqrt_1m_alphas_cumprod[i]/sqrt_alphas_cumprod[i]))
        rhos.append(config.lambda_*(config.sigma**2)/(sigma_ks[i]**2))            
    rhos = torch.tensor(rhos).to(config.device)
    sigmas = torch.tensor(sigmas).to(config.device)
    sigma_ks = torch.tensor(sigma_ks).to(config.device)
    params = {
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_1m_alphas_cumprod": sqrt_1m_alphas_cumprod,
        "reduced_alpha_cumprod": reduced_alpha_cumprod,
        "t_start": t_start,
        "sigmas": sigmas,
        "rhos": rhos,
        "sigma_ks": sigma_ks,
        "seq": seq,
        "progress_seq": progress_seq
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


def load_diffusion_model(config):
    """Load the diffusion model from open ai"""
    model, _ = create_model_and_diffusion(**config.guided_diffusion)
    model.load_state_dict(torch.load(config.model_path, map_location="cpu"))
    model = model.to(config.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def run_evaluation(x_final, image_gt, config, device, fid_scorer=None):
    """
    Calcule les métriques (PSNR, LPIPS) pour une image, 
    et accumule les features pour le FID global si fid_scorer est fourni.
    """

    x_0_output = (x_final / 2 + 0.5).clamp(0, 1)
    
    img_est_uint8 = (x_0_output * 255).to(torch.uint8)
    img_gt_uint8 = ((image_gt / 2 + 0.5) * 255).clamp(0, 255).to(torch.uint8)

    img_psnr_gt = np.transpose(img_gt_uint8.squeeze(0).cpu().numpy(), (1, 2, 0))
    img_psnr_est = np.transpose(img_est_uint8.squeeze(0).cpu().numpy(), (1, 2, 0))
    psnr_score = calculate_psnr(img_psnr_gt, img_psnr_est)

    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    lpips_score = loss_fn_vgg(x_final.detach(), image_gt).item()
    
    del loss_fn_vgg

    # Accumulate features for FID across the batch
    if fid_scorer is not None:
        fid_scorer.update(img_gt_uint8, real=True)
        fid_scorer.update(img_est_uint8, real=False)
    
    return {
        "psnr": psnr_score,
        "lpips": lpips_score
    }
