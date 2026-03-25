"""The main code. You can modify config.yaml to change the parameters to run the code"""

import torch
import numpy as np
import os
from pnp_denoising_diffusion.utils.score import calculate_psnr, calculate_fid_process   
import lpips
from pnp_denoising_diffusion.utils.utils import load_config, set_seed
from pnp_denoising_diffusion.utils import utils_model 
from pnp_denoising_diffusion.utils.load_image import load_image
from pnp_denoising_diffusion.utils.read_image import read_and_save
from pnp_denoising_diffusion.utils.plot_image import imshow
from pnp_denoising_diffusion.transform import transform_image 
from pnp_denoising_diffusion.diffusion import simple_diffusion_step, single_diffpir_step
from pnp_denoising_diffusion.utils.diffusion_utils import (
    get_params_diffusion, transfer_model_shape, initialize_x,
    load_diffusion_model, run_evaluation
    )


if __name__ == "__main__":
    print("⏳ Loading config, parameters and images...")
    config = load_config("config.yaml")
    if os.path.exists(config.name_folder_result):
        raise FileExistsError(f"🛑 : The folder '{config.name_folder_result}' exist, change it in config or delete the folder")
    os.makedirs(config.name_folder_result)
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    params = get_params_diffusion(config)
    
    image = load_image(config.path_to_image) # [256, 256, 3]
    image_transformed, mask = transform_image(image, config)
    image, image_transformed, mask = transfer_model_shape(
        image, image_transformed, mask, config.device
        )
    y = image_transformed
    x = initialize_x(params, config, y)

    print("⏳ Loading the model and the weights...")
    model = load_diffusion_model(config)

    imshow(x, title='init random', save_path=f"results/{config.name_folder_result}/init_random.png", show=False)
    imshow(y, title='image transformé', save_path=f"results/{config.name_folder_result}/Image_intiale_Transformée.png", show=False)


    print(f"--- Reverse Diffusion --- {len(params.seq)} ")
    progress_img = []
    # reverse diffusion for one image from random noise
    for i in range(len(params.seq)):
        if i % 50 == 0:
            print(f"Step {i}")
        curr_sigma = params.sigmas[params.seq[i]].cpu().numpy()
        t_i = utils_model.find_nearest(params.reduced_alpha_cumprod, curr_sigma)
        
        # skip iters
        if t_i > params.t_start:
            continue

        # Déterminer le prochain timestep pour le saut DDIM
        t_im1 = utils_model.find_nearest(params.reduced_alpha_cumprod, params.sigmas[params.seq[min(i+1, len(params.seq)-1)]].cpu().numpy())

        # -------------------------------------------------------
        # SOLUTION ANALYTIQUE ET SAUT DDIM (DiffPIR)
        # -------------------------------------------------------
        if i < (config.num_train_timesteps - config.noise_model_t):
            x_next, x0_est = single_diffpir_step(
                x, y, mask, t_i, t_im1, model, params.rhos, params.sigmas, params.alphas_cumprod, config.guidance_scale, zeta = config.zeta
            )
            x = x_next
        else:
            x_next, _ = simple_diffusion_step(model, x, t_i, t_im1, params.alphas_cumprod, eta=0.0)
            x = x_next

        current_x0 = x0_est if 'x0_est' in locals() else x
        x_0_progress = (current_x0 / 2 + 0.5)

        if (params.seq[i] in params.progress_seq):
            x_show = x_0_progress.clone().detach().cpu().numpy()       #[0,1]
            x_show = np.squeeze(x_show)
            if x_show.ndim == 3:
                x_show = np.transpose(x_show, (1, 2, 0))
            imshow(x_show, title=f'Denoised Image {i}', save_path=f"results/{config.name_folder_result}/etape_{i}.png", show=False)
            progress_img.append(x_show)
        torch.cuda.empty_cache()
    
    x[mask.to(torch.bool)] = y[mask.to(torch.bool)]

    metrics = run_evaluation(x, image, config, device)
    
    print(f"✅ Finish ! PSNR: {metrics['psnr']:.2f}, LPIPS: {metrics['lpips']:.4f}")
