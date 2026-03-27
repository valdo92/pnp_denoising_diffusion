"""The main code for the face swap. You can modify config_face_swap.yaml to change the parameters to run the code"""

import torch
import numpy as np
import os
from pnp_denoising_diffusion.utils.utils import load_config, set_seed
from pnp_denoising_diffusion.utils import utils_model 
from pnp_denoising_diffusion.utils.load_image import load_image
from pnp_denoising_diffusion.utils.plot_image import imshow
from pnp_denoising_diffusion.transform import transform_image_face_swap
from pnp_denoising_diffusion.diffusion import single_diffpir_step
from pnp_denoising_diffusion.utils.diffusion_utils import (
    get_params_diffusion, transfer_model_shape, initialize_x, load_diffusion_model,
    transfer_model_shape_one_image
    )


if __name__ == "__main__":
    print("⏳ Loading config, parameters and images...")
    config = load_config("config_face_swap.yaml")
    
    # if os.path.exists("results/" + config.name_folder_result):
    #    raise FileExistsError(f"🛑 : The folder '{config.name_folder_result}' exist, change it in config or delete the folder")
    os.makedirs(f"results/{config.name_folder_result}", exist_ok=True)
    os.system(f"cp config_face_swap.yaml results/{config.name_folder_result}/")
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    
    print("⏳ Loading the model and the weights...")
    model = load_diffusion_model(config)

       
    print("Processing the images") 
    params = get_params_diffusion(config)
    image_1 = load_image("data/" + config.name_image_1) # [256, 256, 3]
    image_2 = load_image("data/" + config.name_image_2)
    image_transformed, mask = transform_image_face_swap(image_1, image_2, config)
    image_1, image_transformed, mask = transfer_model_shape(
        image_1, image_transformed, mask, config.device 
        )
    image_2 = transfer_model_shape_one_image(image_2, config.device)
    y = image_transformed
    x = initialize_x(params, config, y)

    
    imshow(x, title='init random', save_path=f"results/{config.name_folder_result}/image_init_random.png", show=False)
    imshow(y, title='image transformé', save_path=f"results/{config.name_folder_result}/image_intiale_Transforme.png", show=False)


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
        x_next, x0_est = single_diffpir_step(
            x, y, mask, t_i, t_im1, model, params.rhos, params.sigmas, 
            params.alphas_cumprod, config.guidance_scale, zeta = config.zeta, face_swap=True
        )
        x = x_next

        x_0_progress = (x / 2 + 0.5)

        if (params.seq[i] in params.progress_seq):
            x_show = x_0_progress.clone().detach().cpu().numpy()       #[0,1]
            x_show = np.squeeze(x_show)
            if x_show.ndim == 3:
                x_show = np.transpose(x_show, (1, 2, 0))
            imshow(x_show, title=f'Denoised Image {i}', save_path=f"results/{config.name_folder_result}/etape_{i}.png", show=False)
            progress_img.append(x_show)
        torch.cuda.empty_cache()

    imshow(x, title='final_image', save_path=f"results/{config.name_folder_result}/final_image.png", show=False)
    y_vis = (y / 2 + 0.5).clamp(0, 1).squeeze().cpu().numpy().transpose(1, 2, 0)
    x_vis = (x / 2 + 0.5).clamp(0, 1).squeeze().cpu().numpy().transpose(1, 2, 0)
    image_1_vis = (image_1 / 2 + 0.5).clamp(0, 1).squeeze().cpu().numpy().transpose(1, 2, 0)
    image_2_vis = (image_2 / 2 + 0.5).clamp(0, 1).squeeze().cpu().numpy().transpose(1, 2, 0)

    composite = np.concatenate([
        image_1_vis,
        image_2_vis,
        x_vis
    ], axis=1)
    imshow(composite, title='Comparison: Image 1 | Image 2 | Result', 
           save_path=f"results/{config.name_folder_result}/comparison.png", show=False)
