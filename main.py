"""The main code. You can modify config.yaml to change the parameters to run the code"""

import torch
from torch import device
import torch 
from pnp_denoising_diffusion.utils.score import calculate_psnr, calculate_fid   
import lpips
from pnp_denoising_diffusion.utils import load_config, set_seed, create_model_and_diffusion, dif
from pnp_denoising_diffusion.utils.utils import load_config, set_seed
from pnp_denoising_diffusion.utils.load_image import load_image
from pnp_denoising_diffusion.utils.read_image import read_and_save
from pnp_denoising_diffusion.transform import transform_image
from pnp_denoising_diffusion.guided_diffusion.script_util import create_model_and_diffusion


from pnp_denoising_diffusion.diffusion import simple_diffusion_step, single_diffpir_step


if __name__ == "__main__":
    config = load_config("config.yaml")
    set_seed(config.seed)
    image = load_image(config.path_to_image)  # 269 x 269 x 3
    image = image[:256, :256, :] # 256 x 256 x 3
    image_transformed = transform_image(image, config)
    
    # Initializing x
    x = torch.randn((256, 256, 3))
    model, diffusion = create_model_and_diffusion(**config.guided_diffusion)
    model.load_state_dict(torch.load(config.model_path, map_location="cpu"))

    

    read_and_save(image_transformed, config.path_to_save)
    
    

    model, diffusion = create_model_and_diffusion(config)
    model.load_state_dict(torch.load(config.model_path, map_location="cpu"))
    model.eval()


    # reverse diffusion for one image from random noise
    for i in range(len(seq)):
        curr_sigma = sigmas[seq[i]].cpu().numpy()
        t_i = utils_model.find_nearest(reduced_alpha_cumprod, curr_sigma)
        
        # skip iters
        if t_i > t_start:
            continue

        # Déterminer le prochain timestep pour le saut DDIM
        t_im1 = utils_model.find_nearest(reduced_alpha_cumprod, sigmas[seq[min(i+1, len(seq)-1)]].cpu().numpy())

        for u in range(iter_num_U):
            # -------------------------------------------------------
            # SOLUTION ANALYTIQUE ET SAUT DDIM (DiffPIR)
            # -------------------------------------------------------
            if i < (num_train_timesteps - noise_model_t):
                x_next, x0_est = single_diffpir_step(
                    x, y, mask, t_i, t_im1, model, rhos, sigmas, alphas_cumprod, guidance_scale
                )
                x = x_next
            else:
                x_next, _ = simple_diffusion_step(model, x, t_i, t_im1, alphas_cumprod, eta=0.0)
                x = x_next

            # -------------------------------------------------------
            # STEP 4: RE-BRUITAGE (Si itérations U > 1)
            # -------------------------------------------------------
            # Si on n'est pas à la dernière itération interne, on "remonte" le bruit
            if u < iter_num_U - 1 and seq[i] != seq[-1]:
                # On utilise le ratio d'alphas pour rajouter la dose exacte de bruit
                sqrt_alpha_effective = torch.sqrt(alphas_cumprod[t_i] / alphas_cumprod[t_im1])
                noise_to_add = torch.sqrt(
                    (1 - alphas_cumprod[t_i]) - (sqrt_alpha_effective**2 * (1 - alphas_cumprod[t_im1]))
                )
                x = sqrt_alpha_effective * x + noise_to_add * torch.randn_like(x)

        # Sauvegarde et affichage (x_0 est mis à l'échelle [0, 1])
        x_0_output = (x / 2 + 0.5)

    #recover intial ground truth image
    x[mask.to(torch.bool)] = y[mask.to(torch.bool)]
    # ... (votre logique de sauvegarde progressive ici)

    # for img in config.test_images:
    #     print(f"Processing image: {img}")
    #     # Load the noisy image and the ground truth image
    #     img_L_tensor = ...  # Load the noisy image as a tensor
    #     img_H_tensor = ...  # Load the ground truth image as a tensor

    #     # Denoising process using the diffusion model
    #     x_0 = ...  # Denoised image obtained from the diffusion model

    # ### Phase De Test
    #     loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    #     test_results = dict()  
    #     test_results['lpips'] = []   
    #     test_results['psnr'] = []
    #     test_results['fid'] = []


    #     fid_score = calculate_fid(denoised_images, ground_truth_images) ## tensor in range [0, 255]
    #     test_results['fid'].append(fid_score)
    #     lpips_score = loss_fn_vgg(x_0.detach()*2-1, img_H_tensor) ## tensor in range [-1, 1]
    #     test_results['lpips'].append(lpips_score.item())
    #     psnr_score = calculate_psnr(img_H_tensor.cpu().numpy(), x_0.cpu().numpy()) ## numpy array in range [0, 255]
    #     test_results['psnr'].append(psnr_score)

    # ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    # ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
    # ave_fid = sum(test_results['fid']) / len(test_results['fid'])

    # print("Average PSNR:", ave_psnr)
    # print("Average LPIPS:", ave_lpips)
    # print("Average FID:", ave_fid)
    image = load_image(config.path_to_image)  # 269x269x3
    read_and_save(image, config.path_to_save)
    print("piche")
