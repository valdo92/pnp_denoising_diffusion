"""The main code. You can modify config.yaml to change the parameters to run the code"""

import torch
import numpy as np 
from pnp_denoising_diffusion.utils.score import calculate_psnr, calculate_fid_process   
import lpips
from pnp_denoising_diffusion.utils.utils import load_config, set_seed
from pnp_denoising_diffusion.utils import utils_model 
from pnp_denoising_diffusion.utils.load_image import load_image
from pnp_denoising_diffusion.utils.read_image import read_and_save
from pnp_denoising_diffusion.utils.plot_image import imshow
from pnp_denoising_diffusion.transform import transform_image 
from pnp_denoising_diffusion.guided_diffusion.script_util import create_model_and_diffusion
from pnp_denoising_diffusion.diffusion import simple_diffusion_step, single_diffpir_step


if __name__ == "__main__":
    config = load_config("config.yaml")
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("--- Parameters ---")
    # TODO: mettre ça dans une fonction
    ### PREPARATION DES PARAMETRES DE LA DIFFUSION ###
    skip = config.num_train_timesteps // config.iter_num
    betas = np.linspace(config.beta_start, config.beta_end, config.num_train_timesteps, dtype=np.float32)
    betas = torch.from_numpy(betas).to(device)
    # Remplacement sécurisé
    # betas = torch.linspace(config.beta_start, config.beta_end, config.num_train_timesteps, dtype=torch.float32).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)  
    t_start = config.num_train_timesteps - 1 

    image = load_image(config.path_to_image)  # 269 x 269 x 3
    image = image[:256, :256, :] # 256 x 256 x 3
    image_transformed, mask = transform_image(image, config)
    

    # transfering to shape 1x3x256x256 and mapping to [-1, 1] 
    # TODO: mettre ça dans une fonction
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device)
    image = image * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    
    mask = torch.from_numpy(mask).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    image_transformed = torch.from_numpy(
        image_transformed
        ).permute(2, 0, 1).float().unsqueeze(0).to(device)
    image_transformed = image_transformed * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    y = image_transformed

    # TODO: mettre ça dans config 
    noise_level_img = 0.0 # Default value
    noise_model_t = 0 # Default value

    t_y = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_level_img)
    sqrt_alpha_effective = sqrt_alphas_cumprod[t_start] / sqrt_alphas_cumprod[t_y]
    x = sqrt_alpha_effective * y + torch.sqrt(sqrt_1m_alphas_cumprod[t_start]**2 - \
                    sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_y]**2) * torch.randn_like(y)
    

    model, diffusion = create_model_and_diffusion(**config.guided_diffusion)
    model.load_state_dict(torch.load(config.model_path, map_location="cpu"))
    model = model.to(device)
    model.eval()

    progress_img = []
    # create sequence of timestep for sampling
    if config.skip_type == 'uniform':
        seq = [i*skip for i in range(config.iter_num)]
        if skip > 1:
            seq.append(config.num_train_timesteps-1)
    elif config.skip_type == "quad":
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
    rhos, sigmas, sigma_ks = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device), torch.tensor(sigma_ks).to(device)
    
    imshow(x, title='init random', save_path="results/init_random.png", show=False)
    imshow(y, title='image transformé', save_path="results/Image_intiale_Transformée.png", show=False)


    print(f"--- Reverse Diffusion --- {len(seq)} ")
    # reverse diffusion for one image from random noise
    for i in range(len(seq)):
        print(i)
        curr_sigma = sigmas[seq[i]].cpu().numpy()
        t_i = utils_model.find_nearest(reduced_alpha_cumprod, curr_sigma)
        
        # skip iters
        if t_i > t_start:
            continue

        # Déterminer le prochain timestep pour le saut DDIM
        t_im1 = utils_model.find_nearest(reduced_alpha_cumprod, sigmas[seq[min(i+1, len(seq)-1)]].cpu().numpy())

        # -------------------------------------------------------
        # SOLUTION ANALYTIQUE ET SAUT DDIM (DiffPIR)
        # -------------------------------------------------------
        if i < (config.num_train_timesteps - noise_model_t):
            x_next, x0_est = single_diffpir_step(
                x, y, mask, t_i, t_im1, model, rhos, sigmas, alphas_cumprod, config.guidance_scale
            )
            x = x_next
        else:
            x_next, _ = simple_diffusion_step(model, x, t_i, t_im1, alphas_cumprod, eta=0.0)
            x = x_next

            # # -------------------------------------------------------
            # # STEP 4: RE-BRUITAGE (Si itérations U > 1)
            # # -------------------------------------------------------
            # # Si on n'est pas à la dernière itération interne, on "remonte" le bruit
            # if u < config.iter_num_U - 1 and seq[i] != seq[-1]:
            #     # On utilise le ratio d'alphas pour rajouter la dose exacte de bruit
            #     sqrt_alpha_effective = torch.sqrt(config.alphas_cumprod[t_i] / config.alphas_cumprod[t_im1])
            #     noise_to_add = torch.sqrt(
            #         (1 - config.alphas_cumprod[t_i]) - (sqrt_alpha_effective**2 * (1 - config.alphas_cumprod[t_im1]))
            #     )
            #     x = sqrt_alpha_effective * x + noise_to_add * torch.randn_like(x)

        current_x0 = x0_est if 'x0_est' in locals() else x
        x_0_progress = (current_x0 / 2 + 0.5)

        if (seq[i] in progress_seq):
            x_show = x_0_progress.clone().detach().cpu().numpy()       #[0,1]
            x_show = np.squeeze(x_show)
            if x_show.ndim == 3:
                x_show = np.transpose(x_show, (1, 2, 0))
            imshow(x_show, title=f'Denoised Image {i}', save_path=f"results/etape_{i}.png", show=False)
            progress_img.append(x_show)
        
    #recover intial ground truth image
    x[mask.to(torch.bool)] = y[mask.to(torch.bool)]

    x_0_output = (x / 2 + 0.5)
    print( "--- Measurements ---")
    # ### Phase De Test
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    test_results = dict()  
    test_results['lpips'] = []   
    test_results['psnr'] = []
    test_results['fid'] = []

    # Convert to standard formats for metrics
    image_norm = image  # Already in [-1, 1]
    
    # image_uint8 converts from [-1, 1] to [0, 255]
    image_uint8 = ((image / 2 + 0.5) * 255).clamp(0, 255).to(torch.uint8)
    
    img_psnr_gt = np.transpose(image_uint8.squeeze(0).cpu().numpy(), (1, 2, 0)) # [0, 255] HWC
    img_psnr_est = np.transpose(x_0_uint8.squeeze(0).cpu().numpy(), (1, 2, 0)) # [0, 255] HWC

    fid_score = calculate_fid_process(x_0_uint8, image_uint8) ## tensor in range [0, 255]
    test_results['fid'].append(fid_score)
    lpips_score = loss_fn_vgg(x_0_output.detach()*2-1, image_norm) ## tensor in range [-1, 1]
    test_results['lpips'].append(lpips_score.item())
    psnr_score = calculate_psnr(img_psnr_gt, img_psnr_est) ## numpy array in range [0, 255]
    test_results['psnr'].append(psnr_score)

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
    ave_fid = sum(test_results['fid']) / len(test_results['fid'])

    print("Average PSNR:", ave_psnr)
    print("Average LPIPS:", ave_lpips)
    print("Average FID:", ave_fid)
    read_and_save(img_psnr_est, config.path_to_save)
    print("piche")
