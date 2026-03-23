"""The main code. You can modify config.yaml to change the parameters to run the code"""

from torch import device
from pnp_denoising_diffusion.utils.score import calculate_psnr, calculate_fid   
import lpips
from pnp_denoising_diffusion.utils import load_config, set_seed
from pnp_denoising_diffusion.utils.utils import load_config, set_seed
from pnp_denoising_diffusion.utils.load_image import load_image
from pnp_denoising_diffusion.utils.read_image import read_and_save



if __name__ == "__main__":
    config = load_config("config.yaml")
    set_seed(config.seed)
    
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
