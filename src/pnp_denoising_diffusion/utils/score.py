import math
import numpy as np
import torch 
import cv2
#from torchmetrics.image.fid import FrechetInceptionDistance

def calculate_fid_process(denoised_images, ground_truth_images):
    """
    Calcule le FID entre les images débruitées et les images de référence.
    denoised_images & ground_truth_images: Tenseurs (B, C, H, W) type uint8
    """
    # Initialisation (feature=2048 est le standard pour Inception-v3)
    fid = FrechetInceptionDistance(feature=2048).to(denoised_images.device)

    # Mise à jour avec les images réelles (Ground Truth)
    fid.update(ground_truth_images, real=True)
    
    # Mise à jour avec les images générées (Denoised)
    fid.update(denoised_images, real=False)

    # Calcul final
    return fid.compute().item()


def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

# def calculate_psnr_batch(batch1, batch2, max_pixel=2.0, eps=1e-10):
#     if not batch1.shape == batch2.shape:
#         raise ValueError('Input images must have the same dimensions.')
#     mse = torch.mean((batch1 - batch2) ** 2, axis=(1, 2, 3))
#     zeros = torch.zeros_like(mse)
#     inf = torch.ones_like(mse) * float('inf')
#     psnr_values = torch.where(mse == 0, inf, 20 * torch.log10(max_pixel / torch.sqrt(mse + eps)))
#     psnr_values = torch.where(torch.isnan(psnr_values), zeros, psnr_values)
#     mean_psnr = torch.mean(psnr_values)
#     return mean_psnr.item()

# def calculate_ssim(img1, img2, border=0):
#     '''calculate SSIM
#     the same outputs as MATLAB's
#     img1, img2: [0, 255]
#     '''
#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')
#     h, w = img1.shape[:2]
#     img1 = img1[border:h-border, border:w-border]
#     img2 = img2[border:h-border, border:w-border]

#     if img1.ndim == 2:
#         return ssim(img1, img2)
#     elif img1.ndim == 3:
#         if img1.shape[2] == 3:
#             ssims = []
#             for i in range(3):
#                 ssims.append(ssim(img1, img2))
#             return np.array(ssims).mean()
#         elif img1.shape[2] == 1:
#             return ssim(np.squeeze(img1), np.squeeze(img2))
#     else:
#         raise ValueError('Wrong input image dimensions.')


# def ssim(img1, img2):
#     C1 = (0.01 * 255)**2
#     C2 = (0.03 * 255)**2

#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())

#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()