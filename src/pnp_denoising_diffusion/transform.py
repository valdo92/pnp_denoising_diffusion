"""Transform the image (so far only inpainting)"""
import numpy as np

def transform_image(image, config):
    """
    Transform the image by adding inpainting
    The image is 256x256x3 between 0 and 1
    """
    mask = _get_mask(config) 
    image_transformed = image * mask
    return image_transformed 

def _get_mask(config):
    """
    Give the mask for the inpainting from the config
    Put a 128x128 mask in the middle of the image
    """
    mask = np.ones((256, 256, 3))
    mask[64:128+64, 64:128+64, :] = 0 
    return mask

