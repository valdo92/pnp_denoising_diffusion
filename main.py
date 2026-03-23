"""The main code. You can modify config.yaml to change the parameters to run the code"""

from pnp_denoising_diffusion.utils.utils import load_config, set_seed
from pnp_denoising_diffusion.utils.load_image import load_image
from pnp_denoising_diffusion.utils.read_image import read_and_save

if __name__ == "__main__":
    config = load_config("config.yaml")
    set_seed(config.seed)
    
    image = load_image(config.path_to_image)  # 269x269x3
    read_and_save(image, config.path_to_save)
    print("piche")
