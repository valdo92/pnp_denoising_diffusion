"""The main code. You can modify config.yaml to change the parameters to run the code"""

from pnp_denoising_diffusion.utils.utils import load_config, set_seed

if __name__ == "__main__":
    config = load_config("config.yaml")
    set_seed(config.seed)
    
    print("piche")
