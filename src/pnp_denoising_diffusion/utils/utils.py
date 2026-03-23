"""Utility functions"""
from box import Box
import yaml
import numpy as np
import random
import torch

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return Box(config)

def set_seed(seed=42):
    seed = seed 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
