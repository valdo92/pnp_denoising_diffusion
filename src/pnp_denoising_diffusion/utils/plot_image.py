import matplotlib.pyplot as plt
import numpy as np
import torch

def imshow(x, title=None, cbar=False, figsize=None):
    # Si x est un tenseur PyTorch, on le détache et on le passe sur CPU en numpy
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        
    x_show = np.squeeze(x)
    
    # Si l'image est au format (Canaux, Hauteur, Largeur), on transpose en (Hauteur, Largeur, Canaux)
    # On vérifie que la dimension 0 correspond bien à des canaux (généralement 1, 3 ou 4)
    if x_show.ndim == 3 and x_show.shape[0] in [1, 3, 4]:
        x_show = np.transpose(x_show, (1, 2, 0))
        
    # Squeeze une nouvelle fois au cas où on aurait eu 1 seul canal (H, W, 1) -> (H, W)
    x_show = np.squeeze(x_show)
    
    # Si ce sont des floats, matplotlib s'attend à des valeurs entre 0 et 1
    if np.issubdtype(x_show.dtype, np.floating):
        # Si l'image est entre -1 et 1, on la ramène entre 0 et 1
        if x_show.min() < 0.0:
            x_show = (x_show + 1.0) / 2.0
        x_show = np.clip(x_show, 0.0, 1.0)
    elif np.issubdtype(x_show.dtype, np.integer):
        x_show = np.clip(x_show, 0, 255)

    plt.figure(figsize=figsize)
    # cmap=gray seulement si c'est une image en niveaux de gris (2 dimensions)
    cmap = 'gray' if x_show.ndim == 2 else None
    
    plt.imshow(x_show, interpolation='nearest', cmap=cmap)
    
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()