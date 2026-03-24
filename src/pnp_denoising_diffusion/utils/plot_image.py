import matplotlib.pyplot as plt
import numpy as np

def imshow(x, title=None, cbar=False, figsize=None):
    #x_show = x.clone().detach().cpu().numpy()       #[0,1]
    x_show = np.squeeze(x)
    if x_show.ndim == 3:
        x_show = np.transpose(x_show, (1, 2, 0))
    plt.figure(figsize=figsize)
    plt.imshow(np.squeeze(x_show), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()