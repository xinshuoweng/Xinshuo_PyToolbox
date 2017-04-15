# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import matplotlib.pyplot as plt

import __init__paths__
from check import isimage, is_path_exists_or_creatable, isfile

def visualize_save_image(image, vis=True, save_path=None, save=False):
    assert isimage(image), 'input is not a good image'
    if save:
        assert is_path_exists_or_creatable(save_path) and isfile(save_path), 'save path is not valid'

    dpi = 80  
    width = image.shape[1]
    height = image.shape[0]
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(image, interpolation='nearest')
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

    if save:
        fig.savefig(save_path, dpi=dpi, transparent=True)
    if vis:
        plt.show()

    plt.close(fig)