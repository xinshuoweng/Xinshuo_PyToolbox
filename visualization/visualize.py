# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import matplotlib.pyplot as plt
import numpy as np

import __init__paths__
from check import isimage, is_path_exists_or_creatable, isfile, islist, isnparray, isgrayimage, iscolorimage

def visualize_save_image(image, vis=True, save=False, save_path=None):
    if islist(image):
        print('visualizing a list of images:')
        index = 1
        for image_tmp in image:
            print('processing %d/%d' % (index, len(image)))
            visualize_save_image(image_tmp, vis, save_path, save)
            index += 1
        return

    assert isnparray(image), 'input image is not a numpy array {}'.format(type(image))
    assert isimage(image), 'input is not a good image, shape is {}'.format(image.shape)
    if save:
        assert is_path_exists_or_creatable(save_path) and isfile(save_path), 'save path is not valid'

    dpi = 80  
    width = image.shape[1]
    height = image.shape[0]
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    if iscolorimage(image):
        ax.imshow(image, interpolation='nearest')
    elif isgrayimage(image):
        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.reshape(image, image.shape[:-1])
        ax.imshow(image, interpolation='nearest', cmap='gray')
    else:
        assert False, 'image is not correct'
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

    if save:
        fig.savefig(save_path, dpi=dpi, transparent=True)
    if vis:
        plt.show()

    plt.close(fig)