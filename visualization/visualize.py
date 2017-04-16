# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import matplotlib.pyplot as plt
import numpy as np

import __init__paths__
from check import isimage, is_path_exists_or_creatable, isfile, islist, isnparray, isgrayimage, iscolorimage
from file_io import mkdir_if_missing

def visualize_save_image(image, vis=True, save=False, save_path=None, debug=True):
    if islist(image):
        imagelist = image
        save_path_list = save_path
        if vis:
            print('visualizing a list of images:')
        if save:
            print('saving a list of images')
            if debug:
                assert islist(save_path_list), 'for saving a list of images, please provide a list of saving path'
                assert all(is_path_exists_or_creatable(save_path_tmp) and isfile(save_path_tmp) for save_path_tmp in save_path_list), 'save path is not valid'
                assert len(save_path_list) == len(imagelist), 'length of list for saving path and data is not equal'
        index = 0
        for image_tmp in imagelist:
            print('processing %d/%d' % (index+1, len(imagelist)))
            visualize_save_image(image_tmp, vis, save_path[i], save)
            index += 1
        return

    if debug:
        assert isnparray(image), 'input image is not a numpy array {}'.format(type(image))
        assert isimage(image), 'input is not a good image, shape is {}'.format(image.shape)

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
        if debug:
            assert is_path_exists_or_creatable(save_path) and isfile(save_path), 'save path is not valid: %s' % save_path
            mkdir_if_missing(save_path)
        fig.savefig(save_path, dpi=dpi, transparent=True)
    if vis:
        plt.show()

    plt.close(fig)