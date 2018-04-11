# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes functions about images I/O
import numpy as np
from scipy.misc import imsave
from PIL import Image

from file_io import mkdir_if_missing
from xinshuo_miscellaneous import is_path_exists_or_creatable, isimage, isscalar, is_path_exists, safepath
######################################################### image related #########################################################
def load_image(src_path, resize_factor=1.0, rotate=0, mode='numpy', debug=True):
    '''
    load an image from given path

    parameters:
        resize_factor:      resize the image (>1 enlarge)
        mode:               numpy or pil, specify the format of returned image
        rotate:             counterclockwise rotation in degree

    output:
        img:                an uint8 rgb image (numpy or pil)
    '''

    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    src_path = safepath(src_path)

    if debug:
        assert is_path_exists(src_path), 'txt path is not correct at %s' % src_path
        assert mode == 'numpy' or mode == 'pil', 'the input mode for returned image is not correct'
        assert (isscalar(resize_factor) and resize_factor > 0) or len(resize_factor) == 2, 'the resize factor is not correct: {}'.format(resize_factor) 

    with open(src_path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')

            # rotation
            if rotate != 0:
                img = img.rotate(rotate, expand=True)

            # scaling
            if isscalar(resize_factor):
                width, height = img.size
                img = img.resize(size=(int(width*resize_factor), int(height*resize_factor)), resample=Image.BILINEAR)
            elif len(resize_factor) == 2:
                resize_width, resize_height = int(resize_factor[0]), int(resize_factor[1])
                img = img.resize(size=(resize_width, resize_height), resample=Image.BILINEAR)    
            else:
                assert False, 'the resize factor is neither a scalar nor a (width, height)'

            # formating
            if mode == 'numpy':
                img = np.array(img)

    return img

def save_image_from_data(save_path, data, debug=True, vis=False):
    save_path = safepath(save_path)
    if debug:
        assert isimage(data), 'input data is not image format'
        assert is_path_exists_or_creatable(save_path), 'save path is not correct'
    
    mkdir_if_missing(save_path)
    imsave(save_path, data)