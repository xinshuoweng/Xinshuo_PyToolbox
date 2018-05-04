# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes functions about images I/O
import numpy as np
from PIL import Image

from xinshuo_miscellaneous.python.private import safe_path
from xinshuo_images.python.private import safe_image

from file_io import mkdir_if_missing
from xinshuo_miscellaneous import is_path_exists_or_creatable, isimage, isscalar, is_path_exists
from xinshuo_images import image_rotate, image_resize
######################################################### image related #########################################################
def load_image(src_path, resize_factor=1.0, target_size=None, input_angle=0, warning=True, debug=True):
    '''
    load an image from given path, with preprocessing of resizing and rotating

    parameters:
        resize_factor:      a scalar
        target_size:        a list of tuple or numpy array with 2 elements, representing height and width
        input_angle:        a scalar, counterclockwise rotation in degree

    output:
        img:                an uint8 rgb numpy image
    '''
    src_path = safe_path(src_path, warning=warning, debug=debug)
    if debug: assert is_path_exists(src_path), 'txt path is not correct at %s' % src_path

    with open(src_path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
            img = image_rotate(img, input_angle=input_angle, warning=warning, debug=debug)
            img = image_resize(img, resize_factor=resize_factor, target_size=target_size, warning=warning, debug=debug)
            # scaling
            # if isscalar(resize_factor):
                # width, height = img.size
                # img = img.resize(size=(int(round(width*resize_factor)), int(round(height*resize_factor))), resample=Image.BILINEAR)
            # elif len(resize_factor) == 2:
                # resize_width, resize_height = int(resize_factor[0]), int(resize_factor[1])
                # img = img.resize(size=(resize_width, resize_height), resample=Image.BILINEAR)    
            # else: assert False, 'the resize factor is neither a scalar nor a (width, height)'
            
            # if mode == 'numpy': img = np.array(img)
    return img

def save_image(input_image, save_path, warning=True, debug=True):
    save_path = safe_path(save_path, warning=warning, debug=debug); mkdir_if_missing(save_path)
    np_image, _ = safe_image(input_image, warning=warning, debug=debug)
    pil_image = Image.fromarray(np_image)
    # imsave(save_path, input_image)
    pil_image.save(save_path)