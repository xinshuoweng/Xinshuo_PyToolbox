# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes private functions for internal use only
import numpy as np
from xinshuo_miscellaneous import ispilimage, isnpimage

def safe_image(input_image):
	'''
	return a numpy image no matter what input is
	make sure the output numpy image is a copy of the input image
	'''
	if ispilimage(input_image):
		np_image = np.array(input_image)
	elif isnpimage(input_image):
		np_image = input_image.copy()
	else:
		assert False, 'only pil and numpy images are supported, might be the case the image is float but has range of [0, 255], or might because the data is float64'

	return np_image	