# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes private functions for internal use only
import numpy as np
from xinshuo_miscellaneous import ispilimage, isnpimage, isnparray, isnpimage_dimension, isnannparray

def safe_image(input_image, warning=True):
	'''
	return a numpy image no matter what format the input is
	make sure the output numpy image is a copy of the input image

	parameters:
		input_image:		pil or numpy image, color or gray, float or uint

	outputs:
		np_image:			numpy image, with the same color and datatype as the input
		isnan:				return True if any nan value exists
	'''
	if ispilimage(input_image):
		np_image = np.array(input_image)
	elif isnpimage(input_image):
		np_image = input_image.copy()
	else:
		assert False, 'only pil and numpy images are supported, might be the case the image is float but has range of [0, 255], or might because the data is float64'

	isnan = isnannparray(np_image)
	if warning and isnan:
		print('nan exists in the image data')

	return np_image, isnan

def safe_image_like(input_image, warning=True):
	'''
	return an image-like numpy no matter what format the input is
	make sure the output numpy image is a copy of the input image

	note:
		an image-like numpy array is an array with image-like shape, but might contain arbitrary value

	parameters:
		input_image:		pil image or image-like array, color or gray, float or uint

	outputs:
		np_image:			numpy image, with the same color and datatype as the input
		isnan:				return True if any nan value exists
	'''
	if ispilimage(input_image):
		np_image = np.array(input_image)
	elif isnparray(input_image):
		np_image = input_image.copy()
		assert isnpimage_dimension(np_image), 'the input is not an image-like numpy array'
	else:
		assert False, 'only pil and numpy image-like data are supported'

	isnan = isnannparray(np_image)
	if warning and isnan:
		print('nan exists in the image data')

	return np_image, isnan