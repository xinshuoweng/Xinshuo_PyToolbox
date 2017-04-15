# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file contains function to preprocess data
from cv2 import imread
import numpy as np

import __init__paths__
from check import isnparray, iscolorimage, istuple, islist


def normalize_data(data, data_range=None):
	'''
	this function normalizes 1-d label to 0-1
	'''
	assert isnparray(data), 'only numpy array is supported'

	if data_range is None:
		max_value = np.max(data)
		min_value = np.min(data)
	else:
		if istuple(data_range) or islist(data_range):
			assert len(data_range) == 2, 'data range is not correct'
		elif isnparray(data_range):
			assert data_range.size == 2, 'data range is not correct'
		max_value = data_range[1]
		min_value = data_range[0]

	normalized_data = data - min_value
	normalized_data = normalized_data / (max_value - min_value)
	return normalized_data


def unnormalize_data(data, data_range):
	'''
	this function unnormalizes 1-d label to normal scale based on range of data
	'''
	assert isnparray(data), 'only numpy array is supported'
	if istuple(data_range) or islist(data_range):
		assert len(data_range) == 2, 'data range is not correct'
	elif isnparray(data_range):
		assert data_range.size == 2, 'data range is not correct'
	max_value = data_range[1]
	min_value = data_range[0]

	return data * (max_value - min_value) + min_value


def preprocess_image_caffe(image_data):
	'''
	this function preprocesses image for caffe only,
	including transfer from rgb to bgr
	from HxWxC to NxCxHxW
	'''
	assert iscolorimage(image_data), 'input is not a image format'	

	image_data = np.reshape(image_data, (1, ) + image_data.shape)
	image_data = image_data[:, :, :, [2, 1, 0]]                 # from rgb to bgr, currently [batch, height, weight, channels]
	image_data = np.transpose(image_data, (0, 3, 1, 2))         # permute to [batch, channel, height, weight]
	return image_data

def unpreprocess_image_caffe(image_data):
	'''
	this function unpreprocesses image for caffe only,
	including transfer from bgr to rgb
	from NxCxHxW to a list of HxWxC 
	'''
	assert isnparray(image_data) and image_data.ndim == 4, 'input is not correct'	

	image_data = np.transpose(image_data, (0, 2, 3, 1))         # permute to [batch, height, weight, channel]
	# image_data = image_data[:, :, :, [2, 1, 0]]                 # from bgr to rgb 
	image_data_list = list()
	for i in xrange(image_data.shape[0]):
		image_data_list.append(image_data[i, :, :, :])
	return image_data_list