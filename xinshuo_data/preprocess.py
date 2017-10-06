# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file contains function to preprocess data
import numpy as np
from numpy.testing import assert_almost_equal

import __init__paths__
from xinshuo_python import isnparray, iscolorimage, istuple, islist, CHECK_EQ_LIST, isimage, isgrayimage, isuintimage, isfloatimage
from xinshuo_visualization import visualize_save_image


def identity(data, data_range=None, debug=True):
    if debug:
        print('debug mode is on during identity function. Please turn off after debuging')
        assert isnparray(data), 'data is not correct'
    return data
    

def normalize_data(data, data_range=None, debug=True):
	'''
	this function normalizes 1-d label to 0-1
	'''
	if debug:
		assert isnparray(data), 'only numpy array is supported'
		print('debug mode is on during data normalizing. Please turn off after debuging')

	if data_range is None:
		max_value = np.max(data)
		min_value = np.min(data)
	else:
		if debug:
			if istuple(data_range) or islist(data_range):
				assert len(data_range) == 2, 'data range is not correct'
			elif isnparray(data_range):
				assert data_range.size == 2, 'data range is not correct'
		max_value = data_range[1]
		min_value = data_range[0]

	normalized_data = float(data - min_value)
	normalized_data = normalized_data / (max_value - min_value)
	
	if debug:
		unnormalized = unnormalize_data(data=normalized_data, data_range=(min_value, max_value), debug=False)
		assert_almost_equal(data, unnormalized, decimal=6, err_msg='data is not correct: %f vs %f' % (data, unnormalized))
	return normalized_data


def unnormalize_data(data, data_range, debug=True):
	'''
	this function unnormalizes 1-d label to normal scale based on range of data
	'''
	if debug:
		print('debug mode is on during data unnormalizing. Please turn off after debuging')
		assert isnparray(data), 'only numpy array is supported'
		if istuple(data_range) or islist(data_range):
			assert len(data_range) == 2, 'data range is not correct'
		elif isnparray(data_range):
			assert data_range.size == 2, 'data range is not correct'
	max_value = data_range[1]
	min_value = data_range[0]
	unnormalized = data * (max_value - min_value) + min_value

	if debug:
		normalized = normalize_data(data=unnormalized, data_range=data_range, debug=False)
		assert_almost_equal(data, normalized, decimal=6, err_msg='data is not correct: %f vs %f' % (data, normalized))
	return unnormalized


def preprocess_image_caffe(image_datalist, debug=True, vis=False):
	'''
	this function preprocesses image for caffe only,
	including transfer from rgb to bgr
	from HxWxC to NxCxHxW
	'''
	if debug:
		print('debug mode is on during preprocessing. Please turn off after debuging')
		assert islist(image_datalist), 'input is not a list of image'
		assert all(isimage(image_data, debug=debug) for image_data in image_datalist), 'input is not a list of image'
		shape_list = [image_data.shape for image_data in image_datalist]
		assert CHECK_EQ_LIST(shape_list), 'image shape is not equal inside one batch'

	data_warmup = image_datalist[0]
	if iscolorimage(data_warmup, debug=debug):
		color = True
	elif isgrayimage(data_warmup, debug=debug):
		color = False
		if data_warmup.ndim == 2:
			image_datalist = [np.reshape(image_data, image_data.shape + (1, )) for image_data in image_datalist]
	else:
		assert False, 'only color or gray image is supported'

	if isuintimage(data_warmup, debug=debug):
		uint = True
	elif isfloatimage(data_warmup, debug=debug):
		uint = False
	else:
		assert False, 'only uint8 or float image is supported'		

	if debug:
		if color:
			assert all(iscolorimage(image_data, debug=debug) for image_data in image_datalist), 'input should be all color image format'	
		else:
			assert all(isgrayimage(image_data, debug=debug) and image_data.ndim == 3 and image_data.shape[-1] == 1 for image_data in image_datalist), 'input should be all grayscale image format'	
		if uint:
			assert all(isuintimage(image_data, debug=debug) for image_data in image_datalist), 'input should be all color image format'	
		else:
			assert all(isfloatimage(image_data, debug=debug) for image_data in image_datalist), 'input should be all grayscale image format'	

	batch_size = len(image_datalist)
	caffe_input_data = np.zeros((batch_size, ) + image_datalist[0].shape, dtype=data_warmup.dtype)

	# fill one minibatch data
	index = 0
	for image_data in image_datalist:
		caffe_input_data[index, :, :, :] = image_data
		index += 1
		
	if color:
		caffe_input_data = caffe_input_data[:, :, :, [2, 1, 0]]                 # from rgb to bgr, currently [batch, height, weight, channels]
	
	if debug:
		if vis:																# visualize swapped channel
			print('visualization in debug mode is on during preprocessing. Please turn off after confirmation')
			for index in xrange(caffe_input_data.shape[0]):
				image_tmp_swapped = caffe_input_data[index]
				print('\n\nPlease make sure the image is not RGB after swapping channel')
				visualize_save_image(image_tmp_swapped, debug=debug)
		assert caffe_input_data.shape[-1] == 3 or caffe_input_data.shape[-1] == 1, 'channel is not correct'
	caffe_input_data = np.transpose(caffe_input_data, (0, 3, 1, 2))         # permute to [batch, channel, height, weight]
	
	if debug:
		assert caffe_input_data.shape[1] == 3 or caffe_input_data.shape[1] == 1, 'channel is not correct'
	return caffe_input_data

def unpreprocess_image_caffe(image_datablob, pixel_mean=None, swap_channel=True, debug=True):
	'''
	this function unpreprocesses image for caffe only,
	including transfer from bgr to rgb
	from NxCxHxW to a list of HxWxC 
	'''
	if debug:
		print('debug mode is on during unpreprocessing. Please turn off after debuging')
		assert isnparray(image_datablob) and image_datablob.ndim == 4, 'input is not correct'	
		assert image_datablob.shape[1] == 1 or image_datablob.shape[1] == 3, 'this is not an blob of image, channel is not 1 or 3'

	if pixel_mean is not None:
		assert pixel_mean.shape == (1, 1, 3) or pixel_mean.shape == (1, ), 'pixel mean is not correct'
		pixel_mean_reshape = np.reshape(pixel_mean, (1, 3, 1, 1))
		image_datablob += pixel_mean_reshape

	image_datablob = np.transpose(image_datablob, (0, 2, 3, 1))         # permute to [batch, height, weight, channel]	

	if image_datablob.shape[-1] == 3 and swap_channel:	# channel dimension
		image_datablob = image_datablob[:, :, :, [2, 1, 0]]             # from bgr to rgb for color image
	image_data_list = list()
	for i in xrange(image_datablob.shape[0]):
		image_data_list.append(image_datablob[i, :, :, :])
	return image_data_list