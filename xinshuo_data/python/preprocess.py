# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file contains function to preprocess data
import numpy as np
from PIL import Image
from numpy.testing import assert_almost_equal

from xinshuo_miscellaneous import isnparray, iscolorimage, istuple, islist, CHECK_EQ_LIST_SELF, isimage, isgrayimage, isuintimage, isfloatimage
from xinshuo_visualization import visualize_image


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
		# print('debug mode is on during data normalizing. Please turn off after debuging')

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

	# print data
	normalized_data = np.zeros((1, 1))
	normalized_data[0] = float(data - min_value)
	normalized_data[0] = normalized_data / (max_value - min_value)
	
	if debug:
		unnormalized = unnormalize_data(data=normalized_data, data_range=(min_value, max_value), debug=False)
		assert_almost_equal(data, unnormalized, decimal=6, err_msg='data is not correct: %f vs %f' % (data, unnormalized))
	return normalized_data


def unnormalize_data(data, data_range, debug=True):
	'''
	this function unnormalizes 1-d label to normal scale based on range of data
	'''
	if debug:
		# print('debug mode is on during data unnormalizing. Please turn off after debuging')
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

