# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes functions of basic probability and statistics
import math, cv2, numpy as np

from private import safe_npdata
from xinshuo_miscellaneous import isnparray

def hist_equalization(input_data, num_bins=256, debug=True):
	'''
	convert a N-d numpy data (or list) with random distribution to a 1-d data with equalized histogram
	e.g., for the samples from a gaussian distribution, the data points are dense in the middle, the cdf increases fast
	in the middle so that the discrete cdf is sparse in the middle, the equalized data points are interpolated from cdf such
	that the density can be the same for the middle and the rest

	parameters:
		input_data:		a list or a numpy data, could be any shape, not necessarily a 1-d data, can be integer data (uint8 image) or float data (float32 image)
		num_bins:		bigger, the histogram of equalized data points is more flat

	outputs:
		data_equalized:	equalized data with the same shape as input, it is float with [0, 1]
	'''
	np_data = safe_npdata(input_data)

	if debug:
		assert isnparray(np_data), 'the input data is not a numpy data'

	ori_shape = np_data.shape
	np_data = np_data.flatten()
	hist, xs = np.histogram(np_data, num_bins, density=True)	# return distribution and X's coordinates
	cdf = hist.cumsum()
	cdf = cdf / cdf[-1]			# sparse in the middle
	data_equalized = np.interp(np_data, xs[:-1], cdf)

	return data_equalized.reshape((ori_shape))

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

def identity(data, data_range=None, debug=True):
    if debug:
        print('debug mode is on during identity function. Please turn off after debuging')
        assert isnparray(data), 'data is not correct'
    return data