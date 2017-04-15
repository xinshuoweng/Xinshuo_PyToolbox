# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file contains function to preprocess data
from cv2 import imread
import numpy as np

import __init__paths__
from check import isnparray


def normalize_data(data, data_range=None):
	assert isnparray(data), 'only numpy array is supported'

	if data_range is None:
		max_value = np.max(data)
		min_value = np.min(data)
	else:
		assert isnparray(data_range) and data_range.size == 2, 'data range is not correct'
		max_value = data_range[1]
		min_value = data_range[0]

	normalized_data = data - min_value
	normalized_data = normalized_data / (max_value - min_value)
	return normalized_data


def unnormalize_data(data, data_range):
	assert isnparray(data), 'only numpy array is supported'
	assert isnparray(data_range) and data_range.size == 2, 'data range is not correct'
	max_value = data_range[1]
	min_value = data_range[0]

	return data * (max_value - min_value) + min_value
