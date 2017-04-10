# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file defines a set of help function when using python in matlab

import numpy as np
import array

def get_nparray_from_array(tuple_array):
	'''
	this function convert a tuple of 1-d array to 
	a 2-d numpy arrya
	'''
	assert isinstance(tuple_array, tuple), 'input should be a tuple'
	assert all(isinstance(array_tmp, array.array) for array_tmp in tuple_array), 'input is not a tuple of array'

	return np.array(tuple_array)


def toy_function(nparray):
	assert isinstance(nparray, np.ndarray), 'input is not correct'
	return nparray + 1