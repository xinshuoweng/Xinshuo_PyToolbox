# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes private functions for internal use only
import numpy as np

from xinshuo_miscellaneous import islist, isnparray

def safe_data(input_data):
	'''
	copy a list of data or a numpy data to the buffer for use
	'''
	if islist(input_data):
		return np.array(input_data)
	elif isnparray(input_data):
		return input_data.copy()
	else:
		assert False, 'only list of data and numpy data are supported'