# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes private functions for internal use only
import copy

from type_check import islist

################################################################## conversion ##################################################################
def safe_list(input_data, warning=True, debug=True):
	'''
	copy a list to the buffer for use

	parameters:
		input_data:		a list

	outputs:
		safe_data:		a copy of input data
	'''
	if debug: islist(input_data), 'the input data is not a list'
	safe_data = copy.copy(input_data)
	return safe_data