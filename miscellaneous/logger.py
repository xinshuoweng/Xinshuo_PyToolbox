# Author: Xinshuo Weng
# Email: xinshuo.weng@gmail.com

# logging



def print_log(print_string, log):
	print("{}".format(print_string))
	if log is not None:
		log.write('{}\n'.format(print_string))
		log.flush()

def print_np_shape(nparray, debug=True):
	'''
	print a string to represent the shape of a numpy array
	'''
	if debug:
		assert isnparray(nparray), 'input is not a numpy array and does not have any shape'

	return '(%s)' % (functools.reduce(lambda x, y: str(x) + ', ' + str(y), nparray.shape))

def print_torch_size(torch_size):
	dims = len(torch_size)
	string = '['
	for idim in range(dims):
		string = string + ' {}'.format(torch_size[idim])
	return string + ']'