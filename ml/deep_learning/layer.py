# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import __init__
from type_check import isstring

class Input(object):
	'''
	define an input layer which contains info about the input data
	'''
	def __init__(self, data, name):
		assert isinstance(data, np.ndarray), 'the input data layer shoule contains numpy array'
		assert isstring(name), 'the name of input layer should be a string'
		self._data = data
		self._name = name

	@property
	def name(self):
		return self._name

	@property
	def data(self):
		return self._data

	@property
	def datatype(self):
		return self.data.dtype

	def get_memory_usage_data():
		return self.data.nbytes

class Layer(object):
	'''
	define necessary layer parameter and property for deep learning
	'''

	def __init__(self, name, nInputPlane, nOutputPlane, kernal_size=None, stride=None, padding=None, datatype=None):
		assert kernal_size is None or type(kernal_size) is int or len(kernal_size) == 2, 'kernal size is not correct'
		assert stride is None or type(stride) is int or len(stride) == 2, 'stride size is not correct'
		assert padding is None or type(padding) is int or len(padding) == 2, 'padding size is not correct'
		assert type(nInputPlane) is int, 'number of input channel is not correct'
		assert type(nOutputPlane) is int, 'number of output channel is not correct'
		assert isstring(name), 'name of layer is not correct'
		if type(kernal_size) is not int and kernal_size is not None:
			assert all(item > 0 and type(item) is int for item in kernal_size), 'kernal size must be positive integer'
		if type(stride) is not int and stride is not None:
			assert all(stride > 0 and type(item) is int for item in stride), 'stride must be positive integer'
		if type(padding) is not int and padding is not None:
			assert all(padding >= 0 and type(item) is int for item in padding), 'padding must be non-negative integer'
		if datatype is not None:
			assert any(datatype == item for item in ['uint', 'single', 'double'])
		else:
			datatype = 'single'
			print 'datatype of the layer is not defined. By default, we use single floating point to save the parameter'

		# set horizontal and vertical parameter as the same if only one dimentional input is obtained
		if type(kernal_size) is int:
			kernal_size = (kernal_size, kernal_size)
		if type(stride) is int:
			stride = (stride, stride)
		if type(padding) is int:
			padding = (padding, padding)

		self._kernal_size = kernal_size
		self._stride = stride
		self._padding = padding
		self._nInputPlane = nInputPlane
		self._nOutputPlane = nOutputPlane
		self._datatype = datatype
		self._name = name

	@property
	def name(self):
		return self._name

	@property
	def kernal_size(self):
		return self._kernal_size

	@property
	def stride(self):
		return self._stride

	@property
	def padding(self):
		return self._padding

	@property
	def nInputPlane(self):
		return self._nInputPlane

	@property
	def datatype(self):
		return self._datatype

	@property
	def nOutputPlane(self):
		return self._nOutputPlane

	@property
	def type(self):
 		raise NotImplementedError

	def get_output_blob_shape(self, bottom_shape):
 		assert len(bottom_shape) > 0, 'no bottom blob is obtained'
 		pass

 	def get_num_param(self):
 		raise NotImplementedError

 	def get_memory_usage_param(self):
 		if self.datatype == 'single':
 			return self.get_num_param() * 4 	# single has 4 bytes
 		elif self.datatype == 'double':
 			return self.get_num_param() * 8
 		elif self.datatype == 'uint':
 			return self.get_num_param()



class Convolution(Layer):
	'''
	define a 2d convolutional layer
	'''

	def __init__(self, name, nInputPlane, nOutputPlane, kernal_size, stride=None, padding=None, datatype=None):
		super(Convolution, self).__init__(name, nInputPlane, nOutputPlane, kernal_size, stride, padding, datatype)
		if stride is None:
			stride = (1, 1)
		if padding is None:
			padding = (0, 0)
		if datatype is None:
			datatype = 'single'

	@property
	def type(self):
 		return 'Convolution'

	def get_output_blob_shape(self, bottom_shape):
		assert len(bottom_shape) == 1 and len(bottom_shape[0]) == 3, 'bottom blob is not correct'
		assert False

	def get_num_param(self):
		return self.kernal_size[0] * self.kernal_size[1] * self.nInputPlane * self.nOutputPlane


  
class Pooling(Layer):
	'''
	define a 2d pooling layer
	'''

	def __init__(self, nInputPlane, nOutputPlane, kernal_size, stride=None, padding=None, datatype=None):
		super(Pooling, self).__init__(name, nInputPlane, nOutputPlane, kernal_size, stride, padding, datatype)
		if stride is None:
			stride = (1, 1)
		if padding is None:
			padding = (0, 0)
		if datatype is None:
			datatype = 'single'

	@property
	def type(self):
 		return 'Pooling'

	def get_output_blob_shape(self, bottom):
		assert len(bottom) == 1 and len(bottom[0]) == 3, 'bottom blob is not correct'
		assert False

	def get_num_param(self):
		return 0






