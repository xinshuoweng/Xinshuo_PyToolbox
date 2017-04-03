# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np
import sys
from operator import mul

import __init__paths__
from type_check import isstring

class AbstractLayer(object):
	'''
	define an abstract layer for all type of layers
	'''
	def __init__(self, name, bottom=None, datatype=None, paramtype=None):
		if datatype is not None:
			assert any(datatype == item for item in ['uint', 'single', 'double']), 'type of data should be one of ''uint8'' ''single'' ''double'' '
		else:
			datatype = 'single'
			print 'datatype of the layer is not defined. By default, we use single floating point to save the data'
		if paramtype is not None:
			assert any(paramtype == item for item in ['uint', 'single', 'double']), 'type of parameter should be one of ''uint8'' ''single'' ''double'' '
		else:
			paramtype = 'single'
			print 'paramtype of the layer is not defined. By default, we use single floating point to save the parameter'
		assert isstring(name), 'the name of input layer should be a string'	
		if bottom is not None:
			assert len(bottom) > 0 and all(isinstance(layer_tmp, AbstractLayer) for layer_tmp in bottom), 'bottom layer is not correct'

		self._bottom = bottom
		self._name = name
		# self._data = None
		# self._params = None
		self._datatype = datatype
		self._paramtype = paramtype

	@property
	def name(self):
		return self._name

	@name.setter
	def name(self, name):
		assert isstring(name), 'the name of a layer should be a string'

	@property
	def bottom(self):
		return self._bottom

	# @property
	# def data(self):
 # 		return self._data

	# @data.setter
	# def data(self, data):
 # 		raise NotImplementedError

 # 	@property
	# def params(self):
 # 		return self._params

	# @params.setter
	# def params(self, params):
 # 		raise NotImplementedError

	@property
	def paramtype(self):
 		return self._paramtype

	@paramtype.setter
	def paramtype(self, paramtype):
 		assert isstring(paramtype), 'the type of parameter should be a string'
 		assert any(paramtype is item for item in ['uint', 'single', 'double']), 'type of parameter should be one of ''uint8'' ''single'' ''double'' '
 		self._paramtype = paramtype

	@property
	def datatype(self):
 		return self._datatype

	@datatype.setter
	def datatype(self, datatype):
 		assert isstring(datatype), 'the type of data should be a string'
 		assert any(datatype is item for item in ['uint', 'single', 'double']), 'type of data should be one of ''uint8'' ''single'' ''double'' '
 		self._datatype = datatype


	@property
	def type(self):
 		raise NotImplementedError

 	def get_num_param(self, bottom_shape=None):
 		raise NotImplementedError

 	def get_output_blob_shape(self, bottom_shape=None):
 		raise NotImplementedError

	def get_memory_usage_param(self, bottom_shape=None):
 		if self._paramtype == 'single':
 			return self.get_num_param(bottom_shape) * 4 	# single has 4 bytes
 		elif self._paramtype == 'double':
 			return self.get_num_param(bottom_shape) * 8		# double has 8 bytes
 		elif self._paramtype == 'uint':
 			return self.get_num_param(bottom_shape)			# unsigned integer has 1 byte

class Input(AbstractLayer):
	'''
	define an input layer which contains info about the input data
	'''
	def __init__(self, name, inputshape, datatype=None, paramtype=None):
		super(Input, self).__init__(name=name, datatype=datatype, paramtype=paramtype)
		assert isinstance(inputshape, tuple) and len(inputshape) > 0, 'the input shape should be a tuple'
		self._inputshape = inputshape
		# assert isinstance(data, np.ndarray), 'the input data layer should contains numpy array'
		# self._data = data
	# @AbstractLayer.data.setter
	# def data(self, data):
	# 	assert isinstance(data, np.ndarray), 'the input data layer should contains numpy array'
	# 	self._data = data

	# @AbstractLayer.params.setter
	# def params(self, params):
	# 	assert False, 'No parameter can be set in the input layer'

	@property
	def inputshape(self):
 		return self._inputshape

	@property
	def type(self):
 		return 'Input'

 	def get_num_param(self, bottom_shape=None):
 		assert bottom_shape is None, 'No bottom layer before Input layer'
 		return 0

 	def get_output_blob_shape(self, data):
 		assert isinstance(data, np.ndarray), 'the input data layer should contains numpy array'
 		return [data.shape]

class Layer(AbstractLayer):
	'''
	define necessary layer parameter and property for deep learning
	parameters are following HxW format
	'''
	def __init__(self, name, bottom=None, nOutputPlane=None, kernal_size=None, stride=None, padding=None, datatype=None, paramtype=None):
		super(Layer, self).__init__(name=name, bottom=bottom, datatype=datatype, paramtype=paramtype)
		# assert nInputPlane is None or (type(nInputPlane) is int and nInputPlane > 0), 'number of input channel is not correct'
		assert nOutputPlane is None or (type(nOutputPlane) is int and nOutputPlane > 0), 'number of output channel is not correct'
		assert kernal_size is None or type(kernal_size) is int or len(kernal_size) == 2, 'kernal size is not correct'
		assert stride is None or type(stride) is int or len(stride) == 2, 'stride size is not correct'
		assert padding is None or type(padding) is int or len(padding) == 2, 'padding size is not correct'
		# assert params is None or isinstance(params, np.ndarray), 'parameter is not correct'

		if type(kernal_size) is not int and kernal_size is not None:
			assert all(item > 0 and type(item) is int for item in kernal_size), 'kernal size must be positive integer'
		if type(stride) is not int and stride is not None:
			assert all(stride > 0 and type(item) is int for item in stride), 'stride must be positive integer'
		if type(padding) is not int and padding is not None:
			assert all(padding >= 0 and type(item) is int for item in padding), 'padding must be non-negative integer'

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
		# self._nInputPlane = nInputPlane
		self._nOutputPlane = nOutputPlane
		# self._params = params

	# @property
	# def nInputPlane(self):
	# 	return self._nInputPlane

	@property
	def nOutputPlane(self):
		return self._nOutputPlane

	@property
	def kernal_size(self):
		return self._kernal_size

	@property
	def stride(self):
		return self._stride

	@property
	def padding(self):
		return self._padding

class Convolution(Layer):
	'''
	define a 2d convolutional layer
	'''
	def __init__(self, name, nOutputPlane, kernal_size, bottom=None, stride=None, padding=None, datatype=None, paramtype=None):
		super(Convolution, self).__init__(name=name, bottom=bottom, nOutputPlane=nOutputPlane, kernal_size=kernal_size, 
			stride=stride, padding=padding, datatype=datatype, paramtype=paramtype)
		# assert params.ndim == 3, 'the parameter of convolution layer should be 3-d array'
		# assert params.shape(0) == self.nInputPlane, 'first dimension of parameter in convolution layer is not correct'
		# self._params = params
		if self._stride is None:
			self._stride = (1, 1)
		if self._padding is None:
			self._padding = (0, 0)

	# @AbstractLayer.data.setter
	# def data(self, data):
	# 	assert isinstance(data, np.ndarray), 'the data of convolution layer should contains numpy array'
	# 	assert data.ndim == 4, 'the data of convolution layer should be 4-d array'
	# 	assert data.shape(3) == self.nOutputPlane, 'last dimension of data in convolution layer is not correct'
	# 	self._data = data

	# @AbstractLayer.params.setter
	# def params(self, params):
	# 	assert isinstance(params, np.ndarray), 'the parameter of convolution layer should contains numpy array'
	# 	assert params.ndim == 4, 'the parameter of convolution layer should be 3-d array'
	# 	assert params.shape(0) == self.nInputPlane, 'first dimension of parameter in convolution layer is not correct'
	# 	assert params.shape() == self.nInputPlane, 'first dimension of parameter in convolution layer is not correct'
	# 	self._params = params

	@property
	def type(self):
 		return 'Convolution'

	def get_num_param(self, bottom_shape):
		assert len(bottom_shape) == 1 and len(bottom_shape[0]) == 3 and isinstance(bottom_shape[0], tuple), 'bottom shape is not correct'
		num_weights = self.kernal_size[0] * self.kernal_size[1] * bottom_shape[0][-1] * self.nOutputPlane
		num_bias = self._nOutputPlane
		return num_weights + num_bias

	def get_output_blob_shape(self, bottom_shape):
		assert len(bottom_shape) == 1 and len(bottom_shape[0]) == 3 and isinstance(bottom_shape[0], tuple), 'bottom shape is not correct'
		return [tuple((np.array(bottom_shape[0][0:2]) + 2*np.array(self.padding) - np.array(self.kernal_size)) / np.array(self.stride) + 1) + (self.nOutputPlane, )]

  
class Pooling(Layer):
	'''
	define a 2d pooling layer
	'''
	def __init__(self, name, kernal_size, bottom=None, stride=None, padding=None, datatype=None, paramtype=None):
		super(Pooling, self).__init__(name=name, bottom=bottom, kernal_size=kernal_size, stride=stride, padding=padding, datatype=datatype, paramtype=paramtype)
		if self._stride is None:
			self._stride = (1, 1)
		if self._padding is None:
			self._padding = (0, 0)

	# @AbstractLayer.data.setter
	# def data(self, data):
	# 	# assert isinstance(data, np.ndarray), 'the data of convolution layer should contains numpy array'
	# 	assert data.ndim == 4, 'the data of convolution layer should be 4-d array'
	# 	assert data.shape(3) == self.nOutputPlane, 'last dimension of data in convolution layer is not correct'
	# 	self._data = data

	# @AbstractLayer.params.setter
	# def params(self, params):
	# 	# assert isinstance(params, np.ndarray), 'the parameter of convolution layer should contains numpy array'
	# 	assert params.ndim == 3, 'the parameter of convolution layer should be 3-d array'
	# 	assert params.shape(0) == self.nInputPlane, 'first dimension of parameter in convolution layer is not correct'
	# 	self._params = params

	@property
	def type(self):
 		return 'Pooling'

	def get_num_param(self, bottom_shape=None):
		return 0

	def get_output_blob_shape(self, bottom_shape):
		assert len(bottom_shape) == 1 and len(bottom_shape[0]) == 3 and isinstance(bottom_shape[0], tuple), 'bottom shape is not correct'
		return [tuple((np.array(bottom_shape[0][0:2]) + 2*np.array(self.padding) - np.array(self.kernal_size)) / np.array(self.stride) + 1) + (bottom_shape[0][2], )]



class Dense(Layer):
	'''
	define a fully connected layer
	'''
	def __init__(self, name, nOutputPlane, bottom=None, datatype=None, paramtype=None):
		super(Dense, self).__init__(name=name, bottom=bottom, nOutputPlane=nOutputPlane, datatype=datatype, paramtype=paramtype)

	@property
	def type(self):
 		return 'Dense'

	def get_num_param(self, bottom_shape):
		assert len(bottom_shape) == 1 and len(bottom_shape[0]) > 0 and isinstance(bottom_shape[0], tuple), 'bottom shape is not correct'
		num_weights = reduce(mul, bottom_shape[0]) * self.nOutputPlane
		num_bias = self.nOutputPlane
		return num_weights + num_bias

	def get_output_blob_shape(self, bottom_shape=None):
		if bottom_shape is not None:
			assert len(bottom_shape) == 1 and len(bottom_shape[0]) > 0 and isinstance(bottom_shape[0], tuple), 'bottom shape is not correct'
		return [(self.nOutputPlane, )]



class Activation(AbstractLayer):
	'''
	define a fully connected layer
	'''
	def __init__(self, name, function, bottom=None, datatype=None, paramtype=None):
		super(Activation, self).__init__(name=name, bottom=bottom, datatype=datatype, paramtype=paramtype)
		assert isstring(function), 'the function used in dense layer should be a string'
		assert any(function is item for item in ['linear', 'relu', 'sigmoid', 'tanh']), 'type of parameter should be one of ''linear'' ''relu'' ''tanh'' ''sigmoid'' '
		self._function = function

	@property
	def type(self):
 		return 'Activation'

 	@property
	def function(self):
 		return self._function

	def get_num_param(self, bottom_shape=None):
		if bottom_shape is not None:
			assert len(bottom_shape) == 1 and len(bottom_shape[0]) > 0 and isinstance(bottom_shape[0], tuple), 'bottom shape is not correct'
		return 0

	def get_output_blob_shape(self, bottom_shape):
		assert len(bottom_shape) == 1 and len(bottom_shape[0]) > 0 and isinstance(bottom_shape[0], tuple), 'bottom shape is not correct'
		return bottom_shape






