# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np
import sys
from operator import mul
import functools

import __init__paths__
from check import isstring, CHECK_EQ_LIST


DATATYPE = ['uint', 'single', 'double', 'boolean']
PARAMTYPE = ['uint', 'single', 'double', 'boolean']
ACTIVATION_FUNCTION = ['linear', 'relu', 'sigmoid', 'tanh']


class AbstractLayer(object):
	'''
	define an abstract layer for all type of layers
	'''
	def __init__(self, name, datatype=None, paramtype=None):
		if datatype is not None:
			self._datatype_check(datatype)
		else:
			datatype = 'single'
			print 'datatype of the layer is not defined.' \
				'By default, we use single floating point to save the data'
		if paramtype is not None:
			self._paramtype_check(paramtype)
		else:
			paramtype = 'single'
			print 'paramtype of the layer is not defined. By default,' \
				'we use single floating point to save the parameter'
		assert isstring(name), 'the name of input layer should be a string'	

		self._name = name
		# self._data = None
		# self._params = None
		self._datatype = datatype
		self._paramtype = paramtype
		self._top = None
		self._bottom = None

	@property
	def name(self):
		return self._name

	# @name.setter
	# def name(self, name):
	# 	assert isstring(name), 'the name of a layer should be a string'
	# 	self._name = name

	@property
	def top(self):
		'''
		all layers can have multiple top layers
		'''
		return self._top

	@top.setter
	def top(self, top):
		self._top = self._top_check(top)

	def top_append(self, top):
		top = self._top_check(top)
		if self._top is None:
			self._top = top
		else:
			if top is None:
				top = []
			self._top = self._top + top   	# append top layer to the existing top layers

	def _top_check(self, top):
		if isinstance(top, AbstractLayer):
			top = [top]	
		else:
			assert top is None or isinstance(top, list) and len(top) > 0 \
				and all(isinstance(layer_tmp, Layer) for layer_tmp in top), \
				'top layer is not correct'		# top layer cannot be inputlayer
		return top


	@property
	def bottom(self):
		'''
		different may have different number of bottom layers
		'''
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
		self._paramtype_check(paramtype)
 		self._paramtype = paramtype

 	def _paramtype_check(self, paramtype):
 		assert isstring(paramtype), 'the type of parameter should be a string'
 		assert any(paramtype is item for item in PARAMTYPE), \
 			'type of parameter should be one of "%s"' \
 			% functools.reduce(lambda x, y: str(x) + '" "' + str(y), PARAMTYPE)

	@property
	def datatype(self):
 		return self._datatype

	@datatype.setter
	def datatype(self, datatype):
		self._datatype_check(datatype)
 		self._datatype = datatype

 	def _datatype_check(self, datatype):
 		assert isstring(datatype), 'the type of data should be a string'
 		assert any(datatype is item for item in DATATYPE), \
 			'type of data should be one of "%s"' \
 			% functools.reduce(lambda x, y: str(x) + '" "' + str(y), DATATYPE)


	@property
	def type(self):
 		raise NotImplementedError

 	def get_num_param(self, bottom_shape=None):
 		'''
		the bottom shape is a list of tuple
 		'''
 		raise NotImplementedError

 	def get_output_blob_shape(self, bottom_shape=None):
 		'''
		this function returns a list of tuple
 		'''
 		raise NotImplementedError

	def get_memory_usage_param(self, bottom_shape=None):
		'''
		this function calculate memory usage for the layer
		Note that we also consider the diff variable in each layer.
		So the memory usage should be double size
		'''
		num_param = 2 * self.get_num_param(bottom_shape)
 		if self._paramtype == 'single':
 			return num_param * 4 		# single has 4 bytes
 		elif self._paramtype == 'double':
 			return num_param * 8		# double has 8 bytes
 		elif self._paramtype == 'uint':
 			return num_param			# unsigned integer has 1 byte
 		elif self._paramtype == 'boolean':
 			return num_param 			# boolean has 1 byte
 		else:
 			assert False, 'Unknown parameter datatype error'

class Input(AbstractLayer):
	'''
	define an input layer which contains info about the input data
	'''
	def __init__(self, name, inputshape, datatype=None, paramtype=None):
		super(Input, self).__init__(name=name, datatype=datatype, paramtype=paramtype)
		assert isinstance(inputshape, tuple) and len(inputshape) > 0, \
			'the input shape should be a tuple'
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
	define layer except Input layer
	the bottom layer is not needed to be a list as input, it will be handled here.
	'''
	def __init__(self, name, nOutputPlane=None, bottom=None, datatype=None, paramtype=None):
		super(Layer, self).__init__(name=name, datatype=datatype, paramtype=paramtype)
		assert nOutputPlane is None or (type(nOutputPlane) is int and nOutputPlane > 0), \
			'number of output channel is not correct'
		self._nOutputPlane = nOutputPlane			
		self._bottom = self._bottom_check(bottom)


	@property
	def nOutputPlane(self):
		return self._nOutputPlane

	@AbstractLayer.bottom.setter
	def bottom(self, bottom):
		self._bottom = self._bottom_check(bottom)

	def bottom_append(self, bottom):
		bottom = self._bottom_check(bottom)
		if self._bottom is None:
			self._bottom = bottom
		else:
			if bottom is None:
				bottom = []
			self._bottom = self._bottom + bottom

	def _bottom_check(self, bottom):
		if isinstance(bottom, AbstractLayer):
			bottom = [bottom]	
		else:
			assert bottom is None or (isinstance(bottom, list) and len(bottom) > 0 \
				and all(isinstance(layer_tmp, AbstractLayer) for layer_tmp in bottom)), \
				'bottom layer is not correct'		# bottom layer could be any layer

		return bottom

	def _bottom_shape_check(self, bottom_shape):
		assert bottom_shape is not None, 'bottom shape cannot be none'
		assert len(bottom_shape) > 0 and isinstance(bottom_shape, list) \
			and all(isinstance(bottom_shape_tmp, tuple) for bottom_shape_tmp in bottom_shape), \
			'bottom shape is not correct'

class SpatialLayer(Layer):
	'''
	define necessary layer parameter and property for deep learning
	parameters are following HxW format
	'''
	def __init__(self, name, bottom=None, nOutputPlane=None, kernal_size=None, stride=None, 
		padding=None, datatype=None, paramtype=None):
		super(SpatialLayer, self).__init__(name=name, nOutputPlane=nOutputPlane, bottom=bottom, 
			datatype=datatype, paramtype=paramtype)
		assert kernal_size is None or type(kernal_size) is int or len(kernal_size) == 2, \
			'kernal size is not correct'
		assert stride is None or type(stride) is int or len(stride) == 2, \
			'stride size is not correct'
		assert padding is None or type(padding) is int or len(padding) == 2, \
			'padding size is not correct'
		# assert params is None or isinstance(params, np.ndarray), 'parameter is not correct'

		if type(kernal_size) is not int and kernal_size is not None:
			assert all(item > 0 and type(item) is int for item in kernal_size), \
				'kernal size must be positive integer'
		if type(stride) is not int and stride is not None:
			assert all(stride > 0 and type(item) is int for item in stride), \
				'stride must be positive integer'
		if type(padding) is not int and padding is not None:
			assert all(padding >= 0 and type(item) is int for item in padding), \
				'padding must be non-negative integer'

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

		# self._params = params

	# @property
	# def nInputPlane(self):
	# 	return self._nInputPlane

	@property
	def kernal_size(self):
		return self._kernal_size

	@property
	def stride(self):
		return self._stride

	@property
	def padding(self):
		return self._padding

	def _bottom_shape_check(self, bottom_shape):
		'''
		Note that spatial layer can only accept bottom with 3 dimension
		'''
		super(SpatialLayer, self)._bottom_shape_check(bottom_shape)
		assert all(len(bottom_shape_tmp) == 3 for bottom_shape_tmp in bottom_shape), \
			'bottom shape is not correct'

class Convolution(SpatialLayer):
	'''
	define a 2d convolutional layer
	'''
	def __init__(self, name, nOutputPlane, kernal_size, bottom=None, stride=None, padding=None, 
		datatype=None, paramtype=None):
		super(Convolution, self).__init__(name=name, bottom=bottom, nOutputPlane=nOutputPlane, 
			kernal_size=kernal_size, stride=stride, padding=padding, datatype=datatype, 
			paramtype=paramtype)
		# assert params.ndim == 3, 'the parameter of convolution layer should be 3-d array'
		# assert params.shape(0) == self.nInputPlane, 'first dimension of parameter in convolution layer is not correct'
		# self._params = params
		if self._stride is None:
			self._stride = (1, 1)
		if self._padding is None:
			self._padding = (0, 0)
		assert self._bottom is None or len(self._bottom) == 1, \
			'Convolution layer can only have one bottom layer'
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

	@AbstractLayer.bottom.setter
	def bottom(self, bottom):
		self._bottom = self._bottom_check(bottom)
		assert self._bottom is None or len(self._bottom) == 1, 'Convolution layer can at most have one bottom layer'

	def bottom_append(self, bottom):
		super(Convolution, self).bottom_append(bottom)
		assert self._bottom is None or len(self._bottom) == 1, 'Convolution layer can at most have one bottom layer'

	def get_num_param(self, bottom_shape):
		self._bottom_shape_check(bottom_shape)
		assert len(bottom_shape) == 1, 'Convolution layer can at most have one bottom layer'
		
		num_weights = self.kernal_size[0] * self.kernal_size[1] * bottom_shape[0][-1] \
			* self.nOutputPlane
		num_bias = self._nOutputPlane
		return num_weights + num_bias

	def get_output_blob_shape(self, bottom_shape):
		self._bottom_shape_check(bottom_shape)
		assert len(bottom_shape) == 1, 'Convolution layer can at most have one bottom layer'

		return [tuple((np.array(bottom_shape[0][0:2]) + 2*np.array(self.padding) \
			- np.array(self.kernal_size)) / np.array(self.stride) + 1) + (self.nOutputPlane, )]

  
class Pooling(SpatialLayer):
	'''
	define a 2d pooling layer
	'''
	def __init__(self, name, kernal_size, bottom=None, stride=None, padding=None, 
		datatype=None, paramtype=None):
		super(Pooling, self).__init__(name=name, bottom=bottom, kernal_size=kernal_size, 
			stride=stride, padding=padding, datatype=datatype, paramtype=paramtype)
		if self._stride is None:
			self._stride = (1, 1)
		if self._padding is None:
			self._padding = (0, 0)
		assert self._bottom is None or len(self._bottom) == 1, \
			'Pooling layer can only have one bottom layer'

	@property
	def type(self):
 		return 'Pooling'

	@AbstractLayer.bottom.setter
	def bottom(self, bottom):
		self._bottom = self._bottom_check(bottom)
		assert self._bottom is None or len(self._bottom) == 1, 'Pooling layer can at most have one bottom layer'

	def bottom_append(self, bottom):
		super(Pooling, self).bottom_append(bottom)
		assert self._bottom is None or len(self._bottom) == 1, 'Pooling layer can at most have one bottom layer'

	def get_num_param(self, bottom_shape):
		self._bottom_shape_check(bottom_shape)
		assert len(bottom_shape) == 1, 'Pooling layer can at most have one bottom layer'

		return 0

	def get_output_blob_shape(self, bottom_shape):
		self._bottom_shape_check(bottom_shape)
		assert len(bottom_shape) == 1, 'Pooling layer can at most have one bottom layer'

		return [tuple((np.array(bottom_shape[0][0:2]) + 2*np.array(self.padding) 
			- np.array(self.kernal_size)) / np.array(self.stride) + 1) + (bottom_shape[0][2], )]


class Dense(Layer):
	'''
	define a fully connected layer
	'''
	def __init__(self, name, nOutputPlane, bottom=None, datatype=None, paramtype=None):
		super(Dense, self).__init__(name=name, bottom=bottom, nOutputPlane=nOutputPlane, 
			datatype=datatype, paramtype=paramtype)
		assert self._bottom is None or len(self._bottom) == 1, 'Dense layer can at most have one bottom layer'


	@AbstractLayer.bottom.setter
	def bottom(self, bottom):
		self._bottom = self._bottom_check(bottom)
		assert self._bottom is None or len(self._bottom) == 1, 'Dense layer can at most have one bottom layer'

	def bottom_append(self, bottom):
		super(Dense, self).bottom_append(bottom)
		assert self._bottom is None or len(self._bottom) == 1, 'Dense layer can at most have one bottom layer'

	@property
	def type(self):
 		return 'Dense'

	def get_num_param(self, bottom_shape):
		self._bottom_shape_check(bottom_shape)
		assert len(bottom_shape) == 1, 'Dense layer can at most have one bottom layer'

		num_weights = reduce(mul, bottom_shape[0]) * self.nOutputPlane
		num_bias = self.nOutputPlane
		return num_weights + num_bias

	def get_output_blob_shape(self, bottom_shape):
		if bottom_shape is not None:
			self._bottom_shape_check(bottom_shape)
			assert len(bottom_shape) == 1, 'Dense layer can at most have one bottom layer'

		return [(self.nOutputPlane, )]



class Activation(Layer):
	'''
	define a fully connected layer
	'''
	def __init__(self, name, function, bottom=None, datatype=None, paramtype=None):
		super(Activation, self).__init__(name=name, bottom=bottom, datatype=datatype, 
			paramtype=paramtype)
		assert isstring(function), 'the function used in dense layer should be a string'
		assert any(function is item for item in ACTIVATION_FUNCTION), \
			'type of parameter should be one of "%s"' \
			% functools.reduce(lambda x, y: str(x) + '" "' + str(y), ACTIVATION_FUNCTION)
		assert self._bottom is None or len(self._bottom) == 1, \
			'Activation layer can only have one bottom layer'

		self._function = function

	@property
	def type(self):
 		return 'Activation'
	
	@AbstractLayer.bottom.setter
	def bottom(self, bottom):
		self._bottom = self._bottom_check(bottom)
		assert self._bottom is None or len(self._bottom) == 1, 'Activation layer can at most have one bottom layer'

	def bottom_append(self, bottom):
		super(Activation, self).bottom_append(bottom)
		assert self._bottom is None or len(self._bottom) == 1, 'Activation layer can at most have one bottom layer'

 	@property
	def function(self):
 		return self._function

	def get_num_param(self, bottom_shape):
		self._bottom_shape_check(bottom_shape)
		assert len(bottom_shape) == 1, 'Activation layer can at most have one bottom layer'

		return 0

	def get_output_blob_shape(self, bottom_shape):
		self._bottom_shape_check(bottom_shape)
		assert len(bottom_shape) == 1, 'Activation layer can at most have one bottom layer'

		return bottom_shape


class Concat(Layer):
	'''
	define a concat layer, which can concatenate several layers along specific dimension
	Note: it's a litte bit different than Caffe, since the axis doesn't include the first 
	batch dimension. So if the input data is (N)xHxWxC, axis 1 means to concatenate along W dimension
	'''
	def __init__(self, name, axis, bottom=None, datatype=None, paramtype=None):
		super(Concat, self).__init__(name=name, bottom=bottom, datatype=datatype, 
			paramtype=paramtype)
		assert isinstance(axis, int) and axis >= 0, 'axis for concatenation should be an non-negative integer'
		self._axis = axis

	@property
	def type(self):
 		return 'Concat'
	
 	@property
 	def axis(self):
 		return self._axis

	@AbstractLayer.bottom.setter
	def bottom(self, bottom):
		self._bottom = self._bottom_check(bottom)

	def bottom_append(self, bottom):
		super(Concat, self).bottom_append(bottom)

	def _bottom_shape_check(self, bottom_shape):
		super(Concat, self)._bottom_shape_check(bottom_shape)

		dimension = len(bottom_shape[0])
		bottom_shape_list_removal = []
		for bottom_shape_tmp in bottom_shape:
			assert len(bottom_shape_tmp) == dimension, 'all bottom layer during concatenation should have same dimension'

			# construct a list of list without the dimension along the axis
			bottom_shape_list_tmp = list(bottom_shape_tmp)
			del bottom_shape_list_tmp[self._axis]
			
			bottom_shape_list_removal.append(bottom_shape_list_tmp)
		assert CHECK_EQ_LIST(bottom_shape_list_removal), 'bottom shape should be equal for all bottom layers except for the dimension to concatenate'

	def get_num_param(self, bottom_shape):
		self._bottom_shape_check(bottom_shape)
		return 0

	def get_output_blob_shape(self, bottom_shape):
		self._bottom_shape_check(bottom_shape)	

		axis_list = []
		for bottom_shape_tmp in bottom_shape:
			axis_list += [bottom_shape_tmp[self._axis]]

		# sum the dimension along the axis
		top_shape = list(bottom_shape[0])
		top_shape[self._axis] = sum(axis_list)
		return [tuple(top_shape)]




