# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from collections import OrderedDict
from operator import mul

import __init__
from type_check import isstring
from layer import *

class Net(object):
	'''
	connect all layers to form a network 
	define blobs for parameters and data throughout all layers
	'''
	def __init__(self):
		self._blobs = OrderedDict()
		self._layers = OrderedDict()
		self._nb_entries = 0
		self._compiled = False
		# if InputLayer is None:
		#  	nb_entries = 0
		# else:
		#  	assert isinstance(InputLayer, Input), 'the input layer is not valid'
		#  	nb_entries = 1
		#  	self._layers[InputLayer.name] = InputLayer
		#  	if InputData is None:
		#  		self._blobs[InputLayer.name] = {'data': None, 'params': None}

	@property
	def blobs(self):
		return self._blobs

	@property
	def layers(self):
		return self._layers

	@property
	def nb_entries(self):
		return self._nb_entries

	@property
	def __len__(self):
		return self._nb_entries


class SequentialNet(Net):
	def __init__(self):
		super(SequentialNet, self).__init__()

	# add one more layer to the network, only supporting sequential right now
	def append(self, layer):
		assert isinstance(layer, AbstractLayer), 'layer appended is not a valid Layer'
		assert not self._blobs.has_key(layer.name), 'layer name conflict'
		
		if self._nb_entries == 0:	# add the first input layer
			assert isinstance(layer, Input), 'First layer appended should be Input layer'		

		self._layers[layer.name] = layer			
		self._nb_entries += 1
		self._compiled = False

	def delete(self, layer_name):
		assert isstring(layer_name), 'the layer should be queried by a string name'
		if self._blobs.has_key(layer_name):
			assert isinstance(self._layers[layer_name], 'Input'), 'the input layer is not able to delete. You might want to use reshape function to change the input shape.'
			del self._blobs[layer_name]
			del self._layers[layer_name]
			self._nb_entries -= 1
		else:
			assert False, 'No layer queried existing'
		self._compiled = False

	def compile(self, input_data):
		assert self._nb_entries > 0, 'no layer existing'
		assert isinstance(self.layers.values()[0], Input), 'the first layer of network is not Input layer'
		assert len(self._blobs) == len(self._layers)

		# get the input layer
		inputlayer = self.layers.values()[0]
		self._blobs[inputlayer.name] = {'data': np.ndarray(inputlayer.inputshape), 'params': None}
		previous_layer_name = inputlayer.name
		for layer_name, layer in self._layers.items()[1:]:		# get data shape for all layers
			bottom_shape = self._blobs[previous_layer_name].data.shape
			output_shape = layer.get_output_blob_shape(bottom_shape)
			self._blobs[layer_name] = {'data': np.ndarray(output_shape), 'params': None}
			previous_layer_name = layer_name
		
		self.set_input_data(input_data)
		self._compiled = True		# the network is ready to use

	def set_input_data(self, input_data):
		assert isinstance(input_data, np.ndarray) and input_data.ndim > 1, 'the input data is not correct'
		assert input_data.shape[1:] == self.blobs.values()[0].input_data.shape, 'the data feeding is not compatible with the network. Please change the input data or reshape the Input layer'
		self._batch_size = input_data.shape[0]

	def __print__(self):
		assert self._compiled, 'the network is not compiled'
		print('Layer (type)\t\tOutput Shape\t\tParam\t\tMemory Usage(data, param)')
		print('============================================================================')
		total_param = 0
		total_memory = 0
		for layer_name in self._blobs.keys():
			layer = self._layers[layer_name]
			output_shape = self._blobs[layer_name].data.shape
			layer_num_param = layer.get_num_param()
			memory_data = self.get_memory_usage_data_layer(layer_name)
			memory_param = layer.get_memory_usage_param()
			memory = memory_data + memory_param
			print('{} ({})\t\t({}, {}, {})\t\t{}\t\t{}({}, {})'.format(layer_name, layer.type, output_shape[0], output_shape[1], output_shape[2], layer_num_param,
				memory, memory_data, memory_param))
			print('----------------------------------------------------------------------------')
			total_param += layer_num_param
			total_memory += memory
		print('============================================================================')
		print('Total params: {}'.format(total_param))

	def get_memory_usage_data_layer(self, layer_name):
		'''
		this function return memory usage for data given a specific layer
		batch_size is considered
		'''
		assert self._compiled, 'the network is not compiled'
		assert isstring(layer_name), 'the name of layer queried should be a string'
		assert self._layers.has_key(layer_name), 'no layer queried exists'
		
		datashape = (self._batch_size, ) + self._blobs[layer_name].data.shape
		layer = self._layers[layer_name]
		num_pixel = reduce(mul, datashape)
		if layer.paramtype == 'single':
 			return num_pixel * 4 		# single has 4 bytes
 		elif layer.paramtype == 'double':
 			return num_pixel * 8		# double has 8 bytes
 		elif layer.paramtype == 'uint':
 			return num_pixel			# unsigned integer has 1 byte
		else:
			assert False, 'unknown error while calculating memory usage for data'
