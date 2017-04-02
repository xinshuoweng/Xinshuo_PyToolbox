# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from collections import OrderedDict

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
		return self.nb_entries

	# add one more layer to the network, only supporting sequential right now
	def append(self, layer):
		assert isinstance(layer, AbstractLayer), 'layer appended is not a valid Layer'
		assert not self._blobs.has_key(layer.name), 'layer name conflict'
		
		if self._nb_entries == 0:	# add the first input layer
			assert isinstance(layer, Input), 'First layer appended should be Input layer'		
			self._blobs[layer.name] = {'data': np.ndarray(layer.inputshape), 'params': None}
		elif self._nb_entries > 0:
			previous_layer_name = self._layer.keys()[self._nb_entries - 1]
			bottom_shape = self._blobs[previous_layer_name].data.shape
			output_shape = layer.get_output_blob_shape(bottom_shape)
			self._blobs[layer.name] = {'data': np.ndarray(output_shape), 'params': None}
		
		self._layers[layer.name] = layer			
		self._nb_entries += 1

	def delete(self, layer_name):
		assert isstring(layer_name), 'the layer should be queried by a string name'
		if self._blobs.has_key(layer_name):
			del self._blobs[layer_name]
			del self._layers[layer_name]
			self._nb_entries -= 1
		else:
			assert False, 'No layer queried existing'

	def __print__(self):
		print('Layer (type)\t\tOutput Shape\t\tParam')
		print('============================================================================')
		total_param = 0
		for layer_name in self._blobs.keys():
			layer = self._layers[layer_name]
			if layer.type is 'Input':
				continue
			output_shape = self._blobs[layer_name].data.shape
			layer_num_param = layer.get_num_param()
			print('{} ({})\t\t({}, {}, {})\t\t{}'.format(layer_name, layer.type, output_shape[0], output_shape[1], output_shape[2], layer_num_param))
			total_param += layer_num_param
		# no layer existing right now
		print('============================================================================')
		print('Total params: {}'.format(total_param))

	def get_memory_usage(self):
		'''
		this function return memory usage of a given network
		'''
		bottom = [network[0]]

		for index in xrange(1, len(network)):
			layer = network[index]
			num_param = layer.get_num_param()
			memory_usage_param = layer.get_memory_usage_param()
			top = layer.get_output_blob(bottom)
