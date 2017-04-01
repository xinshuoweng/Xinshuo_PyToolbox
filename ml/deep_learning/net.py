# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from collections import OrderedDict

import __init__
from type_check import isstring
from layer import *

class Net(object):
	def __init__(self, InputLayer=None):
		self._blob = OrderedDict()
		if InputLayer is None:
		 	nb_entries = 0
		else:
		 	assert isinstance(InputLayer, Input), 'the input layer is not valid'
		 	nb_entries = 1
		 	self._blobs[InputLayer.name] = InputLayer

		self._nb_entries = nb_entries

	@property
	def blobs(self):
		return self._blobs

	@property
	def nb_entries(self):
		return self._nb_entries

	@property
	def __len__(self):
		return self.nb_entries

	# add one more layer to the network
	def append(self, layer):
		if self.nb_entries == 0:
			assert isinstance(layer, Input), 'First layer appended should be Input layer'
		assert isinstance(layer, Layer), 'layer appended is not a valid Layer'
		assert not self.blobs.has_key(layer.name), 'layer name conflict'
		self._nb_entries += 1
		self._blobs[layer.name] = layer

	def delete(self, layer_name):
		assert isstring(layer_name), 'the layer should be queried by a string name'
		if self.blobs.has_key(layer_name):
			del self.blobs[layer_name]
		else:
			assert False, 'No layer queried existing'

	def print(self):
		print('Layer (type)\tOutput Shape\tParam')
		for name, layer in self.blobs.items():
			print('{} ({})')

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
