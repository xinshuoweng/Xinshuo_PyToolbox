# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from collections import OrderedDict
from operator import mul
import humanize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import __init__
from type_check import isstring
from plot import autopct_generator, fixOverLappingText
from layer import *
from file_io import is_path_exists_or_creatable

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
		assert self._compiled, 'the network is not compiled'
		return self._blobs

	@property
	def layers(self):
		assert self._compiled, 'the network is not compiled'
		return self._layers

	@property
	def nb_entries(self):
		assert self._compiled, 'the network is not compiled'
		return self._nb_entries

	@property
	def __len__(self):
		assert self._compiled, 'the network is not compiled'
		return self._nb_entries

	@property
	def batch_size(self):
		assert self._compiled, 'the network is not compiled'
		return self._batch_size

class SequentialNet(Net):
	def __init__(self):
		super(SequentialNet, self).__init__()

	# add one more layer to the network, only supporting sequential right now
	def append(self, layer):
		assert isinstance(layer, AbstractLayer), 'layer appended is not a valid Layer'
		assert not self._layers.has_key(layer.name), 'layer name conflict'
		
		if self._nb_entries == 0:	# add the first input layer
			assert isinstance(layer, Input), 'First layer appended should be Input layer'		

		self._layers[layer.name] = layer			
		self._nb_entries += 1
		self._compiled = False

	def delete(self, layer_name):
		assert isstring(layer_name), 'the layer should be queried by a string name'
		if self._blobs.has_key(layer_name):
			assert not isinstance(self._layers[layer_name], Input), 'the input layer is not able to delete. You might want to use reshape function to change the input shape.'
			del self._blobs[layer_name]
			del self._layers[layer_name]
			self._nb_entries -= 1
		else:
			assert False, 'No layer queried existing'
		self._compiled = False

	def compile(self, input_data):
		'''
		assign data to all layers
		'''
		assert self._nb_entries > 0, 'no layer existing'
		assert isinstance(self._layers.values()[0], Input), 'the first layer of network is not Input layer'

		# get the input layer
		inputlayer = self._layers.values()[0]
		self._blobs[inputlayer.name] = {'data': np.ndarray(inputlayer.inputshape), 'params': None}
		previous_layer_name = inputlayer.name
		for layer_name, layer in self._layers.items()[1:]:		# get data shape for all layers
			bottom_shape = [self._blobs[previous_layer_name]['data'].shape]
			output_shape = layer.get_output_blob_shape(bottom_shape)
			self._blobs[layer_name] = {'data': np.ndarray(output_shape), 'params': None}
			previous_layer_name = layer_name
		
		self.set_input_data(input_data)
		assert len(self._blobs) == len(self._layers)
		self._compiled = True		# the network is ready to use

	def set_input_data(self, input_data):
		'''
		feed data to input data layer and get batch size
		'''
		assert isinstance(input_data, np.ndarray) and input_data.ndim > 1, 'the input data is not correct'
		assert input_data.shape[1:] == self._blobs.values()[0]['data'].shape, 'the data feeding is not compatible with the network. Please change the input data or reshape the Input layer'
		assert all(isinstance(tmp, int) is True for tmp in input_data.shape), 'input shape should be a tuple of integer'
		assert input_data.shape[0] > 0, 'batch size must be positive'
		self._batch_size = input_data.shape[0]

	def summary(self, savepath=None):
		assert self._compiled, 'the network is not compiled'
		assert savepath is None or is_path_exists_or_creatable(savepath), 'save path is not correct'
		# print terminal network info
		print('=======================================================================================================')
		print('{:<30}{:<25}{:<15}{:<20}'.format('Layer (type)', 'Output Shape', 'Param', 'Memory Usage(data, param)'))
		print('-------------------------------------------------------------------------------------------------------')
		total_param = 0
		memory_data_list = dict()
		memory_param_list = dict()
		previous_layer_shape = None
		for layer_name in self._blobs.keys():
			layer = self._layers[layer_name]
			output_shape = self._blobs[layer_name]['data'].shape
			layer_num_param = layer.get_num_param(previous_layer_shape)
			memory_data = self.get_memory_usage_data_layer(layer_name)
			memory_param = layer.get_memory_usage_param(previous_layer_shape)
			memory_data_list[layer_name] = memory_data
			memory_param_list[layer_name] = memory_param
			memory = memory_data + memory_param
			memory_data_format = humanize.naturalsize(memory_data, binary=True)
			memory_param_format = humanize.naturalsize(memory_param, binary=True)
			memory_format = humanize.naturalsize(memory, binary=True)
			print('{:<30}{:<25}{:<15}{:<20}'.format('%s (%s)' % (layer_name, layer.type), '(%d, %d, %d, %d)' % (self._batch_size, output_shape[0], output_shape[1], output_shape[2]),
				layer_num_param, '%s (%s, %s)' % (memory_format, memory_data_format, memory_param_format)))
			print('-------------------------------------------------------------------------------------------------------')
			total_param += layer_num_param
			previous_layer_shape = [output_shape]

		total_data_usage = sum(memory_data_list.values())
		total_param_usage = sum(memory_param_list.values())
		total_memory = dict()
		total_memory['data'] = total_data_usage
		total_memory['param'] = total_param_usage
		total_memory_usage = sum(total_memory.values())
		print('Total params: {}'.format(total_param))
		print('Total memory usage: {}'.format(humanize.naturalsize(total_memory_usage, binary=True)))
		print('=======================================================================================================')

		# remove small percentage during display
		remove_threshold = 0.01
		for key, value in memory_data_list.items():
			if value < remove_threshold * total_data_usage:
				del memory_data_list[key]
		for key, value in memory_param_list.items():
			if value < remove_threshold * total_param_usage:
				del memory_param_list[key]
		for key, value in total_memory.items():
			if value < remove_threshold * total_memory_usage:
				del total_memory[key]

		# plot pie chart for memory usage
		threshold_display_percentage = 5
		fig = plt.figure()
		gs = gridspec.GridSpec(2, 2)
		fig.suptitle('Network Memory Usage', fontsize=16)
		ax = plt.subplot(gs[0, 0])
		ax.set_title('Data Memory Usage: %s' % humanize.naturalsize(total_data_usage, binary=True), fontsize=10)
		max_index = np.argmax(np.array(memory_data_list.values()))	# find the one to explode
		explode_list = [0] * len(memory_data_list)
		explode_list[max_index] = 0.1
		text = plt.pie(memory_data_list.values(), labels=memory_data_list.keys(), autopct=autopct_generator(threshold_display_percentage), shadow=True, explode=explode_list)[1]
		# fixOverLappingText(text)
		plt.axis('equal')
		ax = plt.subplot(gs[0, 1])
		ax.set_title('Parameter Memory Usage: %s' % humanize.naturalsize(total_param_usage, binary=True), fontsize=10)
		max_index = np.argmax(np.array(memory_param_list.values()))	# find the one to explode
		explode_list = [0] * len(memory_param_list)
		explode_list[max_index] = 0.1
		text = plt.pie(memory_param_list.values(), labels=memory_param_list.keys(), autopct=autopct_generator(threshold_display_percentage), shadow=True, explode=explode_list)[1]
		# fixOverLappingText(text)
		plt.axis('equal')
		ax = plt.subplot(gs[1, :])
		ax.set_title('Total Memory Usage: %s' % humanize.naturalsize(total_memory_usage, binary=True), fontsize=10)
		max_index = np.argmax(np.array(total_memory.values()))	# find the one to explode
		explode_list = [0] * len(total_memory)
		explode_list[max_index] = 0.1
		text = plt.pie(total_memory.values(), labels=total_memory.keys(), autopct=autopct_generator(threshold_display_percentage), shadow=True, explode=explode_list)[1]
		# fixOverLappingText(text)
		plt.axis('equal')
		plt.savefig(savepath)
		plt.show()


	def get_memory_usage_data_layer(self, layer_name):
		'''
		this function return memory usage for data given a specific layer
		batch_size is considered
		'''
		assert self._compiled, 'the network is not compiled'
		assert isstring(layer_name), 'the name of layer queried should be a string'
		assert self._layers.has_key(layer_name), 'no layer queried exists'
		
		datashape = (self._batch_size, ) + self._blobs[layer_name]['data'].shape
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
