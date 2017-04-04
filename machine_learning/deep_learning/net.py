# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from collections import OrderedDict
from operator import mul
import humanize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, time
import functools
from graphviz import Graph, Digraph
# import pydot

import __init__paths__
from check import isstring, CHECK_EQ_LIST
from plot import autopct_generator, fixOverLappingText
from layer import *
from file_io import is_path_exists_or_creatable, fileparts


Layer_supported = [Input, Convolution, Pooling, Dense, Activation, Concat]

# TODO: once layers with multiple blobs as output (e.g. split or slice layer) added, 
# compile, construct_graph, summary functions need to be modified

class Net(object):
	'''
	connect all layers to form a network 
	define blobs for parameters and data throughout all layers
	'''
	def __init__(self, inputlayers):
		if isinstance(inputlayers, Input):
			inputlayers = [inputlayers]
		else:
			assert isinstance(inputlayers, list) and len(inputlayers) > 0 \
				and all(isinstance(input_tmp, Input) for input_tmp in inputlayers), \
				'input layer for network container is not correct'

		self._inputlayers = inputlayers
		self._blobs = OrderedDict()
		self._layers = OrderedDict()

		# assign blobs to input layers 
		for input_tmp in inputlayers:
			assert not self._layers.has_key(input_tmp.name), 'input layer should not have same name'
			self._layers[input_tmp.name] = input_tmp
			self._blobs[input_tmp.name] = {'data': np.ndarray(input_tmp.inputshape, dtype='uint8'), 'params': None}

		self._nb_entries = len(self._inputlayers)
		self._compiled = False


	@property
	def inputlayers(self):
		return self._inputlayers

	@property
	def layers(self):
		return self._layers

	@property
	def nb_entries(self):
		return self._nb_entries

	@property
	def __len__(self):
		return self._nb_entries

	@property
	def blobs(self):
		assert self._compiled, 'the network is not compiled'
		return self._blobs

	@property
	def batch_size(self):
		assert self._compiled, 'the network is not compiled'
		return self._batch_size

	def add(self, layer):
		'''
		this function adds a layer to network container and return the layer for reference
		'''
		assert isinstance(layer, Layer), 'layer appended is not a valid Layer'
		assert any(isinstance(layer, item) for item in Layer_supported), \
			'The sequential model only support "%s" right now' % \
			functools.reduce(lambda x, y: str(x) + '" "' + str(y), Layer_supported)
		assert not self._layers.has_key(layer.name), 'layer name conflict'
		self._compiled = False

	def remove(self, layer_name):
		'''
		remove a layer based on the name of that layer
		this function must be overridden
		'''
		raise NotImplementedError

	def reshape_input(self, inputlayer_name):
		raise NotImplementedError

	def get_memory_usage_data_layer(self, layer_name):
		'''
		this function return memory usage for data given a specific layer
		batch_size is considered
		'''
		assert self._compiled, 'the network is not compiled'
		assert isstring(layer_name), 'the name of layer queried should be a string'
		assert self._layers.has_key(layer_name), 'no layer queried exists'
		
		# we treat activation layer as in-place layer and don't store data in it
		if isinstance(self._layers[layer_name], Activation):
			return 0

		datashape = (self._batch_size, ) + self._blobs[layer_name]['data'].shape
		layer = self._layers[layer_name]
		num_pixel = reduce(mul, datashape)
		if layer.datatype == 'single':
 			return num_pixel * 4 		# single has 4 bytes
 		elif layer.datatype == 'double':
 			return num_pixel * 8		# double has 8 bytes
 		elif layer.datatype == 'uint':
 			return num_pixel			# unsigned integer has 1 byte
		else:
			assert False, 'unknown error while calculating memory usage for data'


	def compile(self, input_data):
		'''
		assign data blobs for all layers except input layers which are already assigned 
		during initialization
		'''
		for layer_name, layer in self._layers.items():		# get data shape for all layers
			if isinstance(layer, Input):
				continue
			bottom_shape = []
			for bottom_layer in self._layers[layer_name].bottom:
				bottom_shape += [self._blobs[bottom_layer.name]['data'].shape]

			output_shape = layer.get_output_blob_shape(bottom_shape)[0]		# now all layers only output one output blob
			self._blobs[layer_name] = {'data': np.ndarray(output_shape, dtype='uint8'), 'params': None}

		self.set_input_data(input_data)
		assert len(self._blobs) == len(self._layers)
		self._compiled = True		# the network is ready to use


	def set_input_data(self, input_data):
		'''
		feed data to input data layer and get batch size
		one sequential model can only have one input
		'''
		if len(self._inputlayers) == 1:
			if isinstance(input_data, np.ndarray):
				input_data = [input_data]
			else:
				assert isinstance(input_data, list) and len(input_data) == 1, \
					'input data shape is not equal to the input layer shape'
		else:
			assert isinstance(input_data, list) and len(input_data) == len(self._inputlayers), \
				'input data shape is not equal to the input layer shape'

		index = 0
		batch_size_list = []
		for input_data_tmp in input_data:
			assert isinstance(input_data_tmp, np.ndarray), \
				'the input data should be numpy array'
			assert input_data_tmp.shape[1:] == self._blobs.values()[index]['data'].shape, \
				'the data feeding is not compatible with the network. ' \
				+ 'Please change the input data or reshape the Input layer'
			assert input_data_tmp.shape[0] > 0, 'batch size must be positive'
			index += 1
			batch_size_list += [input_data_tmp.shape[0]]

		assert CHECK_EQ_LIST(batch_size_list), 'batch size is not equal across all input layers'
		self._batch_size = batch_size_list[0]


	def construct_graph(self):
		'''
		this function return a graph object for the sequential model
		'''
		assert self._compiled, 'the network is not compiled'

		# construct the graph
		# graph = pydot.Dot(graph_type='graph')
		graph = Digraph(comment='Model Architecture')	

		# define nodes for all other layers and edges
		for layer_name, layer in self._layers.items():
			output_shape = self._blobs[layer_name]['data'].shape
			graph.node(layer_name, '"%s"\n%s (%d, %s)' % (layer_name, layer.type, self._batch_size, \
				functools.reduce(lambda x, y: str(x) + ', ' + str(y), output_shape)))
			# graph.node(layer_name, layer_name)
			# pydot.Node('"%s"\n%s\n(%d, %s)' % (layer_name, layer.type, self._batch_size, \
				# functools.reduce(lambda x, y: str(x) + ', ' + str(y), output_shape)))
			
			# graph.add_node(pydot.Node(layer_name))

			if layer.top is not None:
				for top_tmp in layer.top:
					graph.edge(layer_name, top_tmp.name)
					# graph.add_edge(pydot.Edge(layer_name, top_tmp.name))
		return graph


	def summary(self, visualize=False, model_path=None, table_path=None, chart_path=None, 
		display_threshold=4):
		'''
		print all info about the network to terminal and local path

		input parameter:
			model_name: a string for denoting the save file of model graph

			table_path: path to save network info for all layers including number of parameter, memory usage, output shape, layer name, layer type
			
			chart_path: path to save memory usage chart, which will display how much percentage of memory to take for each layer

			display_threshold: due to limited space of pie chart, we may not want to display the percentage number for some layers which only take a
							very small piece of pie. So, we suppress them by using threshold. For example, if the threshold is 5, percentage number
							lower than 5% will not be displayed 

			remove_threshold: we may also want to remove some layers totally from the pie chart, which takes up very little memory space and will overlap
							in the pie chart and make it ugly. So, we remove them by using this threshold. For example, if the threshold is 1, percentage
							lower than 1% will be removed 
		'''
		assert self._compiled, 'the network is not compiled'
		assert table_path is None or is_path_exists_or_creatable(table_path), \
			'table path is not correct'
		assert chart_path is None or is_path_exists_or_creatable(chart_path), \
			'chart path is not correct'
		assert model_path is None or is_path_exists_or_creatable(model_path), \
			'model path is not correct'	
		if table_path is None:
			table_path = 'model_table.txt'
		if chart_path is None:
			chart_path = 'memory_chart.png'
		if model_path is None:
			model_path = 'model_graph.pdf'


		# save the model graph
		graph = self.construct_graph()
		model_save_path, model_name, ext = fileparts(model_path)
		graph.format = ext[1:]
		graph.render(os.path.join(model_save_path, model_name), view=visualize, cleanup=True)
		# os.system('rm %s' % os.path.join(model_save_path, model_name))		# delete source string file produced by graphviz
		# graph.write_png(model_path)


		# print terminal network info to a table and file
		file_handler = open(table_path, 'w')
		print >> file_handler, 'Network Info Summary'
		print >> file_handler, '======================================================================================================='
		print >> file_handler, '{:<30}{:<25}{:<15}{:<20}'.format('Layer (type)', 'Output Shape', 'Param', 'Memory Usage(data, param)')
		print >> file_handler, '-------------------------------------------------------------------------------------------------------'
		total_param = 0
		memory_data_dict = dict()
		memory_param_dict = dict()
		previous_layer_shape = None
		for layer_name in self._blobs.keys():
			layer = self._layers[layer_name]
			output_shape = self._blobs[layer_name]['data'].shape
			layer_num_param = layer.get_num_param(previous_layer_shape)
			memory_data = self.get_memory_usage_data_layer(layer_name)
			memory_param = layer.get_memory_usage_param(previous_layer_shape)
			memory_data_dict[layer_name] = memory_data
			memory_param_dict[layer_name] = memory_param
			memory = memory_data + memory_param
			memory_data_format = humanize.naturalsize(memory_data, binary=True)
			memory_param_format = humanize.naturalsize(memory_param, binary=True)
			memory_format = humanize.naturalsize(memory, binary=True)

			# define a lambda symbolic function to format a string for connecting varying length of output shape
			print >> file_handler, '{:<30}{:<25}{:<15}{:<20}'.format('%s (%s)' % (layer_name, layer.type), \
				'(%d, %s)' % (self._batch_size, functools.reduce(lambda x, y: str(x) + ', ' + str(y), \
				output_shape)), layer_num_param, '%s (%s, %s)' % (memory_format, memory_data_format, \
				memory_param_format))
			print >> file_handler, '-------------------------------------------------------------------------------------------------------'
			total_param += layer_num_param
			previous_layer_shape = [output_shape]

		total_data_usage = sum(memory_data_dict.values())
		total_param_usage = sum(memory_param_dict.values())
		total_memory = dict()
		total_memory['data'] = total_data_usage
		total_memory['param'] = total_param_usage
		total_memory_usage = sum(total_memory.values())
		print >> file_handler, 'Total params: {:,}'.format(total_param)
		print >> file_handler, 'Total memory usage: {} (data: {}, param: {})'.format(humanize.naturalsize(total_memory_usage, binary=True), humanize.naturalsize(total_data_usage, binary=True), 
			humanize.naturalsize(total_param_usage, binary=True))
		print >> file_handler, '======================================================================================================='
		file_handler.close()

		# suppress the label below the threshold
		def suppress_labels(memory_dict, display_threshold):
			total_usage = sum(memory_dict.values())
			suppress_list = []
			for key, value in memory_dict.items():
				if value < display_threshold * total_usage / 100.:
					suppress_list.append(key)

			original_labels = memory_dict.keys()
			for candidate in suppress_list:
				index = original_labels.index(candidate)
				original_labels[index] = ''
			return original_labels

		# plot pie chart for memory usage
		fig = plt.figure()
		gs = gridspec.GridSpec(2, 2)
		fig.suptitle('Network Memory Usage', fontsize=16)
		
		def plot_pie_chart(ax, memory_dict, title, display_threshold):
			ax.set_title('%s: %s' % (title, humanize.naturalsize(sum(memory_dict.values()), binary=True)), 
				fontsize=10)
			max_index = np.argmax(np.array(memory_dict.values()))	# find the one to explode
			explode_list = [0] * len(memory_dict)
			explode_list[max_index] = 0.1
			plt.pie(memory_dict.values(), labels=suppress_labels(memory_dict, display_threshold), 
				autopct=autopct_generator(display_threshold), shadow=True, explode=explode_list)[1]		# suppress the number below the threshold
			# fixOverLappingText(text)
			plt.axis('equal')

		ax = plt.subplot(gs[0, 0])
		plot_pie_chart(ax, memory_data_dict, 'Data Memory Usage', display_threshold)
		ax = plt.subplot(gs[0, 1])
		plot_pie_chart(ax, memory_param_dict, 'Parameter Memory Usage', display_threshold)
		ax = plt.subplot(gs[1, :])
		plot_pie_chart(ax, total_memory, 'Total Memory Usage', display_threshold)
		plt.savefig(chart_path)
		if visualize:
			plt.show()
		plt.close()


		print ' '
		print 'Network info table is saved to %s' % os.path.abspath(table_path)
		print 'Network memory usage chart is saved to %s' % os.path.abspath(chart_path) 
		print 'Network model graph is saved to %s' % os.path.abspath(model_path) 



class Sequential(Net):
	def __init__(self, inputlayers):
		super(Sequential, self).__init__(inputlayers=inputlayers)

	def add(self, layer):
		super(Sequential, self).add(layer)

		if layer.bottom is None:
			layer.bottom_append(self._layers.values()[-1])	# append the last added layer as bottom automatically
		else:
			assert layer.bottom is [self._layers.values()[-1]] 	# if the bottom is specified explicitly, check if it's right for sequential model

		self._layers.values()[-1].top_append(layer)		# apeend the newly added layer as top of last layer
		self._layers[layer.name] = layer		
		self._nb_entries += 1
		return layer


	def remove(self, layer_name):
		assert isstring(layer_name), 'the layer should be queried by a string name'
		if self._blobs.has_key(layer_name):
			assert not isinstance(self._layers[layer_name], Input), \
				'the input layer is not able to delete. ' \
				'You might want to use reshape function to change the input shape.'

			# TODO: test if reference
			previous_layer = self._layers[layer_name].bottom[0]
			if self._layers[layer_name].top is not None:
				next_layer = self._layers[layer_name].top[0]
			else:
				next_layer = None

			self._layers[layer_name].bottom[0].top = next_layer
			self._layers[layer_name].top[0].bottom = previous_layer
			del self._blobs[layer_name]
			del self._layers[layer_name]
			self._nb_entries -= 1
		else:
			assert False, 'No layer queried existing'
		self._compiled = False


class gModule(Net):
	def __init__(self, inputlayers):
		super(gModule, self).__init__(inputlayers=inputlayers)

	def add(self, layer):
		super(gModule, self).add(layer)
		
		assert layer.bottom is not None and isinstance(layer.bottom, list) and len(layer.bottom) > 0 \
			and all(isinstance(bottom_tmp, AbstractLayer) for bottom_tmp in layer.bottom), \
			'bottom of a layer must be specified correctly when using gModule container'

		# construct the layer connection
		# Note: here we don't need append the bottom layer to the layer added right now as
		# what we do in the sequential model
		for bottom_tmp in layer.bottom:
			assert self._layers.has_key(bottom_tmp.name), 'bottom layer doesn\'t exist in the graph'
			self._layers[bottom_tmp.name].top_append(layer)	# apeend the newly added layer as top of last layer
			
		self._layers[layer.name] = layer
		self._nb_entries += 1
		return layer

	def remove(self, layer_name):
		# Note the tail of layer doesn't have top
		pass