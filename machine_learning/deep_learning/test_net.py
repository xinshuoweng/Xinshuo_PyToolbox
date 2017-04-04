import numpy as np
import pytest
from numpy.testing import assert_allclose

from layer import *
from net import Sequential, gModule


def test_Sequential():
	# create model
	model = Sequential(inputlayers=Input(name='data', inputshape=(384, 256, 3)))
	assert_allclose(model.nb_entries, 1)
	
	model.add(Convolution(name='conv1', nOutputPlane=128, kernal_size=3, padding=1))
	model.add(Pooling(name='pool1', kernal_size=2, stride=2))
	model.compile(np.ndarray([16, 384, 256, 3]))
	assert_allclose(model.nb_entries, 3)
	assert_allclose(model.blobs['data']['data'].shape, (384, 256, 3))
	assert_allclose(model.blobs['conv1']['data'].shape, (384, 256, 128))
	assert_allclose(model.blobs['pool1']['data'].shape, (192, 128, 128))
	# model.summary()
	
	# test delete
	model.remove('conv1')
	model.compile(np.ndarray((16, 384, 256, 3)))
	assert_allclose(model.nb_entries, 2)
	assert_allclose(model.blobs['pool1']['data'].shape, (192, 128, 3))
	assert not model.layers.has_key('conv1')
	assert not model.blobs.has_key('conv1')


def test_gModule():
	# test input layer
	inputlayer1 = Input(name='input', inputshape=(384, 256, 3), datatype='uint')
	model = gModule(inputlayers=inputlayer1)
	assert model.layers['input'].top is None
	assert model.layers['input'].bottom is None
	model.compile(np.ndarray((16, 384, 256, 3)))
	assert_allclose(model.batch_size, 16)

	conv1 = model.add(Convolution(name='conv1', bottom=inputlayer1, nOutputPlane=64, kernal_size=3, 
		padding=1))
	assert model.layers['input'].top == [model.layers['conv1']]
	assert model.layers['conv1'].bottom == [inputlayer1]

	model.add(Convolution(name='conv2_1', bottom=conv1, nOutputPlane=128, kernal_size=3, padding=1))
	assert model.layers['conv1'].top == [model.layers['conv2_1']]
	assert model.layers['conv2_1'].bottom == [model.layers['conv1']]
	
	model.add(Convolution(name='conv2_2', bottom=conv1, nOutputPlane=128, kernal_size=3, padding=1))
	assert model.layers['conv1'].top == [model.layers['conv2_1'], model.layers['conv2_2']]
	assert model.layers['conv2_2'].bottom == [model.layers['conv1']]

	model.compile(np.ndarray((16, 384, 256, 3)))
	assert_allclose(model.nb_entries, 4)
	assert_allclose(model.batch_size, 16)

	# model.remove('conv2_2')
	# assert not model.layers.has_key('conv1')
	# assert not model.blobs.has_key('conv1')
	# model.compile(np.ndarray((16, 384, 256, 3)))
	# assert_allclose(model.nb_entries, 3)

	# model.summary()

if __name__ == '__main__':
    pytest.main([__file__])

