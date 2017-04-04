# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np
import pytest
from numpy.testing import assert_allclose

from layer import *

def test_Input():
	# parameter shape test
	inputlayer = Input(name='data', inputshape=(100, 100, 3))
	assert inputlayer.name is 'data'
	assert inputlayer.type is 'Input'
	assert_allclose(inputlayer.inputshape, (100, 100, 3))
	assert_allclose(inputlayer.get_num_param(), 0)
	assert inputlayer.datatype is 'single'
	assert inputlayer.paramtype is 'single'
	assert inputlayer.bottom is None
	assert inputlayer.top is None

	# test type
	inputlayer = Input(name='data', inputshape=(100, 100, 3), datatype='double')
	assert inputlayer.datatype is 'double'
	inputlayer.datatype = 'uint'
	assert inputlayer.datatype is 'uint'
	inputlayer = Input(name='data', inputshape=(100, 100, 3), datatype='double', paramtype='uint')
	assert inputlayer.paramtype is 'uint'
	inputlayer.paramtype = 'double'
	assert inputlayer.paramtype is 'double'

	# test output shape
	data = np.array((100, 100, 3), dtype='float32')
	assert_allclose(inputlayer.get_output_blob_shape(data), [data.shape])


def test_Convolution():
	# parameter shape test
	convlayer = Convolution(name='conv1', nOutputPlane=512, kernal_size=(3,4), stride=2)
	assert_allclose(convlayer.kernal_size, (3, 4))
	assert_allclose(convlayer.stride, (2, 2))
	convlayer = Convolution(name='conv1', nOutputPlane=512, kernal_size=3, stride=(3,4), padding=1)
	assert_allclose(convlayer.stride, (3, 4))
	assert_allclose(convlayer.padding, (1, 1))
	convlayer = Convolution(name='conv1', nOutputPlane=512, kernal_size=3, stride=2, padding=(3,4))
	assert_allclose(convlayer.padding, (3, 4))

	# convolution test
	convlayer = Convolution(name='conv1', nOutputPlane=512, kernal_size=(3, 4), stride=(2, 3), 
		padding=(1, 2))
	assert convlayer.name is 'conv1'
	assert convlayer.bottom is None
	assert convlayer.top is None
	assert_allclose(convlayer.nOutputPlane, 512)
	assert_allclose(convlayer.kernal_size, (3, 4)) 
	assert_allclose(convlayer.stride, (2, 3))
	assert_allclose(convlayer.padding, (1, 2))
	assert convlayer.datatype is 'single'
	assert convlayer.paramtype is 'single'
	bottom_shape = [(4, 12, 3)]
	assert_allclose(convlayer.get_num_param(bottom_shape), 3*3*4*512 + 512)
	assert_allclose(convlayer.get_output_blob_shape(bottom_shape), [(2, 5, 512)])
	
	# test top related operation
	conv_test = Convolution(name='conv_test', nOutputPlane=512, kernal_size=(3, 4))
	convlayer.top = conv_test
	assert len(convlayer.top) == 1 and convlayer.top[0] is conv_test
	convlayer.top_append(conv_test)
	assert len(convlayer.top) == 2 and convlayer.top[1] is conv_test
	convlayer.top_append([conv_test, conv_test])	
	assert len(convlayer.top) == 4 and all(top_tmp is conv_test for top_tmp in convlayer.top)

	# test bottom layer operation
	convlayer = Convolution(name='conv1', bottom=conv_test, nOutputPlane=512, kernal_size=(3, 4), 
		stride=(2, 3), padding=(1, 2))
	assert len(convlayer.bottom) == 1 and convlayer.bottom[0] is conv_test
	# convlayer = Convolution(name='conv1', bottom=[conv_test, conv_test], nOutputPlane=512, kernal_size=(3, 4), 
	# 	stride=(2, 3), padding=(1, 2))
	convlayer.bottom = None
	assert convlayer.bottom is None
	convlayer.bottom_append(conv_test)
	assert len(convlayer.bottom) == 1 and convlayer.bottom[0] is conv_test
	convlayer.bottom = [conv_test]
	assert len(convlayer.bottom) == 1 and convlayer.bottom[0] is conv_test
	convlayer.bottom = [conv_test]
	assert len(convlayer.bottom) == 1 and convlayer.bottom[0] is conv_test


def test_Pooling():
	# parameter shape test
	poolinglayer = Pooling(name='pool1', kernal_size=(3,4), stride=2)
	assert_allclose(poolinglayer.kernal_size, (3, 4))
	assert_allclose(poolinglayer.stride, (2, 2))
	poolinglayer = Pooling(name='pool1', kernal_size=3, stride=(3,4), padding=1)
	assert_allclose(poolinglayer.stride, (3, 4))
	assert_allclose(poolinglayer.padding, (1, 1))
	poolinglayer = Pooling(name='pool1', kernal_size=3, stride=2, padding=(3,4))
	assert_allclose(poolinglayer.padding, (3, 4))

	# pooling test
	poolinglayer = Pooling(name='pool1', kernal_size=(3, 4), stride=(2, 3), padding=(1, 2))
	assert poolinglayer.name is 'pool1'
	assert_allclose(poolinglayer.kernal_size, (3, 4)) 
	assert_allclose(poolinglayer.stride, (2, 3))
	assert_allclose(poolinglayer.padding, (1, 2))
	assert poolinglayer.datatype is 'single'
	assert poolinglayer.paramtype is 'single'
	bottom_shape = [(4, 12, 512)]
	assert_allclose(poolinglayer.get_num_param(bottom_shape), 0)
	assert_allclose(poolinglayer.get_output_blob_shape(bottom_shape), [(2, 5, 512)])


def test_Dense():
	denselayer = Dense(name='dense1', nOutputPlane=512)
	bottom_shape = [(4, 12, 3)]
	assert_allclose(denselayer.get_num_param(bottom_shape), (4 * 12 * 3 + 1) * 512)
	assert_allclose(denselayer.get_output_blob_shape(bottom_shape), [(512, )])


def test_Activation():
	activation = Activation(name='activation1', function='sigmoid')
	bottom_shape = [(4, 12, 3)]
	assert_allclose(activation.get_num_param(bottom_shape), 0)
	assert_allclose(activation.get_output_blob_shape(bottom_shape), bottom_shape)


def test_Concat():
	concat = Concat(name='concat1', axis=0)
	assert concat.axis == 0
	bottom_shape = [(4, 12, 3), (3, 12, 3), (3, 12, 3)]
	assert_allclose(concat.get_num_param(bottom_shape), 0)
	assert_allclose(concat.get_output_blob_shape(bottom_shape), [(10, 12, 3)])


if __name__ == '__main__':
    pytest.main([__file__])