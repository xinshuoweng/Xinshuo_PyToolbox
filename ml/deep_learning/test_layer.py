# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np
import pytest
from numpy.testing import assert_allclose

from layer import *

def test_Input():
	inputlayer = Input(name='data')
	assert inputlayer.name is 'data'
	assert inputlayer.type is 'Input'
	assert_allclose(inputlayer.get_num_param(), 0)
	assert inputlayer.datatype is 'single'
	assert inputlayer.paramtype is 'single'
	# assert inputlayer.data is None
	# assert inputlayer.params is None

	inputlayer = Input(name='data', datatype='double')
	assert inputlayer.datatype is 'double'

	inputlayer = Input(name='data', datatype='double', paramtype='uint')
	assert inputlayer.paramtype is 'uint'

	# inputlayer.data = data
	# assert_allclose(inputlayer.data, data)
	data = np.array((100, 100, 3), dtype='float32')
	assert_allclose(inputlayer.get_output_blob_shape(data), data.shape)


def test_Convolution():
	# parameter shape test
	convlayer = Convolution(name='conv1', nOutputPlane=512, kernal_size=(3,4), stride=2)
	assert_allclose(convlayer.kernal_size, (3, 4))
	assert_allclose(convlayer.stride, (2, 2))
	convlayer = Convolution(name='conv1', nOutputPlane=512, kernal_size=3, stride=(3,4), padding=1)
	assert_allclose(convlayer.stride, (3, 4))
	assert_allclose(convlayer.padding, (1, 1))
	# params = np.ndarray(())
	convlayer = Convolution(name='conv1', nOutputPlane=512, kernal_size=3, stride=2, padding=(3,4))
	assert_allclose(convlayer.padding, (3, 4))

	# main test
	convlayer = Convolution(name='conv1', nOutputPlane=512, kernal_size=(3, 4), stride=(2, 3), padding=(1, 2))
	assert convlayer.name is 'conv1'
	# assert_allclose(convlayer.nInputPlane, 3)
	assert_allclose(convlayer.nOutputPlane, 512)
	assert_allclose(convlayer.kernal_size, (3, 4)) 
	assert_allclose(convlayer.stride, (2, 3))
	assert_allclose(convlayer.padding, (1, 2))
	assert convlayer.datatype is 'single'
	assert convlayer.paramtype is 'single'
	# assert convlayer.params is None
	# assert convlayer.data is None
	assert_allclose(convlayer.get_num_param(), 3*3*4*512)
	bottom_shape = [(4, 12, 3)]
	assert_allclose(convlayer.get_output_blob_shape(bottom_shape), (2, 5, 512))

def test_Pooling():
	# parameter shape test
	poolinglayer = Pooling(name='pool1', kernal_size=(3,4), stride=2)
	assert_allclose(poolinglayer.kernal_size, (3, 4))
	assert_allclose(poolinglayer.stride, (2, 2))
	poolinglayer = Pooling(name='pool1', kernal_size=3, stride=(3,4), padding=1)
	assert_allclose(poolinglayer.stride, (3, 4))
	assert_allclose(poolinglayer.padding, (1, 1))
	# params = np.ndarray(())
	poolinglayer = Pooling(name='pool1', kernal_size=3, stride=2, padding=(3,4))
	assert_allclose(poolinglayer.padding, (3, 4))

	# main test
	poolinglayer = Pooling(name='pool1', kernal_size=(3, 4), stride=(2, 3), padding=(1, 2))
	assert poolinglayer.name is 'pool1'
	assert_allclose(poolinglayer.kernal_size, (3, 4)) 
	assert_allclose(poolinglayer.stride, (2, 3))
	assert_allclose(poolinglayer.padding, (1, 2))
	assert poolinglayer.datatype is 'single'
	assert poolinglayer.paramtype is 'single'
	# assert convlayer.params is None
	# assert convlayer.data is None
	assert_allclose(poolinglayer.get_num_param(), 0)
	bottom_shape = [(4, 12, 512)]
	assert_allclose(poolinglayer.get_output_blob_shape(bottom_shape), (2, 5, 512))


if __name__ == '__main__':
    pytest.main([__file__])
    # test_Input()