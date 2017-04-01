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

	assert inputlayer.data is None
	data = np.array((100, 100, 3), dtype='float32')
	inputlayer.data = data
	assert_allclose(inputlayer.data, data)
	assert_allclose(inputlayer.get_output_blob_shape(), data.shape)


def Test_Convolution():
	convlayer = Convolution(name='conv1', nInputPlane=3, nOutputPlane=512, kernal_size=3)
	assert convlayer.name is 'conv1'
	assert_allclose(convlayer.nInputPlane, 3)
	assert_allclose(convlayer.nOutputPlane, 512)
	assert_allclose(convlayer.kernal_size, (3, 3)) 
	assert_allclose(convlayer.stride, 1)
	assert_allclose(convlayer.padding, 0)
	assert convlayer.datatype is 'single'
	assert convlayer.paramtype is 'single'
	assert convlayer.params is None
	
	convlayer = Convolution(name='conv1', nInputPlane=3, nOutputPlane=512, kernal_size=3, )
convlayer = Convolution(name='conv1', nInputPlane=3, nOutputPlane=512, kernal_size=3)
convlayer = Convolution(name='conv1', nInputPlane=3, nOutputPlane=512, kernal_size=3)
convlayer = Convolution(name='conv1', nInputPlane=3, nOutputPlane=512, kernal_size=3)



if __name__ == '__main__':
    pytest.main([__file__])
    # test_Input()