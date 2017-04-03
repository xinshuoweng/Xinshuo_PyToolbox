import numpy as np
import pytest
from numpy.testing import assert_allclose

from layer import *
from net import Sequential


def test_SequentialNet():
	# create network
	network = Sequential()
	network.add(Input(name='data', inputshape=(384, 256, 3)))
	network.add(Convolution(name='conv1', nOutputPlane=128, kernal_size=3, padding=1))
	network.add(Pooling(name='pool1', kernal_size=2, stride=2))
	network.compile(np.ndarray((16, 384, 256, 3)))
	assert_allclose(network.nb_entries, 3)
	assert_allclose(network.blobs['data']['data'].shape, (384, 256, 3))
	assert_allclose(network.blobs['conv1']['data'].shape, (384, 256, 128))
	assert_allclose(network.blobs['pool1']['data'].shape, (192, 128, 128))
	network.summary()
	
	# test delete
	network.remove('conv1')
	network.compile(np.ndarray((16, 384, 256, 3)))
	assert_allclose(network.nb_entries, 2)
	assert_allclose(network.blobs['pool1']['data'].shape, (192, 128, 3))
	assert not network.layers.has_key('conv1')
	assert not network.blobs.has_key('conv1')

if __name__ == '__main__':
    pytest.main([__file__])
