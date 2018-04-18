# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import os, sys, numpy as np, pytest

import init_paths
from check import *

def test_isuintimage():
	image = np.zeros((100, 100), dtype='uint8')
	assert isuintimage(image) is True
	image = np.zeros((100, 100, 3), dtype='uint8')
	assert isuintimage(image) is True
	image = np.zeros((100, 100, 1), dtype='uint8')
	assert isuintimage(image) is True
	image = np.ones((100, 100, 1), dtype='uint8')
	assert isuintimage(image) is True

	image = np.zeros((100, 100), dtype='float32')
	assert isuintimage(image) is False
	image = np.zeros((100, 100, 4), dtype='uint8')
	assert isuintimage(image)

def test_isfloatimage():
	image = np.zeros((100, 100), dtype='float32')
	assert isfloatimage(image) is True
	image = np.zeros((100, 100, 3), dtype='float32')
	assert isfloatimage(image) is True
	image = np.zeros((100, 100, 1), dtype='float32')
	assert isfloatimage(image) is True
	image = np.ones((100, 100, 1), dtype='float32')
	assert isfloatimage(image) is True

	image = np.zeros((100, 100), dtype='uint8')
	assert isfloatimage(image) is False
	image = np.zeros((100, 100, 4), dtype='float32')
	assert isfloatimage(image)
	image = np.ones((100, 100, 3), dtype='float32')
	image[0, 0, 0] += 0.00001
	assert isfloatimage(image) is False

if __name__ == '__main__':
	pytest.main([__file__])