# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import os, sys
import numpy as np
import pytest

import __init__paths__
from check import *

def test_is_path_valid():
	path = ''
	assert is_path_valid(path) is False
	path = './'
	assert is_path_valid(path)
	path = 'test'
	assert is_path_valid(path)
	path = 123
	assert is_path_valid(path) is False
	path = 'test.txt'
	assert is_path_valid(path)

def test_is_path_creatable():
	path = ''
	assert is_path_creatable(path) is False
	path = './'
	assert is_path_creatable(path)
	path = 'test'
	assert is_path_creatable(path)
	path = 123
	assert is_path_creatable(path) is False
	path = 'test.txt'
	assert is_path_creatable(path)
	path = '/usr'
	assert is_path_creatable(path) is False
	path = '/mnt/sdc1/xinshuow/dataset/mnist/mnist/train/images/image.jpg'
	assert is_path_creatable(path)
	path = '/mnt/sdc1/xinshuow/dataset/mnist/mnist/train/images/image1/image2/image3'
	assert is_path_creatable(path)

def test_is_path_exists():
	path = ''
	assert is_path_exists(path) is False
	path = './'
	assert is_path_exists(path)
	path = 'test'
	assert is_path_exists(path) is False
	path = 123
	assert is_path_exists(path) is False
	path = 'test.txt'
	assert is_path_exists(path) is False
	path = '../test'
	assert is_path_exists(path)
	path = '/mnt/sdc1/xinshuow/dataset/mnist/mnist/train/images/image0000001.jpg'
	assert is_path_exists(path) is False

def test_is_path_exists_or_creatable():
	path = ''
	assert is_path_exists_or_creatable(path) is False
	path = './'
	assert is_path_exists_or_creatable(path)
	path = 'test'
	assert is_path_exists_or_creatable(path)
	path = 123
	assert is_path_exists_or_creatable(path) is False
	path = 'test.txt'
	assert is_path_exists_or_creatable(path)
	path = '../test'
	assert is_path_exists_or_creatable(path)
	path = '/mnt/sdc1/xinshuow/dataset/mnist/mnist/train/images/image0000001.jpg'
	assert is_path_exists_or_creatable(path)

if __name__ == '__main__':
	pytest.main([__file__])
	# test_is_path_valid()