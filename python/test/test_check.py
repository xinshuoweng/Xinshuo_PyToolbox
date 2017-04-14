# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import os, sys
import numpy as np
import pytest

import __init__paths__
from check import *

def test_is_path_valid():
	path = ''
	assert is_pathname_valid(path) is False
	path = 'test'
	assert is_pathname_valid(path)
	path = 123
	assert is_pathname_valid(path) is False
	path = 'test.txt'
	assert is_pathname_valid(path)

def test_is_path_creatable():
	path = ''
	assert is_path_creatable(path) is False
	path = 'test'
	assert is_path_creatable(path)
	path = 123
	assert is_path_creatable(path) is False
	path = 'test.txt'
	assert is_path_creatable(path)
	path = '/usr'
	assert is_path_creatable(path) is False

def test_is_path_exists():
	path = ''
	assert is_path_exists(path) is False
	path = 'test'
	assert is_path_exists(path) is False
	path = 123
	assert is_path_exists(path) is False
	path = 'test.txt'
	assert is_path_exists(path) is False
	path = '../test'
	assert is_path_exists(path)


if __name__ == '__main__':
	pytest.main([__file__])
	# test_is_path_valid()