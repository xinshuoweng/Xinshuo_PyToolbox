# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import os, sys, numpy as np, pytest

import init_paths
from type_check import isstring, islist, islogical, isscalar, isnparray, istuple, isfunction, isdict, isext, isrange

def test_islist():
	input_test = []
	assert islist(input_test)
	input_test = ['']
	assert islist(input_test)
	input_test = [1]
	assert islist(input_test)
	input_test = [1, 2, 3]
	assert islist(input_test)
	input_test = [[], []]
	assert islist(input_test)
	input_test = list()
	assert islist(input_test)

	input_test = 123
	assert islist(input_test) is False
	input_test = False
	assert islist(input_test) is False
	input_test = dict()
	assert islist(input_test) is False
	input_test = 'ss'
	assert islist(input_test) is False
	input_test = np.array(('sss'))
	assert islist(input_test) is False
	input_test = ('syt')
	assert islist(input_test) is False

if __name__ == '__main__':
	pytest.main([__file__])