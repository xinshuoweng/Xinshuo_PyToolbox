# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import os, sys
import numpy as np

def isstring(string_test):
	return isinstance(string_test, basestring)

def isinteger(integer_test):
	return isinstance(integer_test, int)

def islist(list_test):
	return isinstance(list_test, list)

def isnparray(nparray_test):
	return isinstance(nparray_test, np.ndarray)

def istuple(tuple_test):
	return isinstance(tuple_test, np.ndarray)

def is2dline(line_test):
	return (isnparray(line_test) or islist(line_test) or istuple(line_test)) and len(line_test) == 3

def is2dpts(pts_test):
	return (isnparray(pts_test) or islist(pts_test) or istuple(pts_test)) and len(pts_test) == 2

def CHECK_EQ_LIST(input_list):
	'''
	check all elements in a list are equal
	'''
	assert islist(input_list), 'input is not a list'
	return input_list[1:] == input_list[:-1]