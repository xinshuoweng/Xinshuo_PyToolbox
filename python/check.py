# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import os, sys

def isstring(string):
	return isinstance(string, basestring)

def CHECK_EQ_LIST(input_list):
	'''
	check all elements in a list are equal
	'''
	assert isinstance(input_list, list), 'input is not a list'
	return input_list[1:] == input_list[:-1]