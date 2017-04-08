# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import os, sys

def isstring(string_test):
	return isinstance(string_test, basestring)

def isinteger(integer_test):
	return isinstance(integer_test, int)

def islist(list_test):
	return isinstance(list_test, list)

def CHECK_EQ_LIST(input_list):
	'''
	check all elements in a list are equal
	'''
	assert isinstance(input_list, list), 'input is not a list'
	return input_list[1:] == input_list[:-1]