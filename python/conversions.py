# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file define a set of functions which converting data type
import numpy as np
from check import isstring, isinteger, islist

def character2onehot(character):
	'''
	In this function you need to output a one hot encoding of the ASCII character.
	'''
	assert isinteger(character) or isstring(character), 'input data type is not correct'
	
	# convert string to integer number
	if isstring(character):
		assert len(character) == 1, 'character should be a string with length 1'
		character = ord(character)

	one_hot_vec = np.zeros([self.nFeats, ], dtype='float32')
	one_hot_vec[character] = 1
	return one_hot_vec

def string2onehot(string):
	'''
	convert a string to a set of 2d one hot tensor
	'''
	assert isstring(string) and len(string) > 0, 'input should be a string with length larger than 0'
	
	one_hot_vec = [character2onehot(ord(string[0]))]
	for character in string[1:]:
		one_hot_vec = np.vstack((one_hot_vec, [character2onehot(ord(character))]))
	return one_hot_vec

def string2ord(string):
		'''
	convert a string to a list of ASCII character
	'''
	assert isstring(string) and len(string) > 0, 'input should be a string with length larger than 0'
	
	ord_list = []
	for character in string:
		ord_list.append(ord(character))
	return ord_list

def ord2string(ord_list):
		'''
	convert a list of ASCII character to a string
	'''
	assert islist(ord_list) and len(ord_list) > 0, 'input should be a list of ord with length larger than 0'
	assert all(isinteger(type(tmp)) for tmp in ord_list), 'all elements in the list of ord should be integer'
	
	L = ''
	for o in ord_list:
		L += chr(o)
	
	return L