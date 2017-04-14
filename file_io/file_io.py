# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file contains a set of function for manipulating file io in python
import os, sys

import __init__paths__
from check import is_path_exists, isstring


def fileparts(pathname):
	'''
	this function return a tuple, which contains (directory, filename, extension)
	if the file has multiple extension, only last one will be displayed
	'''
	assert isstring(pathname), 'The input path is not a string'   
	directory = os.path.dirname(os.path.abspath(pathname))
	filename = os.path.splitext(os.path.basename(pathname))[0]
	ext = os.path.splitext(pathname)[1]
	return (directory, filename, ext)

def file_abspath():
    '''
    this function returns a absolute path for current file
    '''
    return os.path.dirname(os.path.abspath(__file__))

def load_list_from_file(file_path):
    '''
    this function reads list from a txt file
    '''
    assert is_path_exists(file_path), 'input path does not exist'
    _, _, extension = fileparts(file_path)
    assert extension == '.txt', 'File doesn''t have valid extension.'
    file = open(file_path, 'r')
    assert file != -1, 'datalist not found'

    fulllist = file.read().splitlines()
    num_elem = len(fulllist)
    file.close()

    return fulllist, num_elem