# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file contains a set of function for manipulating file io in python
import os, sys
import glob

import __init__paths__
from check import is_path_exists, isstring, is_path_exists_or_creatable, isfile, isfolder


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


def load_list_from_folder(folder_path, ext_filter=None):
    '''
    load a list of files from a system folder
    '''
    assert isfolder(folder_path) and is_path_exists(folder_path), 'input folder path is not correct'
    if ext_filter is not None:
        assert isstring(ext_filter), 'extension filder is not correct'
        fulllist = glob.glob(os.path.join(folder_path, '*%s' % ext_filter))
    else:
        fulllist = glob.glob(folder_path)
    num_elem = len(fulllist)
    
    return fulllist, num_elem


def mkdir_if_missing(pathname):
    if isfile(pathname):
        pathname, _, _ = fileparts(pathname)
    assert is_path_exists_or_creatable(pathname), 'input path is not valid or creatable'
    if not is_path_exists(pathname):
        os.mkdir(pathname)


def generate_list_from_folder(save_path, src_path, ext_filter=None):
    assert isfolder(src_path) and is_path_exists(src_path), 'source folder not found or incorrect'
    if not isfile(save_path):
        assert isfolder(save_path), 'save path is not correct'
        save_path = os.path.join(save_path, 'datalist.txt')

    if ext_filter is not None:
        assert isstring(ext_filter), 'extension filter is not correct'

    filepath = file_abspath()
    datalist_name = save_path
    cmd = 'th %s/../file_io/generate_list.lua %s %s %s' % (filepath, src_path, datalist_name, ext_filter)
    os.system(cmd)    # generate data list