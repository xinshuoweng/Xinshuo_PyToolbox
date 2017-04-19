# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file contains a set of function for manipulating file io in python
import os, sys
import glob
import numpy as np
# from PIL import Image
from scipy.misc import imsave
import time

import __init__paths__
from check import is_path_exists, isstring, is_path_exists_or_creatable, isfile, isfolder, isnparray, is_path_creatable, is_path_valid, safepath, islist, isimage, isgrayimage

def fileparts(pathname, debug=True):
	'''
	this function return a tuple, which contains (directory, filename, extension)
	if the file has multiple extension, only last one will be displayed
	'''
	pathname = safepath(pathname, debug=debug)
	if len(pathname) == 0:
		return ('', '', '')
	if pathname[-1] == '/':
		if len(pathname) > 1:
			return (pathname[:-1], '', '')	# ignore the final '/'
		else:
			return (pathname, '', '')	# ignore the final '/'
	directory = os.path.dirname(os.path.abspath(pathname))
	filename = os.path.splitext(os.path.basename(pathname))[0]
	ext = os.path.splitext(pathname)[1]
	return (directory, filename, ext)


def load_list_from_file(file_path):
    '''
    this function reads list from a txt file
    '''
    file_path = safepath(file_path)
    _, _, extension = fileparts(file_path)
    assert extension == '.txt', 'File doesn''t have valid extension.'
    file = open(file_path, 'r')
    assert file != -1, 'datalist not found'

    fulllist = file.read().splitlines()
    fulllist = [os.path.normpath(path_tmp) for path_tmp in fulllist]
    num_elem = len(fulllist)
    file.close()

    return fulllist, num_elem


def load_list_from_folder(folder_path, ext_filter=None):
    '''
    load a list of files from a system folder
    '''
    folder_path = safepath(folder_path)
    assert isfolder(folder_path) and is_path_exists(folder_path), 'input folder path is not correct %s' % folder_path
    if ext_filter is not None:
        assert isstring(ext_filter), 'extension filder is not correct'
        fulllist = glob.glob(os.path.join(folder_path, '*%s' % ext_filter))
    else:
        fulllist = glob.glob(os.path.join(folder_path, '*'))

    fulllist = [os.path.normpath(path_tmp) for path_tmp in fulllist]
    num_elem = len(fulllist)
    
    return fulllist, num_elem


def mkdir_if_missing(pathname):
	pathname = safepath(pathname)
	assert is_path_exists_or_creatable(pathname), 'input path is not valid or creatable: %s' % pathname
	dirname, _, _ = fileparts(pathname)

	if not is_path_exists(dirname):
		mkdir_if_missing(dirname)

	if isfolder(pathname) and not is_path_exists(pathname):
		os.mkdir(pathname)


def generate_list_from_folder(save_path, src_path, ext_filter='jpg'):
	save_path = safepath(save_path)
	src_path = safepath(src_path)
	assert isfolder(src_path) and is_path_exists(src_path), 'source folder not found or incorrect'
	if not isfile(save_path):
		assert isfolder(save_path), 'save path is not correct'
		save_path = os.path.join(save_path, 'datalist.txt')

	if ext_filter is not None:
		assert isstring(ext_filter), 'extension filter is not correct'

	filepath = os.path.dirname(os.path.abspath(__file__))
	cmd = 'th %s/generate_list.lua %s %s %s' % (filepath, src_path, save_path, ext_filter)
	os.system(cmd)    # generate data list


def generate_list_from_data(save_path, src_data, debug=True):
    '''
    generate a file which contains a 1-d numpy array data

    parameter:
        src_data:   a list of 1 element data, or a 1-d numpy array data
    '''
    save_path = safepath(save_path)

    if debug:
        if isnparray(src_data):
            assert src_data.ndim == 1, 'source data is incorrect'
        elif islist(src_data):
            assert all(np.array(data_tmp).size == 1 for data_tmp in src_data), 'source data is in correct'
        assert isfolder(save_path) or isfile(save_path), 'save path is not correct'
        
    if isfolder(save_path):
        save_path = os.path.join(save_path, 'datalist.txt')

    if debug:
        assert is_path_exists_or_creatable(save_path), 'the file cannot be created'

    with open(save_path, 'w') as file:
        for item in src_data:
            file.write('%f\n' % item)


def save_image_from_data(save_path, data, debug=True, vis=False):
    save_path = safepath(save_path)
    if debug:
        assert isimage(data), 'input data is not image format'
        assert is_path_exists_or_creatable(save_path), 'save path is not correct'
        mkdir_if_missing(save_path)

    imsave(save_path, data)