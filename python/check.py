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

def isfunction(func_test):
	return callable(func_test)

def isdict(dict_test):
    return isinstance(dict_test, dict)

def istuple(tuple_test):
    return isinstance(tuple_test, tuple)

def iscolorimage(image_test):
    if not isnparray(image_test):
        return False

    return image_test.ndim == 3 and image_test.shape[2] == 3

def isgrayimage(image_test):
    if not isnparray(image_test):
        return False

    return image_test.ndim == 2 or (image_test.ndim == 3 and image_test.shape[2] == 1)

def isimage(image_test):
    return iscolorimage(image_test) or isgrayimage(image_test)

def is_path_valid(pathname):
    '''
    `True` if the passed pathname is a valid pathname for the current OS;
    `False` otherwise.
    '''
    # If this pathname is either not a string or is but is empty, this pathname
    # is invalid.
    try:
        if not isstring(pathname) or not pathname:
            return False
    except TypeError:
        return False
    else:
        return True

def is_path_creatable(pathname):
    '''
    `True` if the current user has sufficient permissions to create the passed
    pathname; `False` otherwise.
    '''
    if is_path_valid(pathname) is False:
    	return False
    dirname = os.path.dirname(os.path.abspath(pathname))
    return os.access(dirname, os.W_OK)

def is_path_exists_or_creatable(pathname):
    '''
	this function is to justify is given path existing or creatable
    '''
    try:
        return is_path_valid(pathname) and (os.path.exists(pathname) or is_path_creatable(pathname))
    except OSError:
        return False

def is_path_exists(pathname):
    '''
	this function is to justify is given path existing or not
    '''
    try:
        return is_path_valid(pathname) and os.path.exists(pathname)
    except OSError:
        return False

def isfile(pathname):
    if is_path_valid(pathname):
        name = os.path.splitext(os.path.basename(pathname))[0]
        ext = os.path.splitext(pathname)[1]
        return len(name) > 0 and len(ext) > 0
    else:
        return False;


def isfolder(pathname):
    if is_path_valid(pathname):
        name = os.path.splitext(os.path.basename(pathname))[0]
        ext = os.path.splitext(pathname)[1]
        return len(name) > 0 and len(ext) == 0
    else:
        return False


def CHECK_EQ_LIST(input_list):
	'''
	check all elements in a list are equal
	'''
	assert islist(input_list), 'input is not a list'
	return input_list[1:] == input_list[:-1]