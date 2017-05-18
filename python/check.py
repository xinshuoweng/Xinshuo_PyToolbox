# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import os, sys
import numpy as np

def isstring(string_test):
	return isinstance(string_test, basestring)

def isinteger(integer_test):
	return isinstance(integer_test, int)

def isfloat(float_test):
    return isinstance(float_test, float)

def islist(list_test):
	return isinstance(list_test, list)

def islogical(logical_test):
    return isinstance(logical_test, bool)

def islistoflist(list_test):
    if not islist(list_test):
        return False
    if all(islist(tmp) for tmp in list_test):
        return True
    else:
        return False

def isnparray(nparray_test):
	return isinstance(nparray_test, np.ndarray)

def istuple(tuple_test):
	return isinstance(tuple_test, tuple)

def is2dline(line_test):
	return (isnparray(line_test) or islist(line_test) or istuple(line_test)) and np.array(line_test).size == 3

def is2dpts(pts_test):
	return (isnparray(pts_test) or islist(pts_test) or istuple(pts_test)) and np.array(pts_test).size == 2

def isfunction(func_test):
	return callable(func_test)

def isdict(dict_test):
    return isinstance(dict_test, dict)

def istuple(tuple_test):
    return isinstance(tuple_test, tuple)

def isext(ext_test):
    return isstring(ext_test) and ext_test[0] == '.'

def iscolorimage(image_test, debug=True):
    if debug:
        print('is numpy array when testing color image?')
    if not isnparray(image_test):
        return False

    if debug:
        print('image is numpy array. is dimension correct when testing color image?')
    shape_check = (image_test.ndim == 3 and image_test.shape[2] == 3)

    return True if shape_check else False

def isgrayimage(image_test, debug=True):
    if debug:
        print('is numpy array when testing grayscale image?')
    if not isnparray(image_test):
        return False

    if debug:
        print('image is numpy array. is dimension correct when testing grayscale image?')
    shape_check = (image_test.ndim == 2 or (image_test.ndim == 3 and image_test.shape[2] == 1))

    return True if shape_check else False

def isuintimage(image_test, debug=True):
    if debug:
        print('is shape correct when testing uint8 image?')
    if not (isgrayimage(image_test, debug=debug) or iscolorimage(image_test, debug=debug)):
        return False

    if debug:
        print('shapre is correct. is type correct when testing uint8 image?')
    if not image_test.dtype == 'uint8':
        return False

    if debug:
        print('type is also correct. is value inside array correct when testing uint8 image?')
    item_check_le = (image_test <= 255)
    item_check_se = (image_test >= 0)
    if item_check_le.all() and item_check_se.all():
        return True
    else:
        return False

def isfloatimage(image_test, debug=True):
    if debug:
        print('is shape correct when testing float32 image?')
    if not (isgrayimage(image_test, debug=debug) or iscolorimage(image_test, debug=debug)):
        return False

    if debug:
        print('shapre is correct. is type correct when testing float32 image?')
    if not image_test.dtype == 'float32':
        return False

    if debug:
        print('type is also correct. is value inside array correct when testing float32 image?')
    item_check_le = (image_test <= 1.0)
    item_check_se = (image_test >= 0.0)
    if item_check_le.all() and item_check_se.all():
        return True
    else:
        return False

def isimage(image_test, debug=True):
    return isfloatimage(image_test, debug=debug) or isuintimage(image_test, debug=debug)

def isscaledimage(image_test):
    if not isimage(image_test):
        return False
    max_value = np.max(image_test)
    min_value = np.min(image_test)
    assert min_value >= 0, 'image value is not correct'
    assert max_value >= 0, 'image value is not correct' 
    if max_value > 1 and max_value < 255:
        print('input image is raw image in [0, 255]')
        return False
    elif max_value <= 1:
        return True
    else:
        assert False, 'Unknown error'

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

    For folder, it needs the previous level of folder existing
    for file, it needs the folder existing
    '''
    if is_path_valid(pathname) is False:
    	return False
    
    pathname = safepath(pathname)
    pathname = os.path.dirname(os.path.abspath(pathname))
    
    # recursively to find the root existing
    while not is_path_exists(pathname):     
        pathname_new = os.path.dirname(os.path.abspath(pathname))
        if pathname_new == pathname:
            return False
        pathname = pathname_new
    return os.access(pathname, os.W_OK)

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
        pathname = safepath(pathname)
        name = os.path.splitext(os.path.basename(pathname))[0]
        ext = os.path.splitext(pathname)[1]
        return len(name) > 0 and len(ext) > 0
    else:
        return False;

def isfolder(pathname):
    if is_path_valid(pathname):
        pathname = safepath(pathname)
        if pathname == './':
            return True
        name = os.path.splitext(os.path.basename(pathname))[0]
        ext = os.path.splitext(pathname)[1]
        return len(name) > 0 and len(ext) == 0
    else:
        return False

def safepath(pathname, debug=True):
    '''
    convert path to a normal representation
    '''
    if debug:
        assert is_path_valid(pathname), 'path is not valid'
    return os.path.normpath(pathname)

def CHECK_EQ_LIST(input_list):
	'''
	check all elements in a list are equal
	'''
	assert islist(input_list), 'input is not a list'
	return input_list[1:] == input_list[:-1]

def CHECK_EQ_DICT(input_dict1, input_dict2):
    '''
    check all elements in a list are equal
    '''
    assert isdict(input_dict1) and isdict(input_dict2), 'input is not a dictionary'
    assert len(input_dict1) == len(input_dict2), 'length of input dictionary is not equal'

    for key, value in input_dict1.items():
        if input_dict2.has_key(key) and input_dict2[key] == value:
            continue
        else:
            return False
    return True
