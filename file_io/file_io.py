# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file contains a set of function for manipulating file io in python
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../python'))
import type_check

# TODO: CHECK
def is_path_creatable(pathname):
    '''
    `True` if the current user has sufficient permissions to create the passed
    pathname; `False` otherwise.
    '''
    # Parent directory of the passed path. If empty, we substitute the current
    # working directory (CWD) instead.
    assert type_check.isstring(pathname), 'The input path is not a string'
    dirname = os.path.dirname(pathname) or os.getcwd()
    return os.access(dirname, os.W_OK)

# TODO: CHECK
def is_path_exists_or_creatable(pathname):
    '''
    `True` if the passed pathname is a valid pathname for the current OS _and_
    either currently exists or is hypothetically creatable; `False` otherwise.

    This function is guaranteed to _never_ raise exceptions.
    '''
    assert type_check.isstring(pathname), 'The input path is not a string'   
    try:
        # To prevent "os" module calls from raising undesirable exceptions on
        # invalid pathnames, is_pathname_valid() is explicitly called first.
        return is_pathname_valid(pathname) and (
            os.path.exists(pathname) or is_path_creatable(pathname))
    # Report failure on non-fatal filesystem complaints (e.g., connection
    # timeouts, permissions issues) implying this path to be inaccessible. All
    # other exceptions are unrelated fatal issues and should not be caught here.
    except OSError:
        return False

def fileparts(pathname):
	'''
	this function return a tuple, which contains (directory, filename, extension)
	if the file has multiple extension, only last one will be displayed
	'''
	assert type_check.isstring(pathname), 'The input path is not a string'   
	directory = os.path.dirname(pathname)
	filename = os.path.splitext(os.path.basename(pathname))[0]
	ext = os.path.splitext(pathname)[1]
	return (directory, filename, ext)

