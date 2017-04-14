# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file contains a set of function for manipulating file io in python
import os, sys
import pytest

import __init__paths__
from file_io import *


def test_load_list_from_file():
    path = 'test1.txt'
    datalist, num_elem = load_list_from_file(path)
    assert datalist[0] == '/home/xinshuow/test'
    assert datalist[1] == '/home/xinshuow/toy'
    assert num_elem == 2

def test_load_list_from_folder():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test')
    datalist, num_elem = load_list_from_folder(folder_path=path, ext_filter='txt')   
    assert datalist[0] == os.path.abspath('test.txt')
    assert datalist[1] == os.path.abspath('test1.txt')
    assert num_elem == 2

    datalist, num_elem = load_list_from_folder(folder_path=path)
    assert num_elem == 8


def test_mkdir_if_missing():
    path = './'
    mkdir_if_missing(path)
    path = 'test_folder'
    mkdir_if_missing(path)
    path = 'test_folder1/te.txt'
    mkdir_if_missing(path)

if __name__ == '__main__':
    pytest.main([__file__])