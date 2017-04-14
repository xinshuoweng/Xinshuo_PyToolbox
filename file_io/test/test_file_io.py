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

def test_mkdir_if_missing():
    path = './'
    mkdir_if_missing(path)
    path = 'test_folder'
    mkdir_if_missing(path)

if __name__ == '__main__':
    pytest.main([__file__])