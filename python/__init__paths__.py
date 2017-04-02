# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file is for the adding the path for python library
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

