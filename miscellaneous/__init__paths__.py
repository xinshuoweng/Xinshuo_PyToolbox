# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file is for the adding the path for python library
import os, sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(this_dir, '../python')
add_path(file_path)