# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file is for the adding the path for python library
# import os, sys

# def add_path(path):
#     if path not in sys.path:
#         sys.path.insert(0, path)

# this_dir = os.path.dirname(os.path.abspath(__file__))

# # Add python to PYTHONPATH
# py_path = os.path.join(this_dir, '../python')
# add_path(py_path)

# # Add math to PYTHONPATH
# math_path = os.path.join(this_dir, '../math')
# add_path(math_path)


# # Add math to PYTHONPATH
# file_path = os.path.join(this_dir, '../file_io')
# add_path(file_path)


# # Add math to PYTHONPATH
# file_path = os.path.join(this_dir, '../miscellaneous')
# add_path(file_path)

from .preprocess import *
from .synthetic_data import *