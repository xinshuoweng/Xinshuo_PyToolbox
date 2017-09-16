# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file is for the adding the path for python library
import os, sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(os.path.abspath(__file__))

python_path = os.path.join(this_dir, '../python')
add_path(python_path)

file_path = os.path.join(this_dir, '../file_io')
add_path(file_path)

math_path = os.path.join(this_dir, '../math')
add_path(math_path)

bbox_path = os.path.join(this_dir, '../computer_vision', 'bbox_transform')
add_path(bbox_path)

image_path = os.path.join(this_dir, '../images')
add_path(image_path)