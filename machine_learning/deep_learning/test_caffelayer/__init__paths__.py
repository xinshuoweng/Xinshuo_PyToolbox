#!/usr/bin/env python
# -*- coding: utf-8 -*- 

# Author: Xinshuo
# Email: xinshuow@andrew.cmu.edu
"""Set up paths."""

import sys, os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(os.path.abspath(__file__))

# Add caffe to PYTHONPATH
caffe_path = os.path.join(this_dir, '../../caffe', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
util_path = os.path.join(this_dir, '../')
add_path(util_path)

# Add lib to PYTHONPATH
image_tool_path = os.path.join(this_dir, '../../xinshuo_toolbox', 'images')
add_path(image_tool_path)

# Add lib to PYTHONPATH
file_tool_path = os.path.join(this_dir, '../../xinshuo_toolbox', 'data')
add_path(file_tool_path)

# Add lib to PYTHONPATH
math_tool_path = os.path.join(this_dir, '../../xinshuo_toolbox', 'visualization')
add_path(math_tool_path)

# Add lib to PYTHONPATH
python_tool_path = os.path.join(this_dir, '../../xinshuo_toolbox', 'python')
add_path(python_tool_path)


# Add lib to PYTHONPATH
python_tool_path = os.path.join(this_dir, '../../xinshuo_toolbox', 'file_io')
add_path(python_tool_path)


# Add lib to PYTHONPATH
python_tool_path = os.path.join(this_dir, '../../xinshuo_toolbox', 'math')
add_path(python_tool_path)

