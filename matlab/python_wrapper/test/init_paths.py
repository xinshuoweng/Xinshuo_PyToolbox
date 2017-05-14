# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import sys, os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

def main():
	this_dir = os.path.dirname(os.path.abspath(__file__))
	lib_path = os.path.join(this_dir, '../')
	add_path(lib_path)




