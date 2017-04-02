# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import sys, os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.join(os.path.dirname(__file__), '../../python'))