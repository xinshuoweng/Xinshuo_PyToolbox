# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import sys, os
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

math_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../math')
add_path(math_path)