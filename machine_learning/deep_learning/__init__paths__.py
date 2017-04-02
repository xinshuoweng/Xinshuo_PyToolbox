# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import sys, os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

python_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../python')
add_path(python_path)

file_io_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../file_io')
add_path(file_io_path)