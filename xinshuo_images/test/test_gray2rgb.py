# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import pytest
from PIL import Image
import numpy as np

import init_paths
from image_processing import gray2rgb

def test_gray2rgb():
	image_path = 'lena.jpg'
	img = Image.open(image_path).convert('L')
	print(img)
	img = np.array(img)
	print(img)


if __name__ == '__main__':
	pytest.main([__file__])