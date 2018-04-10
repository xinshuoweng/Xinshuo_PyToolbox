# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from image_processing import rgb2hsv
from xinshuo_python import isgrayimage
from xinshuo_visualization import visualize_image

def test_rgb2hsv():
	image_path = 'lena.jpg'
	img = Image.open(image_path).convert('RGB')
	visualize_image(img, vis=True)
	hsv_img = rgb2hsv(img)
	visualize_image(hsv_img, vis=True)

if __name__ == '__main__':
	test_rgb2hsv()
