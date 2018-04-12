# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from image_processing import hsv2rgb, rgb2hsv
from xinshuo_visualization import visualize_image

def test_hsv2rgb():
	image_path = 'lena.jpg'

	# test for rgb pil image
	img = Image.open(image_path).convert('RGB')
	visualize_image(img, vis=True)
	hsv_img = rgb2hsv(img)
	visualize_image(hsv_img, vis=True)

	print(hsv_img)
	print(hsv_img.dtype)

	rgb_img = hsv2rgb(hsv_img)
	visualize_image(rgb_img, vis=True)

	print('\n\nDONE! SUCCESSFUL!!\n')
if __name__ == '__main__':
	test_hsv2rgb()
