# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
from scipy import signal
from scipy import ndimage
import numpy as np

import init_paths
from image_operator import linear_filter
from xinshuo_visualization import visualize_image

def test_image_hist_equalization():
	image_path = '../lena.png'

	# print('testing for grayscale image with sobel along x axis')
	# filter = linear_filter()
	# sobel_kernel = filter.sobel_filter()
	# img = Image.open(image_path).convert('L')
	# filtered_img = signal.convolve2d(img, sobel_kernel)
	# visualize_image(img, vis=True)
	# visualize_image(filtered_img, vis=True)

	# print('testing for grayscale image with sobel along y axis')
	# filter = linear_filter()
	# sobel_kernel = filter.sobel_filter(axis='y')
	# img = Image.open(image_path).convert('L')
	# filtered_img = signal.convolve2d(img, sobel_kernel)
	# visualize_image(img, vis=True)
	# visualize_image(filtered_img, vis=True)

	print('testing for color image with sobel along X axis')
	filter = linear_filter()
	sobel_kernel = filter.sobel_filter()
	img = Image.open(image_path).convert('RGB')
	# filtered_img = signal.convolve2d(img, sobel_kernel)
	filtered_img = ndimage.filters.convolve(img, sobel_kernel)
	visualize_image(img, vis=True)
	visualize_image(filtered_img, vis=True)

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_image_hist_equalization()