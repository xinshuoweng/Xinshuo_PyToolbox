# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
from scipy import ndimage, signal
import numpy as np

import init_paths
from image_operator import linear_filter
from xinshuo_visualization import visualize_image

def test_sobel_filter():
	image_path = '../lena.png'

	print('testing for grayscale image with sobel along x axis')
	img = Image.open(image_path).convert('L')
	filter = linear_filter()
	sobel_kernel = filter.sobel()
	filtered_img = filter.convolve(img)
	visualize_image(img, vis=True)
	visualize_image(filtered_img, vis=True)

	print('testing for grayscale image with sobel along y axis')
	img = Image.open(image_path).convert('L')
	filter = linear_filter()
	sobel_kernel = filter.sobel(axis='y')
	filtered_img = filter.convolve(img)
	visualize_image(img, vis=True)
	visualize_image(filtered_img, vis=True)

	print('testing for color image with sobel along X axis')
	img = np.array(Image.open(image_path).convert('RGB')).astype('float32') / 255.
	filter = linear_filter()
	sobel_kernel = filter.sobel(axis='y')
	sobel_kernel = filter.expand_3d()
	filtered_img = ndimage.filters.convolve(img, sobel_kernel)
	visualize_image(img, vis=True)
	visualize_image(filtered_img, vis=True)

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_sobel_filter()