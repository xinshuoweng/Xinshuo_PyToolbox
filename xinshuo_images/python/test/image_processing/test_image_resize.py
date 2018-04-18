# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from image_processing import image_resize

def test_image_resize():
	image_path = '../lena.png'
	img = Image.open(image_path).convert('RGB')
	im_height, im_width = img.size

	print('test input image as pil image')
	resized_img = image_resize(img, resize_factor=0.3)
	assert resized_img.shape == (int(round(im_height * 0.3)), int(round(im_width * 0.3)), 3)

	print('test input image as numpy uint8 image with resize_factor')
	img = (np.random.rand(400, 300, 1) * 255.).astype('uint8')
	resized_img = image_resize(img, resize_factor=0.5)
	assert resized_img.shape == (200, 150)

	print('test input image as numpy float32 image with target_size')
	img = np.random.rand(400, 300, 3).astype('float32')
	resized_img = image_resize(img, target_size=(1000, 100))
	assert resized_img.shape == (1000, 100, 3)

	print('test input image as numpy float32 image with target_size')
	img = np.random.rand(400, 300, 3).astype('float32')
	resized_img = image_resize(img, target_size=(1000, 100))
	assert resized_img.shape == (1000, 100, 3)

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_image_resize()