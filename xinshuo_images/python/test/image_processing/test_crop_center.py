# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from image_processing import crop_center
from xinshuo_visualization import visualize_image
from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_crop_center():
	image_path = '../lena.jpg'
	img = Image.open(image_path).convert('RGB')

	#################################### test with 4 elements in center_rect #########################################
	print('test 2d matrix')
	np_data = (np.random.rand(5, 5) * 255.).astype('uint8')
	center_rect = [1, 2, 4, 6]
	img_padded, crop_bbox, crop_bbox_clipped = crop_center(np_data, center_rect=center_rect, pad_value=10)
	print(np_data)
	print(img_padded)
	assert img_padded.shape == (6, 4), 'the padded image does not have a good shape'
	assert CHECK_EQ_NUMPY(crop_bbox, np.array([[-1, -1, 4, 6]]))
	assert CHECK_EQ_NUMPY(crop_bbox_clipped, np.array([[0, 0, 3, 5]]))

	print('test 2d matrix')
	np_data = (np.random.rand(5, 5) * 255.).astype('uint8')
	center_rect = [3, 2, 4, 6]
	img_padded, crop_bbox, crop_bbox_clipped = crop_center(np_data, center_rect=center_rect, pad_value=10)
	print(np_data)
	print(img_padded)
	assert img_padded.shape == (6, 4), 'the padded image does not have a good shape'
	assert CHECK_EQ_NUMPY(crop_bbox, np.array([[1, -1, 4, 6]]))
	assert CHECK_EQ_NUMPY(crop_bbox_clipped, np.array([[1, 0, 4, 5]]))

	print('test with grayscale image, clipped on the left')
	center_rect = [0, 50, 100, 100]
	img_cropped, crop_bbox, crop_bbox_clipped = crop_center(img, center_rect=center_rect, pad_value=100)
	assert CHECK_EQ_NUMPY(crop_bbox, np.array([[-50, 0, 100, 100]]))
	assert CHECK_EQ_NUMPY(crop_bbox_clipped, np.array([[0, 0, 50, 100]]))
	visualize_image(img, vis=True)
	visualize_image(img_cropped, vis=True)

	#################################### test with 2 elements in center_rect #########################################

	print('\n\nDONE! SUCCESSFUL!!\n')

if __name__ == '__main__':
	test_crop_center()