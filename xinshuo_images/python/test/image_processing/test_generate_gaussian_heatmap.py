# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np
# from numpy.testing import assert_almost_equal

import init_paths
from image_processing import generate_gaussian_heatmap
from xinshuo_visualization import visualize_image
from xinshuo_math import nparray_resize

def test_generate_gaussian_heatmap():
	print('test single point')
	input_pts = [300, 400, 1]
	image_size = [800, 600]
	std = 10
	heatmap, mask = generate_gaussian_heatmap(input_pts, image_size=image_size, std=std)
	assert heatmap.shape == (800, 600, 2)
	assert mask.shape == (1, 1, 2)
	visualize_image(heatmap[:, :, -1], vis=True)

	print('test two points')
	input_pts = [[300, 400, 1], [400, 400, 1]]
	image_size = [800, 600]
	std = 10
	heatmap, mask = generate_gaussian_heatmap(input_pts, image_size=image_size, std=std)
	assert heatmap.shape == (800, 600, 3)
	assert mask.shape == (1, 1, 3)
	visualize_image(heatmap[:, :, -1], vis=True)

	print('test two points with invalid one')
	input_pts = [[300, 400, 1], [400, 400, -1]]
	image_size = [800, 600]
	std = 10
	heatmap, mask = generate_gaussian_heatmap(input_pts, image_size=image_size, std=std)
	assert heatmap.shape == (800, 600, 3)
	assert mask.shape == (1, 1, 3)
	visualize_image(heatmap[:, :, -1], vis=True)


	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_generate_gaussian_heatmap()