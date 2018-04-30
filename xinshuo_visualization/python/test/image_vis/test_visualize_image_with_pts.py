# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np
from PIL import Image

import init_paths
from xinshuo_visualization import visualize_image_with_pts

def test_visualize_image_with_pts():
	image_path = '../lena.png'

	print('testing for grayscale pil image.')
	img = Image.open(image_path).convert('L')
	pts_array = [[200, 300, 1], [400, 400, 1]]
	visualize_image_with_pts(img, pts_array, vis=True)	

	# print('testing for color pil image.')
	# img = Image.open(image_path).convert('RGB')
	# visualize_image_with_pts(img, vis=True)	

	# print('testing for color numpy uint8 image')
	# img = np.array(Image.open(image_path).convert('RGB')).astype('uint8')
	# visualize_image_with_pts(img, vis=True)	

	# print('testing for color numpy float32 image')
	# img = np.array(Image.open(image_path).convert('RGB')).astype('float32') / 255.
	# visualize_image_with_pts(img, vis=True)	

	# print('testing for all 1 in image')
	# img = np.ones((600, 600, 3), dtype='float32')
	# visualize_image_with_pts(img, vis=True)		

	# print('testing for all 0 in image')
	# img = np.zeros((600, 600, 3), dtype='float32')
	# visualize_image_with_pts(img, vis=True)		

	# print('testing for unusual shape of grayscale image')
	# img = np.random.rand(600, 600, 1).astype('float32')
	# visualize_image_with_pts(img, vis=True)

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_visualize_image_with_pts()