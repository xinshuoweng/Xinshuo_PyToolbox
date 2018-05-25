# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np
from PIL import Image

import init_paths
from xinshuo_visualization import visualize_image_with_pts

def test_visualize_image_with_pts():
	image_path = '../lena.png'

	print('testing basic')
	img = Image.open(image_path).convert('L')
	pts_array = [[200, 300], [400, 400]]
	visualize_image_with_pts(img, pts_array, vis=True)	

	print('testing basic')
	img = Image.open(image_path).convert('L')
	pts_array = [[200, 300, 1], [400, 400, 1]]
	visualize_image_with_pts(img, pts_array, vis=True)	

	print('testing color index')
	img = Image.open(image_path).convert('L')
	pts_array = [[200, 300, 1], [400, 400, 1]]
	visualize_image_with_pts(img, pts_array, color_index=1, vis=True)	

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_visualize_image_with_pts()