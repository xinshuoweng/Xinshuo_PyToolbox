# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np
from numpy.testing import assert_almost_equal

import init_paths
from image_processing import image_draw_mask
from xinshuo_miscellaneous import CHECK_EQ_NUMPY

def test_image_draw_mask():
	mask = '../rainbow.jpg'
	mask = Image.open(mask).convert('RGB')

	print('test with pil image')
	image_path = '../lena.png'
	img = Image.open(image_path).convert('RGB')
	img_bak = img.copy()
	masked_img = image_draw_mask(img, mask)
	masked_img += 1
	assert CHECK_EQ_NUMPY(img_bak, img), 'the original image should be equal to the backup version'

	print('test with pil image')
	image_path = '../lena.png'
	img = Image.open(image_path).convert('RGB')
	img_bak = img.copy()
	masked_img = image_draw_mask(img, mask)
	masked_img += 1
	assert CHECK_EQ_NUMPY(img_bak, img), 'the original image should be equal to the backup version'

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_image_draw_mask()