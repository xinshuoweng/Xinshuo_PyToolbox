# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
from PIL import Image
import numpy as np

import init_paths
from image_processing import crop_center
from xinshuo_visualization import visualize_image

def test_crop_center():
	image_path = 'lena.jpg'
	img = Image.open(image_path).convert('L')
	img = np.array(img)
	print(img.shape)

	center_rect = [5, 50, 20, 20]
	_, crop1, crop2 = crop_center(img, center_rect=center_rect)
	print(crop1)
	print(crop2)

	# print('input grayscale image has dimension of: '),
	# print(img.shape)
	# assert isgrayimage(img), 'the input image is not a gray image'
	# visualize_image(img, vis=True)

	# img_rgb = gray2rgb(img, with_color=True)
	# print('converted rgb image has dimension of: '),
	# print(img_rgb.shape)
	# assert iscolorimage(img_rgb), 'the converted image is not a color image'
	# visualize_image(img_rgb, vis=True)

if __name__ == '__main__':
	test_crop_center()
