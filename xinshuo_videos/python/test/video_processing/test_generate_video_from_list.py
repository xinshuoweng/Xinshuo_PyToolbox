# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import init_paths
from video_processing import generate_video_from_list

def test_generate_video_from_list():
	print('test basic')
	image_list = ['../image0001.jpg', '../image0002.jpg', '../image0003.jpg', '../image0004.jpg', '../image0005.jpg']
	generate_video_from_list(image_list, '../test.mp4')

	print('test slow framerate')
	image_list = ['../image0001.jpg', '../image0002.jpg', '../image0003.jpg', '../image0004.jpg', '../image0005.jpg']
	generate_video_from_list(image_list, '../test_fr5.mp4', framerate=5)

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_generate_video_from_list()