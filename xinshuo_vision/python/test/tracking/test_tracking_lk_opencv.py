# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np, cv2

import init_paths
from xinshuo_io import load_image
from xinshuo_visualization import visualize_image_with_pts
from tracking import tracking_lk_opencv

def test_tracking_lk_opencv():
	print('test color image with backward')
	image_prev = load_image('../frame_prev.jpg')
	image_next = load_image('../frame_next.jpg')
	input_pts = np.random.rand(2, 40)
	input_pts *= 260
	pts_forward, pts_bacward, backward_err = tracking_lk_opencv(image_prev, image_next, input_pts, backward=True)

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_tracking_lk_opencv()
