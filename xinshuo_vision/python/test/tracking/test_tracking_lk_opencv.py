# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np, cv2

import init_paths
from xinshuo_io import load_image
from xinshuo_images import rgb2gray
from xinshuo_visualization import visualize_image_with_pts, visualize_image, visualize_pts_array
from xinshuo_miscellaneous import find_unique_common_from_lists
from tracking import tracking_lk_opencv

def test_tracking_lk_opencv():
	print('test color image with backward')
	image_prev = load_image('../frame_prev.jpg')
	image_next = load_image('../frame_next.jpg')
	num_pts = 300000
	threshold = 0.05
	input_pts = np.random.rand(2, num_pts)
	input_pts *= 360
	pts_forward, pts_bacward, backward_err, found_index = tracking_lk_opencv(image_prev, image_next, input_pts, win_size=15, backward=True)

	assert len(backward_err) == num_pts
	backward_err = np.array(backward_err).reshape((num_pts, 1))
	err_small_index = np.where(backward_err[:, 0] < threshold)[0].tolist()
	print('length of found index is %d' % len(found_index))
	print('length of backward err is %d' % len(err_small_index))
	good_index = find_unique_common_from_lists(err_small_index, found_index)
	print('length of good index is %d' % (len(good_index)))
	pts_good = input_pts[:, good_index]
	pts_forward = pts_forward[:, good_index]
	pts_bacward = pts_bacward[:, good_index]

	print('test color image with forward')
	pts_forward, _, _, found_index = tracking_lk_opencv(image_prev, image_next, pts_good)
	visualize_image_with_pts(image_prev, pts_good, pts_size=2, color_index=2, vis=True, save_path='./0prev.jpg')
	visualize_image_with_pts(image_next, pts_forward, pts_size=2, color_index=2, vis=True, save_path='./1next.jpg')
	visualize_image_with_pts(image_prev, pts_bacward, pts_size=2, color_index=2, vis=True, save_path='./2back.jpg')

	print('test grayscale image with forward')
	pts_forward, _, _, found_index = tracking_lk_opencv(rgb2gray(image_prev), image_next, pts_good)
	visualize_image_with_pts(image_prev, pts_good, pts_size=2, color_index=2, vis=True)
	visualize_image_with_pts(image_next, pts_forward, pts_size=2, color_index=2, vis=True)

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_tracking_lk_opencv()