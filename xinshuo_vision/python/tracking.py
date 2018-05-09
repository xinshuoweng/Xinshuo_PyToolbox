# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes basic tracking algorithms in computer vision
import cv2, numpy as np

from xinshuo_images.python.private import safe_image
from xinshuo_math.python.private import safe_2dptsarray
from xinshuo_images import rgb2gray
from xinshuo_miscellaneous import iscolorimage_dimension, isscalar

def tracking_lk_opencv(input_image1, input_image2, input_pts, backward=False, win_size=15, pyramid=5, warning=True, debug=True):
	'''
	tracking a set of points in two images using Lucas-Kanade tracking implemented in opencv

	parameters:
		input_image1, input_image2:				a pil or numpy image, color or gray
		input_pts: 			a list of 2 elements, a listoflist of 2 elements: 
							e.g., [[1,2], [5,6]], a numpy array with shape or (2, N) or (2, )
		backward:			run backward tracking if true
		win_sie:			window sized used for lucas kanade tracking
		pyramid:			number of levels of pyramid used for lucas kanade tracking

	outputs:
		pts_forward:		tracked points in forward pass, 2 x N float32 numpy
		pts_bacward:		tracked points in backward pass, 2 x N float32 numpy
		backward_err:		error in forward-backward pass check
	'''
	np_image1, _ = safe_image(input_image1, warning=warning, debug=debug)
	np_image2, _ = safe_image(input_image2, warning=warning, debug=debug)
	np_pts = safe_2dptsarray(input_pts, homogeneous=False, warning=warning, debug=debug).astype('float32')		# 2 x N
	num_pts = np_pts.shape[1]
	if debug: assert isscalar(win_size) and isscalar(pyramid), 'the hyperparameters of lucas-kanade tracking is not correct'

	# formatting the input
	if iscolorimage_dimension(np_image1): np_image1 = rgb2gray(np_image1)
	if iscolorimage_dimension(np_image2): np_image2 = rgb2gray(np_image2)

	lk_params = dict(winSize=(win_size, win_size), maxLevel=pyramid, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
	pts_root = np.expand_dims(np_pts.transpose(), axis=1) 			# N x 1 x 2
	
	pts_forward, status_for, err_for = cv2.calcOpticalFlowPyrLK(np_image1, np_image2, pts_root, None, **lk_params) 
	if backward: 
		pts_bacward, status_bac, err_bac = cv2.calcOpticalFlowPyrLK(np_image2, np_image1, pts_forward, None, **lk_params)
		backward_err = abs(pts_forward - pts_bacward)
		pts_bacward = pts_bacward.reshape((num_pts, 2)).transpose()
	else: 
		pts_bacward, backward_err = None, 0.0

	return pts_forward.reshape((num_pts, 2)).transpose(), pts_bacward, backward_err