# Author: Xinshuo Weng
# Email: xinshuo.weng@gmail.com

# this file contains all functions abount evaluation in computer vision
import math
import numpy as np

from bbox_transform import bbox_TLBR2TLWH, pts2bbox
from check import is_path_exists_or_creatable, isnparray
from math_functions import pts_euclidean
from visualize import visualize_ced

def facial_landmark_evaluation(pred_dict, anno_dict, num_pts, error_threshold, debug=True, vis=False, save=True, save_path=None):
	'''
	evaluate the performance of facial landmark detection

	parameter:
		pred_dict: 		a dictionary which keys are the image path and values are 2 x N prediction results
		anno_dict: 		a dictionary which keys are the image path and values are 2 x N annotation results
		vis:			determine if visualizing the pck curve
		save:			determine if saving the visualization results
		save_path:		a path to save the results
	'''
	num_images = len(pred_dict)
	if debug:
		assert num_images > 0, 'the predictions are empty'
		assert num_images == len(anno_dict), 'number of predictions is not equal to number of annotations'

	# calculate normalized mean error for each single image based point-to-point Euclidean distance normalized by the bounding box size
	normed_mean_error_total = np.zeros((num_images, ), dtype='float32')
	count = 0
	for image_path, pts_prediction in pred_dict.items():
		pts_anno = anno_dict[image_path]		# 2 x N annotation
		if debug:
			assert isnparray(pts_anno) and pts_anno.shape[0] == 2 and pts_anno.shape[1] == num_pts, 'shape of annotations is not correct (%d x %d) vs (%d x %d)' % (2, num_pts, pts_anno[0], pts_anno[1])
			assert pts_anno.shape == pts_prediction.shape, 'shape of predictions and annotation is not the same'
		bbox_anno = pts2bbox(pts_anno, debug=debug)		
		bbox_TLWH = bbox_TLBR2TLWH(bbox_anno, debug=debug)
		bbox_size = math.sqrt(bbox_TLWH[0, 2] * bbox_TLWH[0, 3])
		normed_mean_error = pts_euclidean(pts_prediction, pts_anno, debug=debug) / bbox_size
		normed_mean_error_total[count] = normed_mean_error
		count += 1

	# visualize the ced (cumulative error distribution curve)
	visualize_ced(normed_mean_error_total, error_threshold=error_threshold, debug=debug, vis=vis, save=save, save_path=save_path)