# Author: Xinshuo Weng
# Email: xinshuo.weng@gmail.com

# this file contains all functions abount evaluation in computer vision
import math
import numpy as np

from bbox_transform import bbox_TLBR2TLWH, pts2bbox
from check import is_path_exists_or_creatable, isnparray, islist, isdict
from math_functions import pts_euclidean
from visualize import visualize_ced
from file_io import fileparts

def facial_landmark_evaluation(pred_dict_all, anno_dict, num_pts, error_threshold, debug=True, vis=False, save=True, save_path=None):
	'''
	evaluate the performance of facial landmark detection

	parameter:
		pred_dict_all:	a dictionary for all basline methods. Each key is the method name and the value is corresponding prediction dictionary, 
						which keys are the image path and values are 2 x N prediction results
		anno_dict: 		a dictionary which keys are the image path and values are 2 x N annotation results
		vis:			determine if visualizing the pck curve
		save:			determine if saving the visualization results
		save_path:		a path to save the results
	'''
	num_methods = len(pred_dict_all)
	if debug:
		assert isdict(pred_dict_all) and num_methods > 0 and all(isdict(pred_dict) for pred_dict in pred_dict_all.values()), 'predictions result format is not correct'
		assert isdict(anno_dict), 'annotation result format is not correct'

	num_images = len(pred_dict_all.values()[0])
	if debug:
		assert num_images > 0, 'the predictions are empty'
		assert num_images == len(anno_dict), 'number of predictions is not equal to number of annotations'
		assert all(num_images == len(pred_dict) for pred_dict in pred_dict_all.values()), 'number of images in results from different methods are not equal'

	# calculate normalized mean error for each single image based point-to-point Euclidean distance normalized by the bounding box size
	normed_mean_error_dict = dict()
	for method_name, pred_dict in pred_dict_all.items():
		normed_mean_error_total = np.zeros((num_images, ), dtype='float32')
		count = 0
		for image_path, pts_prediction in pred_dict.items():
			_, filename, _ = fileparts(image_path)
			pts_anno = anno_dict[filename]				# 2 x N annotation
			
			# to avoid list object type, do conversion here
			if islist(pts_anno):
				pts_anno = np.asarray(pts_anno)
			if islist(pts_prediction):
				pts_prediction = np.asarray(pts_prediction)

			# to avoid the point location includes the score or occlusion channel, only take the first two channels here
			if pts_anno.shape[0] == 3 or pts_anno.shape[0] == 4:
				pts_anno = pts_anno[0:2, :]
			if pts_prediction.shape[0] == 3 or pts_prediction.shape[0] == 4:
				pts_prediction = pts_prediction[0:2, :]

			if debug:
				assert isnparray(pts_anno) and pts_anno.shape[0] == 2 and pts_anno.shape[1] == num_pts, 'shape of annotations is not correct (%d x %d) vs (%d x %d)' % (2, num_pts, pts_anno[0], pts_anno[1])
				assert pts_anno.shape == pts_prediction.shape, 'shape of predictions and annotation is not the same'

			# calculate normalized mean error
			bbox_anno = pts2bbox(pts_anno, debug=debug)						# obtain the bbox
			bbox_TLWH = bbox_TLBR2TLWH(bbox_anno, debug=debug)				
			bbox_size = math.sqrt(bbox_TLWH[0, 2] * bbox_TLWH[0, 3])
			normed_mean_error = pts_euclidean(pts_prediction, pts_anno, debug=debug) / bbox_size
			normed_mean_error_total[count] = normed_mean_error
			count += 1
		normed_mean_error_dict[method_name] = normed_mean_error_total

	# visualize the ced (cumulative error distribution curve)
	visualize_ced(normed_mean_error_dict, error_threshold=error_threshold, debug=debug, vis=vis, save=save, save_path=save_path)