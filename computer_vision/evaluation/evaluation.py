# Author: Xinshuo Weng
# Email: xinshuo.weng@gmail.com

# this file contains all functions abount evaluation in computer vision
import math
import numpy as np
import os

from bbox_transform import bbox_TLBR2TLWH, pts2bbox
from check import is_path_exists_or_creatable, isnparray, islist, isdict, ispositiveinteger, isscalar, islogical
from math_functions import pts_euclidean
from visualize import visualize_ced, visualize_pts
from file_io import fileparts, mkdir_if_missing

def facial_landmark_evaluation(pred_dict_all, anno_dict, num_pts, error_threshold, normalization_ced=True, normalization_vec=True, debug=True, vis=False, save=True, save_path=None):
	'''
	evaluate the performance of facial landmark detection

	parameter:
		pred_dict_all:	a dictionary for all basline methods. Each key is the method name and the value is corresponding prediction dictionary, 
						which keys are the image path and values are 2 x N prediction results
		anno_dict: 		a dictionary which keys are the image path and values are 2 x N annotation results
		num_pts:		number of points
		vis:			determine if visualizing the pck curve
		save:			determine if saving the visualization results
		save_path:		a directory to save all the results

	visualization:
		1. 2d pck curve (total and point specific) for all points for all methods
		2. point error vector (total and point specific) for all points and for all methods
		3. mean square error
	'''
	num_methods = len(pred_dict_all)
	if debug:
		assert isdict(pred_dict_all) and num_methods > 0 and all(isdict(pred_dict) for pred_dict in pred_dict_all.values()), 'predictions result format is not correct'
		assert isdict(anno_dict), 'annotation result format is not correct'
		assert ispositiveinteger(num_pts), 'number of points is not correct'
		assert isscalar(error_threshold), 'error threshold is not correct'
		assert islogical(normalization_ced) and islogical(normalization_vec), 'normalization flag is not correct'

	num_images = len(pred_dict_all.values()[0])
	if debug:
		assert num_images > 0, 'the predictions are empty'
		assert num_images == len(anno_dict), 'number of predictions is not equal to number of annotations'
		assert all(num_images == len(pred_dict) for pred_dict in pred_dict_all.values()), 'number of images in results from different methods are not equal'

	# calculate normalized mean error for each single image based on point-to-point Euclidean distance normalized by the bounding box size
	# calculate point error vector for each single image based on error vector normalized by the bounding box size
	normed_mean_error_dict = dict()
	normed_mean_error_pts_specific_dict = dict()
	pts_error_vec_dict = dict()
	pts_error_vec_pts_specific_dict = dict()
	for method_name, pred_dict in pred_dict_all.items():
		normed_mean_error_total = np.zeros((num_images, ), dtype='float32')
		normed_mean_error_pts_specifc = np.zeros((num_images, num_pts), dtype='float32')
		pts_error_vec = np.zeros((num_images, 2), dtype='float32')
		pts_error_vec_pts_specific = np.zeros((num_images, 2, num_pts), dtype='float32')
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

			# calculate bbox for normalization
			if normalization_ced or normalization_vec:
				bbox_anno = pts2bbox(pts_anno, debug=debug)					
				bbox_TLWH = bbox_TLBR2TLWH(bbox_anno, debug=debug)				
				bbox_size = math.sqrt(bbox_TLWH[0, 2] * bbox_TLWH[0, 3])
			
			# calculate normalized error for all points
			normed_mean_error = pts_euclidean(pts_prediction, pts_anno, debug=debug)
			if normalization_ced:
				normed_mean_error /= bbox_size
			normed_mean_error_total[count] = normed_mean_error

			# calculate normalized error point specifically
			for pts_index in xrange(num_pts):
				pts_prediction_tmp = pts_prediction[:, pts_index]
				pts_anno_tmp = pts_anno[:, pts_index]
				normed_mean_error_pts_specifc_tmp = pts_euclidean(pts_prediction_tmp, pts_anno_tmp, debug=debug)
				if normalization_ced:
					normed_mean_error_pts_specifc_tmp /= bbox_size
				normed_mean_error_pts_specifc[count, pts_index] = normed_mean_error_pts_specifc_tmp

			# calculate the point error vector
			error_vector = pts_prediction - pts_anno 			# 2 x num_pts
			if normalization_vec:
				error_vector /= bbox_size
			pts_error_vec_pts_specific[count, :, :] = error_vector 
			pts_error_vec[count, :] = np.sum(error_vector, axis=1) / num_pts

			count += 1

		# save results to dictionary
		normed_mean_error_dict[method_name] = normed_mean_error_total
		normed_mean_error_pts_specific_dict[method_name] = normed_mean_error_pts_specifc
		pts_error_vec_dict[method_name] = np.transpose(pts_error_vec)
		pts_error_vec_pts_specific_dict[method_name] = pts_error_vec_pts_specific

	# visualize the error vector map
	print('visualizing error vector distribution map....\n')
	error_vec_save_dir = os.path.join(save_path, 'error_vec')
	mkdir_if_missing(error_vec_save_dir)
	savepath_tmp = os.path.join(error_vec_save_dir, 'error_vector_distribution_all.png')
	visualize_pts(pts_error_vec_dict, title='Point Error Vector Distribution (all points)', debug=debug, vis=vis, save=save, save_path=savepath_tmp)
	for pts_index in xrange(num_pts):
		pts_error_vec_pts_specific_dict_tmp = dict()
		for method_name, error_vec_dict in pts_error_vec_pts_specific_dict.items():
			pts_error_vec_pts_specific_dict_tmp[method_name] = np.transpose(error_vec_dict[:, :, pts_index])
		savepath_tmp = os.path.join(error_vec_save_dir, 'error_vector_distribution_pts_%d.png' % (pts_index+1))
		visualize_pts(pts_error_vec_pts_specific_dict_tmp, title='Point Error Vector Distribution for Point %d' % (pts_index+1), debug=debug, vis=vis, save=save, save_path=savepath_tmp)

	# visualize the ced (cumulative error distribution curve)
	print('visualizing pck curve....\n')
	pck_save_dir = os.path.join(save_path, 'pck')
	mkdir_if_missing(pck_save_dir)
	savepath_tmp = os.path.join(pck_save_dir, 'pck_curve_all.png')
	visualize_ced(normed_mean_error_dict, error_threshold=error_threshold, title='2D PCK curve for all 68 points', debug=debug, vis=vis, save=save, save_path=savepath_tmp)
	for pts_index in xrange(num_pts):
		normed_mean_error_dict_tmp = dict()
		for method_name, error_dict in normed_mean_error_pts_specific_dict.items():
			normed_mean_error_dict_tmp[method_name] = np.reshape(error_dict[:, pts_index], (num_images, ))
		savepath_tmp = os.path.join(pck_save_dir, 'pck_curve_pts_%d.png' % (pts_index+1))
		visualize_ced(normed_mean_error_dict_tmp, error_threshold=error_threshold, title='2D PCK curve for point %d' % (pts_index+1), debug=debug, vis=vis, save=save, save_path=savepath_tmp)
