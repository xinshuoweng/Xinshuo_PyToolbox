# Author: Xinshuo Weng
# Email: xinshuo.weng@gmail.com

# this file contains all functions about evaluation in computer vision
import math
import numpy as np
import os, sys, json, operator
import time
from terminaltables import AsciiTable
from tabulate import tabulate

from bbox_transform import bbox_TLBR2TLWH, pts2bbox
from check import *
from math_functions import pts_euclidean
from visualize import visualize_ced, visualize_pts
from file_io import fileparts, mkdir_if_missing
from conversions import print_np_shape, list_reorder

# for better visualization in error distribution, we center the distribution map and set fixed visualization range for fair comparison
display_range = True
limit = 50
xlim = [-1 * limit, limit]
ylim = [-1 * limit, limit]
# xlim = [-200, 200]
# ylim = [-200, 200]
mse = True
truncated_list = [10, 20, 30, 40, 50]

def facial_landmark_evaluation(pred_dict_all, anno_dict, num_pts, error_threshold, normalization_ced=True, normalization_vec=False, covariance=True, display_list=None, debug=True, vis=False, save=True, save_path=None):
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

	return:
		metrics_all:	a list of list to have detailed metrics over all methods
		ptswise_mse:	a list of list to have average MSE over all key-points for all methods
	'''
	num_methods = len(pred_dict_all)
	if debug:
		assert isdict(pred_dict_all) and num_methods > 0 and all(isdict(pred_dict) for pred_dict in pred_dict_all.values()), 'predictions result format is not correct'
		assert isdict(anno_dict), 'annotation result format is not correct'
		assert ispositiveinteger(num_pts), 'number of points is not correct'
		assert isscalar(error_threshold), 'error threshold is not correct'
		assert islogical(normalization_ced) and islogical(normalization_vec), 'normalization flag is not correct'
		if display_list is not None:
			assert len(display_list) == num_methods, 'display list is not correct %d vs %d' % (len(display_list), num_methods)

	num_images = len(pred_dict_all.values()[0])
	if debug:
		assert num_images > 0, 'the predictions are empty'
		assert num_images == len(anno_dict), 'number of images is not equal to number of annotations: %d vs %d' % (num_images, len(anno_dict))
		assert all(num_images == len(pred_dict) for pred_dict in pred_dict_all.values()), 'number of images in results from different methods are not equal'

	# calculate normalized mean error for each single image based on point-to-point Euclidean distance normalized by the bounding box size
	# calculate point error vector for each single image based on error vector normalized by the bounding box size
	normed_mean_error_dict = dict()
	normed_mean_error_pts_specific_dict = dict()
	normed_mean_error_pts_specific_valid_dict = dict()
	pts_error_vec_dict = dict()
	pts_error_vec_pts_specific_dict = dict()
	mse_error_dict_dict = dict()
	for method_name, pred_dict in pred_dict_all.items():
		normed_mean_error_total = np.zeros((num_images, ), dtype='float32')
		normed_mean_error_pts_specific = np.zeros((num_images, num_pts), dtype='float32')
		normed_mean_error_pts_specific_valid = np.zeros((num_images, num_pts), dtype='bool')
		pts_error_vec = np.zeros((num_images, 2), dtype='float32')					
		pts_error_vec_pts_specific = np.zeros((num_images, 2, num_pts), dtype='float32')
		mse_error_dict = dict()
		count = 0
		count_skip_num_images = 0						# it's possible that no annotation exists on some images, than no error should be counted for those images, we count the number of those images
		for image_path, pts_prediction in pred_dict.items():
			_, filename, _ = fileparts(image_path)
			pts_anno = anno_dict[filename]				# 2 x N annotation
			pts_keep_index = range(num_pts)

			# to avoid list object type, do conversion here
			if islist(pts_anno):
				pts_anno = np.asarray(pts_anno)
			if islist(pts_prediction):
				pts_prediction = np.asarray(pts_prediction)
			if debug:
				assert (is2dptsarray(pts_anno) or is2dptsarray_occlusion(pts_anno)) and pts_anno.shape[1] == num_pts, 'shape of annotations is not correct (%d x %d) vs (%d x %d)' % (2, num_pts, pts_anno.shape[0], pts_anno.shape[1])

			# if the annotation has 3 channels (include extra occlusion channel, we keep only the points with annotations)
			# occlusion: -1 -> not annotated, 0 -> invisible, 1 -> visible, we keep both visible and invisible points
			if pts_anno.shape[0] == 3:	
				pts_keep_index = np.where(np.logical_or(pts_anno[2, :] == 1, pts_anno[2, :] == 0))[0].tolist()
				if len(pts_keep_index) <= 0:		# if no point is annotated in current image
					count_skip_num_images += 1
					continue
				pts_anno = pts_anno[0:2, pts_keep_index]		
				pts_prediction = pts_prediction[:, pts_keep_index]
			
			# to avoid the point location includes the score or occlusion channel, only take the first two channels here	
			if pts_prediction.shape[0] == 3 or pts_prediction.shape[0] == 4:
				pts_prediction = pts_prediction[0:2, :]

			num_pts_tmp = len(pts_keep_index)
			if debug:
				assert pts_anno.shape[1] <= num_pts, 'number of points is not correct: %d vs %d' % (pts_anno.shape[1], num_pts)
				assert pts_anno.shape == pts_prediction.shape, 'shape of annotations and predictions are not the same {} vs {}'.format(print_np_shape(pts_anno, debug=debug), print_np_shape(pts_prediction, debug=debug))
				# print 'number of points to keep is %d' % num_pts_tmp

			# calculate bbox for normalization
			if normalization_ced or normalization_vec:
				assert len(pts_keep_index) == num_pts, 'some points are not annotated. Normalization on PCK curve is not allowed.'
				bbox_anno = pts2bbox(pts_anno, debug=debug)							# 1 x 4
				bbox_TLWH = bbox_TLBR2TLWH(bbox_anno, debug=debug)					# 1 x 4
				bbox_size = math.sqrt(bbox_TLWH[0, 2] * bbox_TLWH[0, 3])			# scalar
			
			# calculate normalized error for all points
			normed_mean_error, _ = pts_euclidean(pts_prediction, pts_anno, debug=debug)	# scalar
			if normalization_ced:
				normed_mean_error /= bbox_size
			normed_mean_error_total[count] = normed_mean_error
			mse_error_dict[image_path] = normed_mean_error

			if normed_mean_error == 0:
				print pts_prediction
				print pts_anno

			# calculate normalized error point specifically
			for pts_index in xrange(num_pts):
				if pts_index in pts_keep_index:			# if current point not annotated in current image, just keep 0
					normed_mean_error_pts_specific_valid[count, pts_index] = True
				else:
					continue

				pts_index_from_keep_list = pts_keep_index.index(pts_index)
				pts_prediction_tmp = np.reshape(pts_prediction[:, pts_index_from_keep_list], (2, 1))
				pts_anno_tmp = np.reshape(pts_anno[:, pts_index_from_keep_list], (2, 1))
				normed_mean_error_pts_specifc_tmp, _ = pts_euclidean(pts_prediction_tmp, pts_anno_tmp, debug=debug)

				if normalization_ced:
					normed_mean_error_pts_specifc_tmp /= bbox_size
				normed_mean_error_pts_specific[count, pts_index] = normed_mean_error_pts_specifc_tmp

			# calculate the point error vector
			error_vector = pts_prediction - pts_anno 			# 2 x num_pts_tmp
			if normalization_vec:
				error_vector /= bbox_size
			pts_error_vec_pts_specific[count, :, pts_keep_index] = np.transpose(error_vector)
			pts_error_vec[count, :] = np.sum(error_vector, axis=1) / num_pts_tmp

			count += 1

		print 'number of skipped images is %d' % count_skip_num_images
		assert count + count_skip_num_images == num_images, 'all cells in the array must be filled %d vs %d' % (count + count_skip_num_images, num_images)
		# print normed_mean_error_total
		# time.sleep(1000)

		# save results to dictionary
		normed_mean_error_dict[method_name] = normed_mean_error_total[:count]
		normed_mean_error_pts_specific_dict[method_name] = normed_mean_error_pts_specific[:count, :]
		normed_mean_error_pts_specific_valid_dict[method_name] = normed_mean_error_pts_specific_valid[:count, :]
		pts_error_vec_dict[method_name] = np.transpose(pts_error_vec[:count, :])												# 2 x num_images
		pts_error_vec_pts_specific_dict[method_name] = pts_error_vec_pts_specific[:count, :, :]
		mse_error_dict_dict[method_name] = mse_error_dict

	# calculate mean value
	if mse:
		mse_value = dict()		# dictionary to record all average MSE for different methods
		mse_dict = dict()		# dictionary to record all point-wise MSE for different keypoints
		for method_name, error_array in normed_mean_error_dict.items():
			mse_value[method_name] = np.mean(error_array)
	else:
		mse_value = None

	# save mse error list to file for each method
	error_list_savedir = os.path.join(save_path, 'error_list')
	mkdir_if_missing(error_list_savedir)
	for method_name, mse_error_dict in mse_error_dict_dict.items():
		mse_error_list_path = os.path.join(error_list_savedir, 'error_%s.txt' % method_name)
		mse_error_list = open(mse_error_list_path, 'w')
		
		sorted_tuple_list = sorted(mse_error_dict.items(), key=operator.itemgetter(1), reverse=True)
		for tuple_index in range(len(sorted_tuple_list)):
			image_path_tmp = sorted_tuple_list[tuple_index][0]
			mse_error_tmp = sorted_tuple_list[tuple_index][1]
			mse_error_list.write('{:<200} {}\n'.format(image_path_tmp, '%.2f' % mse_error_tmp))
		mse_error_list.close()
		print '\nsave mse error list for %s to %s' % (method_name, mse_error_list_path)

	# visualize the ced (cumulative error distribution curve)
	print('visualizing pck curve....\n')
	pck_savedir = os.path.join(save_path, 'pck')
	mkdir_if_missing(pck_savedir)
	pck_savepath = os.path.join(pck_savedir, 'pck_curve_overall.png')
	table_savedir = os.path.join(save_path, 'metrics')
	mkdir_if_missing(table_savedir)
	table_savepath = os.path.join(table_savedir, 'detailed_metrics_overall.txt')
	_, metrics_all = visualize_ced(normed_mean_error_dict, error_threshold=error_threshold, normalized=normalization_ced, truncated_list=truncated_list, title='2D PCK curve (all %d points)' % num_pts, display_list=display_list, debug=debug, vis=vis, save=save, pck_savepath=pck_savepath, table_savepath=table_savepath)
	metrics_title = ['Method Name / Point Index']
	ptswise_mse_table = [[normed_mean_error_pts_specific_dict.keys()[index_tmp]] for index_tmp in xrange(num_methods)]
	for pts_index in xrange(num_pts):
		metrics_title.append(str(pts_index + 1))
		normed_mean_error_dict_tmp = dict()

		for method_name, error_array in normed_mean_error_pts_specific_dict.items():
			normed_mean_error_pts_specific_valid_temp = normed_mean_error_pts_specific_valid_dict[method_name]
			
			# Some points at certain images might not be annotated. When calculating MSE for these specific point, we remove those images to avoid "false" mean average error
			valid_array_per_pts_per_method = np.where(normed_mean_error_pts_specific_valid_temp[:, pts_index] == True)[0].tolist()
			error_array_per_pts = error_array[:, pts_index]
			error_array_per_pts = error_array_per_pts[valid_array_per_pts_per_method]
			num_image_tmp = len(valid_array_per_pts_per_method)
			normed_mean_error_dict_tmp[method_name] = np.reshape(error_array_per_pts, (num_image_tmp, ))
		pck_savepath = os.path.join(pck_savedir, 'pck_curve_pts_%d.png' % (pts_index+1))
		table_savepath = os.path.join(table_savedir, 'detailed_metrics_pts_%d.txt' % (pts_index+1))
		metrics_dict, _ = visualize_ced(normed_mean_error_dict_tmp, error_threshold=error_threshold, normalized=normalization_ced, truncated_list=truncated_list, display2terminal=False, title='2D PCK curve for point %d' % (pts_index+1), display_list=display_list, debug=debug, vis=vis, save=save, pck_savepath=pck_savepath, table_savepath=table_savepath)
		for method_index in range(num_methods):
			method_name = normed_mean_error_pts_specific_dict.keys()[method_index]
			ptswise_mse_table[method_index].append('%.1f' % metrics_dict[method_name]['MSE'])
	
	# reorder the table
	order_index_list = [display_list.index(method_name_tmp) for method_name_tmp in normed_mean_error_pts_specific_dict.keys()]
	order_index_list = [0] + [order_index_tmp + 1 for order_index_tmp in order_index_list]

	# print table to terminal
	ptswise_mse_table = list_reorder([metrics_title] + ptswise_mse_table, order_index_list, debug=debug)
	table = AsciiTable(ptswise_mse_table)
	print '\nprint point-wise average MSE'
	print table.table
	# save table to file
	ptswise_savepath = os.path.join(table_savedir, 'pointwise_average_MSE.txt')
	table_file = open(ptswise_savepath, 'w')
	table_file.write(table.table)
	table_file.close()
	print '\nsave point-wise average MSE to %s' % ptswise_savepath

	# visualize the error vector map
	print('visualizing error vector distribution map....\n')
	error_vec_save_dir = os.path.join(save_path, 'error_vec')
	mkdir_if_missing(error_vec_save_dir)
	savepath_tmp = os.path.join(error_vec_save_dir, 'error_vector_distribution_all.png')
	visualize_pts(pts_error_vec_dict, title='Point Error Vector Distribution (all %d points)' % num_pts, mse=mse, mse_value=mse_value, display_range=display_range, display_list=display_list, xlim=xlim, ylim=ylim, covariance=covariance, debug=debug, vis=vis, save=save, save_path=savepath_tmp)
	for pts_index in xrange(num_pts):
		pts_error_vec_pts_specific_dict_tmp = dict()
		for method_name, error_vec_dict in pts_error_vec_pts_specific_dict.items():
			pts_error_vec_pts_specific_valid = normed_mean_error_pts_specific_valid_dict[method_name]		# get valid flag
			valid_image_index_per_pts = np.where(pts_error_vec_pts_specific_valid[:, pts_index] == True)[0].tolist()		# get images where the points with current index are annotated

			pts_error_vec_pts_specific_dict_tmp[method_name] = np.transpose(error_vec_dict[valid_image_index_per_pts, :, pts_index])		# 2 x num_images
		savepath_tmp = os.path.join(error_vec_save_dir, 'error_vector_distribution_pts_%d.png' % (pts_index+1))
		if mse:
			mse_dict_tmp = visualize_pts(pts_error_vec_pts_specific_dict_tmp, title='Point Error Vector Distribution for Point %d' % (pts_index+1), mse=mse, display_range=display_range, display_list=display_list, xlim=xlim, ylim=ylim, covariance=covariance, debug=debug, vis=vis, save=save, save_path=savepath_tmp)
			mse_best = min(mse_dict_tmp.values())
			mse_single = dict()
			mse_single['mse'] = mse_best
			mse_single['num_images'] = len(valid_image_index_per_pts)			# assume number of valid images is equal for all methods
			mse_dict[pts_index] = mse_single
		else:
			visualize_pts(pts_error_vec_pts_specific_dict_tmp, title='Point Error Vector Distribution for Point %d' % (pts_index+1), mse=mse, display_range=display_range, display_list=display_list, xlim=xlim, ylim=ylim, covariance=covariance, debug=debug, vis=vis, save=save, save_path=savepath_tmp)

	# save mse to json file for further use
	if mse:
		json_path = os.path.join(save_path, 'mse_pts.json')

		# if existing, compare and select the best
		if is_path_exists(json_path):
			with open(json_path, 'r') as file:
				mse_dict_old = json.load(file)
				file.close()

			for pts_index, mse_single in mse_dict_old.items():
				mse_dict_new = mse_dict[int(pts_index)]
				mse_new = mse_dict_new['mse']
				if mse_new < mse_single['mse']:
					mse_single['mse'] = mse_new
				mse_dict_old[pts_index] = mse_single

			with open(json_path, 'w') as file:
				print('overwrite old mse to {}'.format(json_path))
				json.dump(mse_dict_old, file)
				file.close()

		else:
			with open(json_path, 'w') as file:
				print('save mse for all keypoings to {}'.format(json_path))
				json.dump(mse_dict, file)
				file.close()

	print('\ndone!!!!!\n')

	return metrics_all, ptswise_mse_table