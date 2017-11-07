import os, time
import numpy as np
import torch
import json
from collections import OrderedDict
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

from xinshuo_python import isnonnegativeinteger, is2dptsarray, isstring, is2dptsarray_occlusion
from xinshuo_miscellaneous import print_log, print_np_shape
from xinshuo_math import pts_euclidean
# zero-indexed

class eval_meta_tracking_pts():
	def __init__(self, debug=True):
		self.debug = debug
		self.reset()

	def __repr__(self):
		return ('{name}'.format(name=self.__class__.__name__)+'(number of data = {})'.format(len(self)))

	def reset(self):
		self.pts_root = dict()					# 2 x num_pts
		self.pts_forward = dict()				# 2 x num_pts
		self.pts_backward = dict()				# 2 x num_pts
		self.pts_anno = dict()					# 2 x num_pts
		self.image_prev_path = dict()
		self.image_next_path = dict()

		self.pts_valid_index = dict()
		self.key_list = []
		self.index_dict = dict()				# given key, provide index
		self.length = 0

	def __len__(self):
		return self.length

	def convert_index2key(self, index):
		if self.debug:
			assert isnonnegativeinteger(index), 'the input index should be an non-negative integer'

		key = 'image%010d' % index
		return key

	def append(self, pts_root, pts_forward, pts_backward=None, pts_anno=None, image_prev_path=None, image_next_path=None, index=None):
		if index is None:
			index = self.length
		key = self.convert_index2key(index)
		if key in self.key_list:
			assert False, 'the data with index %d already exists' % index
		else:
			self.key_list.append(key)
			self.index_dict[key] = index

		if self.debug:
			assert is2dptsarray_occlusion(pts_root) or is2dptsarray(pts_root), 'the shape of input point array %s in the root frame is not correct' % print_np_shape(pts_root)
			assert is2dptsarray_occlusion(pts_forward) or is2dptsarray(pts_forward), 'the shape of input point array %s in the forward frame is not correct' % print_np_shape(pts_forward)
			assert is2dptsarray_occlusion(pts_backward) or is2dptsarray(pts_backward) or pts_backward is None, 'the shape of input point array %s in the backward frame is not correct' % print_np_shape(pts_backward)
			assert is2dptsarray_occlusion(pts_anno) or is2dptsarray(pts_anno) or pts_anno is None, 'the shape of input point array %s from the annotations is not correct' % print_np_shape(pts_anno)
			assert (isstring(image_prev_path) or image_prev_path is None) and (isstring(image_next_path) or image_next_path is None), 'the input image path is not correct'

		self.pts_root[key] = pts_root
		self.pts_forward[key] = pts_forward
		self.pts_backward[key] = pts_backward
		self.pts_anno[key] = pts_anno
		self.image_prev_path[key] = image_prev_path
		self.image_next_path[key] = image_next_path

		self.length += 1

	def save(self, save_path):
		if self.debug:
			assert is_path_exists_or_creatable(save_path), 'the save path is not correct'

		meta = {'pts_root':      	self.pts_root, 
				'pts_forward':   	self.pts_forward,
				'pts_backward':  	self.pts_backward,
				'pts_anno':     	self.pts_anno,
				'image_prev_path':  self.image_prev_path,
				'image_next_path':  self.image_next_path,
				'length':			self.length,
				'pts_valid_index':	self.pts_valid_index,
				}
		torch.save(meta, save_path)
		print('save point meta data into {}'.format(save_path))

	# def load(self, filename):
	# 	assert os.path.isfile(filename), '{} is not a file'.format(filename)
	# 	checkpoint       = torch.load(filename)
	# 	self.predictions = checkpoint['predictions']
	# 	self.groundtruth = checkpoint['groundtruth']
	# 	self.image_lists = checkpoint['image_lists']
	# 	self.mae_bars    = checkpoint['mae_bars']
	# 	self.spefic_errs = checkpoint['spefic_errs']

	def compute_valid_tracking(self, index, threshold=10):
		'''
		compute the success of the tracking based on threshold
		'''

		key = self.convert_index2key(index)
		pts_root_tmp, pts_anno_tmp, pts_forward_tmp, pts_backward_tmp = self.pts_root[key], self.pts_anno[key], self.pts_forward[key], self.pts_backward[key]
		if pts_anno_tmp is not None:			# compute the valid tracking based-on forward-backward check
			ave_dist, pts_distance = pts_euclidean(pts_forward_tmp, pts_anno_tmp, debug=self.debug)					# 1 x N
			# print 'using anno'
			# print pts_forward_tmp
			# print pts_anno_tmp
		else:								# compute the valid tracking based-on accuracy
			ave_dist, pts_distance = pts_euclidean(pts_root_tmp, pts_backward_tmp, debug=self.debug)
			# print 'using backward'

		valid_index = np.where(np.array(pts_distance) < threshold)[0].tolist()

		pts_forward_valid = pts_forward_tmp[:, valid_index].copy()
		pts_backward_valid = pts_backward_tmp[:, valid_index].copy()

		if pts_anno_tmp is not None:			
			ave_dist_valid, _ = pts_euclidean(pts_forward_valid, pts_anno_tmp[:, valid_index], debug=self.debug)
		else:								
			ave_dist_valid, _ = pts_euclidean(pts_root_tmp[:, valid_index], pts_backward_valid, debug=self.debug)		

		self.pts_valid_index[key] = valid_index
		return valid_index, pts_forward_valid, pts_backward_valid, ave_dist, ave_dist_valid

	def compute_success_rate(self, threshold=10):
		'''
		compute average valid points / tracked points over all images
		'''
		if self.debug:
			assert self.length > 0, 'no enough data'

		N = 1
		success_rate = 0
		for key in self.key_list:
			if not key in self.pts_valid_index:				# if valid index has not already been computed
				index_tmp = self.index_dict[key]
				self.compute_valid_tracking(index=index_tmp, threshold=threshold)

			valid_index_tmp = self.pts_valid_index[key]
			num_pts_tmp = self.pts_forward[key].shape[1]
			success_rate_tmp = valid_index_tmp * 1.0 / num_pts_tmp

			success_rate = (N-1)/N * success_rate + success_rate_tmp / N
			N += 1

		return success_rate

	def compute_error(self, method, threshold=10):
		'''
		compute the average accuracy over all images
		'''

		if self.debug:
			assert self.length > 0, 'no enough data'
			assert method == 'forward' or method == 'backward', 'the accuracy can only be computed based on forward with annotations or backward check'

		N = 1
		average_error = 0
		for key in self.key_list:
			if not key in self.pts_valid_index:				# if valid index has not already been computed
				index_tmp = self.index_dict[key]
				self.compute_valid_tracking(index=index_tmp, threshold=threshold)

			valid_index_tmp = self.pts_valid_index[key]
			if method == 'forward':
				pts_forward, pts_anno = self.pts_forward[key], self.pts_anno[key]
				if self.debug:
					assert pts_anno is not None
				pts_forward_valid, pts_anno_valid = pts_forward[:, valid_index_tmp], pts_anno[:, valid_index_tmp]
				error_tmp, _ = pts_euclidean(pts_forward_valid, pts_anno_valid, debug=self.debug)
			elif method == 'backward':
				pts_backward, pts_root = self.pts_backward[key], self.pts_root[key]
				if self.debug:
					assert pts_backward is not None
				pts_backward_valid, pts_root_valid = pts_backward[:, valid_index_tmp], pts_root[:, valid_index_tmp]
				error_tmp, _ = pts_euclidean(pts_backward_valid, pts_root_valid, debug=self.debug)

			average_error = (N-1)/N * average_error + error_tmp / N
			N += 1

		return average_error

  # def weight_mse(self, distances, avaliable):
  #   num_point = distances.shape[1]
  #   each_mses = np.zeros((num_point), dtype=float)
  #   for i in range(num_point):
  #     dis = distances[:, i]
  #     dis = dis[avaliable[:,i]]
  #     each_mses[i] = dis.mean()
  #   return each_mses

  # def compute_threshold(self, all_ava, all_mse, thresh):
  #   all_ava = all_ava.astype('bool')
  #   all_mse = all_mse.copy()
  #   oks = all_mse < thresh
  #   accuracy = (all_ava * oks).sum() * 1.0 / all_ava.sum()
  #   all_mse[ all_mse >= thresh ] = thresh
  #   avg = all_mse[ all_ava ].mean()
  #   return accuracy, avg

  # def specific_v0(self, avaliable, distances, log):
  #   ## compute total average mse
  #   num_points = distances.shape[1]
  #   assert num_points == len(self.spefic_errs) or self.spefic_errs is None
  #   scales = [1, 1.5, 2]

  #   total_average_mse = distances * avaliable.astype('float')
  #   print_log('Total  average mae : {:.2f}'.format(total_average_mse.sum()/avaliable.sum()) + ', Bars : {}  Scales : {}'.format(self.mae_bars, scales), log)
  #   acc, trucavg = self.compute_threshold(avaliable, distances, 80)
  #   print_log('Thresh average mae : {:.2f}, accuracy : {:5.2f}%'.format(trucavg, acc*100), log)

  #   for i in range(num_points):
  #     string = '--> {:2d}-th point || B'.format(i)
  #     for bar in self.mae_bars:
  #       acc, avg = self.compute_threshold(avaliable[:, i], distances[:, i], bar)
  #       string = string + ' [{:4.1f}]=({:5.2f},{:5.1f}%)'.format(bar, avg, acc*100)

  #     if self.spefic_errs is not None:      
  #       string = string + ' || S'
  #       for scale in scales:
  #         bar = self.spefic_errs[i] * scale
  #         acc, avg = self.compute_threshold(avaliable[:, i], distances[:, i], bar)
  #         string = string + ' [{:4.1f}]=({:5.2f}, {:5.1f}%)'.format(bar, avg, acc*100)

  #     print_log(string, log)
 
  #   return trucavg

  # def get_scale_acc(self):
  #   avaliable, distances = self.get_init()
  #   num_point = self.predictions[0].shape[1]
  #   p_erros = [28.13, 31.90, 26.32, 32.75, 14.16, 18.37, 11.89, 13.25, 13.06, 15.61, 13.54, 13.14, 17.63, 18.75, 13.57, 11.04, 12.76, 16.08, 7.88, 9.2]
  #   scales = [0.5, 1, 1.5, 2]
  #   scale_acc = np.zeros((len(scales), num_point), dtype='float32')
  #   for j, scale in enumerate(scales):
  #     for i in range(num_point):
  #       ava, dis = avaliable[:, i], distances[:, i]
  #       dis = dis[ava]
  #       scale_acc[j, i] = (dis < p_erros[i]*scales[j]).mean()
  #   return scale_acc

  # def get_avg_mae(self, thresh, points):
  #   avaliable, distances = self.get_init()
  #   _, tavg = self.compute_threshold(avaliable[:,points], distances[:,points], thresh)
  #   return tavg

  # def get_init(self, is_x_axis=None):
  #   assert (not self.predictions) == False, 'list is empty'
  #   num_image = len(self.predictions)
  #   num_point = self.predictions[0].shape[1]
    
  #   avaliable = np.zeros((num_image, num_point), dtype=bool)
  #   distances = np.zeros((num_image, num_point), dtype=float)
  #   for idx in range(num_image):
  #     gt = self.groundtruth[idx]
  #     xx = self.predictions[idx]
  #     avaliable[idx, :] = gt[2, :].astype('bool')
  #     xx = np.square(gt[:2,:] - xx[:2,:])
  #     if is_x_axis is None:
  #       distances[idx, :] = np.sqrt(xx[0] + xx[1])
  #     elif is_x_axis:
  #       distances[idx, :] = np.sqrt(xx[0])
  #     else:
  #       distances[idx, :] = np.sqrt(xx[1])
        
  #   return avaliable, distances

  # def general_output_mse(self, avaliable, distances, log):
  #   ## compute total average mse
  #   num_points = distances.shape[1]

  #   total_average_mse = distances * avaliable.astype('float')
  #   print_log('Total  average mae : {:.2f}'.format(total_average_mse.sum()/avaliable.sum()) + ', Bars : {}'.format(self.mae_bars), log)
  #   acc, trucavg = self.compute_threshold(avaliable, distances, 80)
  #   print_log('Thresh average mae : {:.2f}, accuracy : {:5.2f}%'.format(trucavg, acc*100), log)

  #   for i in range(num_points):
  #     string = '--> {:2d}-th point || '.format(i)
  #     for bar in self.mae_bars:
  #       acc, avg = self.compute_threshold(avaliable[:, i], distances[:, i], bar)
  #       string = string + ' [{:4.1f}]=({:5.2f},{:5.1f}%)'.format(bar, avg, acc*100)

  #     print_log(string, log)
 
  #   return trucavg

  # def compute_mse(self, log):
  #   avaliable, distances = self.get_init()
  #   num_image = len(self.predictions)
  #   num_point = self.predictions[0].shape[1]
  #   print_log('compute mse {} examples with {} points'.format(num_image, num_point), log)
  #   mae = self.general_output_mse(avaliable, distances, log)
  #   return mae
  
  # def calculate_acc(self, errors, points):
  #   avaliable, distances = self.get_init()
  #   assert isinstance(errors, list)
  #   assert isinstance(points, list)
  #   accuracies = np.zeros((len(errors), len(points)), dtype='float32')

  #   for i in range(len(points)):
  #     ava, dis = avaliable[:, points[i]], distances[:,points[i]]
  #     dis = dis[ava]
  #     for j, error in enumerate(errors):
  #       accuracies[j, i] = (dis < error).mean()
        
  #   return accuracies