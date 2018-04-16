# Author: Xinshuo
# Email: xinshuow@andrew.cmu.edu
import os, time, torch, json
import numpy as np

from xinshuo_miscellaneous import print_log, print_np_shape, isnonnegativeinteger, is2dptsarray, isstring, is2dptsarray_occlusion, is_path_exists_or_creatable, is_path_exists
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
		self.pts_valid_index = dict()			# given key, provide a list of valid index
		self.key_list = []						# current available keys
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
				'index_dict':		self.index_dict,
				'key_list':			self.key_list,
				}
		torch.save(meta, save_path)
		print('save point meta data into {}'.format(save_path))

	def load(self, save_path):
		if self.debug:
			assert is_path_exists(save_path), '{} is not a file'.format(save_path)

		print('load point meta data from {}'.format(save_path))			
		checkpoint       		= torch.load(save_path)
		self.pts_root 			= checkpoint['pts_root']
		self.pts_forward 		= checkpoint['pts_forward']
		self.pts_backward 		= checkpoint['pts_backward']
		self.pts_anno    		= checkpoint['pts_anno']
		self.image_prev_path 	= checkpoint['image_prev_path']
		self.image_next_path	= checkpoint['image_next_path']

		self.length 			= checkpoint['length']
		self.pts_valid_index 	= checkpoint['pts_valid_index']
		self.index_dict 		= checkpoint['index_dict']
		self.key_list 			= checkpoint['key_list']

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

		N = 1.0
		success_rate = 0
		for key in self.key_list:
			if not key in self.pts_valid_index:				# if valid index has not already been computed
				index_tmp = self.index_dict[key]
				self.compute_valid_tracking(index=index_tmp, threshold=threshold)

			valid_index_tmp = self.pts_valid_index[key]
			num_pts_tmp = self.pts_forward[key].shape[1]
			success_rate_tmp = len(valid_index_tmp) * 1.0 / num_pts_tmp

			success_rate = (N-1)/N * success_rate + success_rate_tmp / N
			N += 1.0

		return success_rate

	def compute_error(self, method, threshold=10):
		'''
		compute the average accuracy over all images
		'''

		if self.debug:
			assert self.length > 0, 'no enough data'
			assert method == 'forward' or method == 'backward', 'the accuracy can only be computed based on forward with annotations or backward check'

		N = 1.0
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
			N += 1.0

		return average_error