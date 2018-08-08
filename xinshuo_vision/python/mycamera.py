# Author: Xinshuo
# Email: xinshuow@cs.cmu.edu

import numpy as np
from xinshuo_miscellaneous import isnparray, strlist2floatlist, isstring
from xinshuo_io import load_txt_file

class MyCamera(object):
	def __init__(self, intrinsics, extrinsics, distortion=None, camera_id=None, warning=True, debug=True):
		self.intrinsics = intrinsics
		self.extrinsics = extrinsics
		if distortion is not None: 
			# to make it full 
			self.distortion = distortion
		if camera_id is not None: 
			assert isstring(camera_id), 'the camera id is a string'
			self.camera_id = camera_id
		
		if debug:
			assert isnparray(self.intrinsics) and self.intrinsics.shape == (3, 3), 'the intrinsics is not correct'
			assert isnparray(self.extrinsics) and self.extrinsics.shape == (3, 4), 'the extrinsics is not correct'

	def get_projection_matrix(self):
		return np.matmul(self.intrinsics, self.extrinsics)





def load_camera_cluster(calibration_file, warning=True, debug=True):
	# assume the camera format is
	# camera id
	# 3 x 3 for intrinsics
	# 1 x N for distortion
	# 3 x 4 for extrinsics
	print('load camera file from %s' % calibration_file)
	camera_data, num_lines = load_txt_file(calibration_file, debug=debug)
	num_lines_each_camera = 9
	camera_cluster = dict()
	for line_index in range(0, num_lines, num_lines_each_camera):
		line_offset = 0
		camera_data_tmp = camera_data[line_index : line_index + num_lines_each_camera]
		# print(camera_data_tmp)
		# line_array = line.split(' ')
		# if len(line_array) == 1:
		
		camera_id_tmp = camera_data_tmp[0].split(' ')[0]
		assert not (camera_id_tmp in camera_cluster), 'the camera id %s is already in the camera cluster' % (camera_id_tmp)
		line_offset += 1
		# print(camera_id_tmp)

		intrinsics_tmp = np.zeros((3, 3), 'float32')
		for intrinsics_line_index in range(3):
			intrinsics_tmp[intrinsics_line_index, 0:3] = np.array(strlist2floatlist(camera_data_tmp[line_offset + intrinsics_line_index].split(' '), warning=warning, debug=debug))
		line_offset += 3
		# print(intrinsics_tmp.dtype)
		# aaa
		
		distortion_tmp = np.array(strlist2floatlist(camera_data_tmp[line_offset].split(' '), warning=warning, debug=debug))
		# print(distortion.shape)
		# num_ele_distort = distortion.size
		# print(num_ele_distort)
		distortion_tmp = np.reshape(distortion_tmp, (1, distortion_tmp.size))
		line_offset += 1

		extrinsics_tmp = np.zeros((3, 4), 'float32')
		for extrinsics_line_index in range(3):
			extrinsics_tmp[extrinsics_line_index, 0:4] = np.array(strlist2floatlist(camera_data_tmp[line_offset + extrinsics_line_index].split(' '), warning=warning, debug=debug))
		line_offset += 3
		# print(extrinsics_tmp)

		camera_cluster[camera_id_tmp] = MyCamera(intrinsics_tmp, extrinsics_tmp, distortion_tmp, camera_id_tmp, warning=warning, debug=debug)

	return camera_cluster