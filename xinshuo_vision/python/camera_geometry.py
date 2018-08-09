# Author: Xinshuo Weng
# Email: xinshuow@cs.cmu.edu
import numpy as np
from xinshuo_math.python.private import safe_2dptsarray
from xinshuo_math import pts_euclidean
from xinshuo_miscellaneous import is2dptsarray_confidence, is2dptsarray_occlusion, is2dptsarray

def triangulate_two_views(input_pts1, input_pts2, projection1, projection2, scaling_factor=1, warning=True, debug=True):
	'''
	triangulation from two views
	'''
	#       projection1 - 3 x 4 Camera Matrix 1
	#       pts_array1 - 3 x N set of points
	#       projection2 - 3 x 4 Camera Matrix 2
	#       pts_array2 - 3 x N set of points
	try: pts_array1 = safe_2dptsarray(input_pts1, homogeneous=True, warning=warning, debug=debug)
	except AssertionError: pts_array1 = safe_2dptsarray(input_pts1, homogeneous=False, warning=warning, debug=debug)
	if debug: assert is2dptsarray(pts_array1) or is2dptsarray_occlusion(pts_array1) or is2dptsarray_confidence(pts_array1), 'first input points are not correct'
	try: pts_array2 = safe_2dptsarray(input_pts2, homogeneous=True, warning=warning, debug=debug)
	except AssertionError: pts_array2 = safe_2dptsarray(input_pts2, homogeneous=False, warning=warning, debug=debug)
	if debug: assert is2dptsarray(pts_array2) or is2dptsarray_occlusion(pts_array2) or is2dptsarray_confidence(pts_array2), 'second input points are not correct'

	pts_array1 = pts_array1.astype('float32')
	pts_array2 = pts_array2.astype('float32')

	# initialization
	num_pts = pts_array1.shape[1]
	pts_array1[0:2, :] = pts_array1[0:2, :] / float(scaling_factor)
	pts_array2[0:2, :] = pts_array2[0:2, :] / float(scaling_factor)

	# least square
	p1T1 = projection1[0, :]		# 1 x 4
	p1T2 = projection1[1, :]
	p1T3 = projection1[2, :]
	p2T1 = projection2[0, :]
	p2T2 = projection2[1, :]
	p2T3 = projection2[2, :]
	# condition_matrix = np.array([[1e-3, 0, 0, 0],			# 4 x 4
	# 							 [0, 1e-3, 0, 0],
	# 							 [0, 0, 1e-3, 0],
	# 							 [0, 0, 0, 1e-6]])
	condition_matrix = np.array([[1, 0, 0, 0],			# 4 x 4
								 [0, 1, 0, 0],
								 [0, 0, 1, 0],
								 [0, 0, 0, 1]])
	pts_3d = np.zeros((num_pts, 4), 'float32')			# N x 4
	pts_3d.fill(-1)
	p1_proj = pts_array1.copy().transpose()				# N x 3
	p2_proj = pts_array2.copy().transpose()
	for i in range(num_pts):
		if pts_array1[2, i] == -1 or pts_array2[2, i] == -1: continue
		U = np.zeros((6, 4), dtype='float32')			# 6 x 4

		U[0, :] = np.multiply(pts_array1[1, i], p1T3) - p1T2									# y * p1T3 - p1T2
		U[1, :] = p1T1 - np.multiply(pts_array1[0, i], p1T3)									# p1T1 - x * p1T3
		U[2, :] = np.multiply(pts_array1[0, i], p1T2) - np.multiply(pts_array1[1, i], p1T1)		# x * p1T2 - y * p1T1
		U[3, :] = np.multiply(pts_array2[1, i], p2T3) - p2T2									# y * p2T3 - p2T2
		U[4, :] = p2T1 - np.multiply(pts_array2[0, i], p2T3)									# p2T1 - x * p2T3
		U[5, :] = np.multiply(pts_array2[0, i], p2T2) - np.multiply(pts_array2[1, i], p2T1)		# x * p2T2 - y * p2T1

		conditioned_U = np.matmul(U, condition_matrix)
		# print(conditioned_U)

		_, _, V = np.linalg.svd(conditioned_U)
		conditioned_pts_3d = V[-1, :]			# 4 x 1, V is the transpose version of V in matlab
		pts_3d[i, :] = np.matmul(condition_matrix, conditioned_pts_3d.reshape((4, 1))).transpose()
		pts_3d[i, :] = np.divide(pts_3d[i, :], pts_3d[i, 3])		# N x 4
		# print(pts_3d)

		# compute reprojection error
		p1_proj[i, :] = (np.matmul(projection1, pts_3d[i, :].transpose().reshape((4, 1)))).transpose()		# 1 x 3
		p2_proj[i, :] = (np.matmul(projection2, pts_3d[i, :].transpose().reshape((4, 1)))).transpose()
		p1_proj[i, :] = np.divide(p1_proj[i, :], p1_proj[i, -1])
		p2_proj[i, :] = np.divide(p2_proj[i, :], p2_proj[i, -1])
		# error = error + np.norm(p1_proj[i, 0:2] - pts_array1[0:2, i]) + np.norm(p2_proj[i, 0:2] - pts_array2[0:2, i])

	error_tmp, error_list = pts_euclidean(p1_proj[:, 0:2].transpose(), pts_array1[0:2, :], warning=warning, debug=debug)
	error = error_tmp * num_pts
	print(error_list)
	error_tmp, error_list = pts_euclidean(p2_proj[:, 0:2].transpose(), pts_array2[0:2, :], warning=warning, debug=debug)
	error += error_tmp * num_pts
	print(error_list)

	pts_3d = pts_3d[:, 0:3].transpose() 		# 3 x N
	return pts_3d, p1_proj.transpose(), p2_proj.transpose()


def triangulate_multiple_views(input_pts, projection, scaling_factor=1, warning=True, debug=True):
	'''
	triangulation from M views
	'''
	#       projection - M x 3 x 4 Camera Matrix 1
	#       pts_array - M x 3 x N set of points
	pts_array = input_pts.copy().astype('float32')
	projection = projection.copy()

	# initialization
	num_pts = pts_array.shape[2]
	num_cam = pts_array.shape[0]
	# pts_array[:, 0:2, :] = pts_array[:, 0:2, :] / float(scaling_factor)
	condition_matrix = np.array([[1e-4, 0, 0, 0],			# 4 x 4
								 [0, 1e-4, 0, 0],
								 [0, 0, 1e-4, 0],
								 [0, 0, 0, 1e-7]])
	# condition_matrix = np.array([[1, 0, 0, 0],			# 4 x 4
	# 							 [0, 1, 0, 0],
	# 							 [0, 0, 1, 0],
	# 							 [0, 0, 0, 1]])
	pts_3d = np.zeros((num_pts, 4), 'float32')			# N x 4
	pts_3d.fill(-1)
	pts_proj = pts_array.copy()							# M x 3 x N
	pts_merged = pts_array.copy()							# M x 3 x N
	for i in range(num_pts):
		count = 0
		cam_valid_list = []
		for cam_index in range(num_cam):
			if pts_array[cam_index, 2, i] == 1: 
				cam_valid_list.append(cam_index)
				count += 1
		if count < 2: continue 					# triangulation is valid until more than 1 point

		# print(i)
		# print(cam_valid_list)
		U = np.zeros((3 * num_cam, 4), dtype='float32')			# 6 x 4
		for cam_index in cam_valid_list:
			p1T1 = projection[cam_index, 0, :]		# 1 x 4
			p1T2 = projection[cam_index, 1, :]
			p1T3 = projection[cam_index, 2, :] 
			U[0 + cam_index*3, :] = np.multiply(pts_array[cam_index, 1, i], p1T3) - p1T2									# y * p1T3 - p1T2
			U[1 + cam_index*3, :] = p1T1 - np.multiply(pts_array[cam_index, 0, i], p1T3)									# p1T1 - x * p1T3
			U[2 + cam_index*3, :] = np.multiply(pts_array[cam_index, 0, i], p1T2) - np.multiply(pts_array[cam_index, 1, i], p1T1)		# x * p1T2 - y * p1T1

		conditioned_U = np.matmul(U, condition_matrix)
		_, _, V = np.linalg.svd(conditioned_U)
		conditioned_pts_3d = V[-1, :]			# 4 x 1, V is the transpose version of V in matlab
		pts_3d[i, :] = np.matmul(condition_matrix, conditioned_pts_3d.reshape((4, 1))).transpose()
		pts_3d[i, :] = np.divide(pts_3d[i, :], pts_3d[i, 3])		# N x 4

		# compute reprojection error
		for cam_index in range(num_cam):
			# print((np.matmul(projection[cam_index, :, :], pts_3d[i, :].transpose().reshape((4, 1)))).shape)
			pts_proj[cam_index, :, i] = np.matmul(projection[cam_index, :, :], pts_3d[i, :].transpose().reshape((4, 1))).reshape((3, ))		# 3 x 1
			pts_proj[cam_index, :, i] = np.divide(pts_proj[cam_index, :, i], pts_proj[cam_index, -1, i])
			if pts_array[cam_index, 2, i] != 1: 
				pts_merged[cam_index, :, i] = pts_proj[cam_index, :, i].copy()			# keep the original, replace the empty one

	for cam_index in range(num_cam):
		valid_index_list = np.where(pts_array[cam_index, 2, :] == 1)[0].tolist()
		# print(type(valid_index_list))
		# print(valid_index_list)
		# print(pts_proj[cam_index, 0:2, :])
		# print(pts_proj[cam_index, 0:2, valid_index_list])
		# bbb
		error_tmp, error_list = pts_euclidean(pts_proj[cam_index, 0:2, valid_index_list].transpose(), pts_array[cam_index, 0:2, valid_index_list].transpose(), warning=warning, debug=debug)
		# error_tmp, error_list = pts_euclidean(p1_proj[:, 0:2].transpose(), pts_array1[0:2, :], warning=warning, debug=debug)
		# error_tmp, error_list = pts_euclidean(p1_proj[:, 0:2].transpose(), pts_array1[0:2, :], warning=warning, debug=debug)
	# error = error_tmp * num_pts
		# print(error_list)

	pts_3d = pts_3d[:, 0:3].transpose() 		# 3 x N
	return pts_3d, pts_proj, pts_merged