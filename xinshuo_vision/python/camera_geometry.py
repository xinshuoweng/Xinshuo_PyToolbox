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

	# initialization
	num_pts = pts_array1.shape[1]
	pts_array1[0:2, :] = pts_array1[0:2, :] / float(scaling_factor)
	pts_array2[0:2, :] = pts_array2[0:2, :] / float(scaling_factor)
	# print(pts_array1)
	# print(pts_array2)
	# aa

	# least square
	p1T1 = projection1[0, :]		# 1 x 4
	p1T2 = projection1[1, :]
	p1T3 = projection1[2, :]
	p2T1 = projection2[0, :]
	p2T2 = projection2[1, :]
	p2T3 = projection2[2, :]

	print(projection1)
	print(projection2)
	condition_matrix = np.array([[1e-3, 0, 0, 0],			# 4 x 4
								 [0, 1e-3, 0, 0],
								 [0, 0, 1e-3, 0],
								 [0, 0, 0, 1e-6]])

	# A = [projection1; projection2];
	# H = (A'*A)\A';
	pts_3d = np.zeros((num_pts, 4), 'float32')			# N x 4
	pts_3d.fill(-1)
	p1_proj = pts_array1.copy().transpose()				# N x 3
	p2_proj = pts_array2.copy().transpose()
	for i in range(num_pts):
		if pts_array1[2, i] == -1 or pts_array2[2, i] == -1: continue
		# print(pts_array1[1, i].shape)
		# print(p1T3.shape)
		U = np.zeros((6, 4), dtype='float32')			# 6 x 4

		U[0, :] = np.multiply(pts_array1[1, i], p1T3) - p1T2									# y * p1T3 - p1T2
		U[1, :] = p1T1 - np.multiply(pts_array1[0, i], p1T3)									# p1T1 - x * p1T3
		U[2, :] = np.multiply(pts_array1[0, i], p1T2) - np.multiply(pts_array1[1, i], p1T1)		# x * p1T2 - y * p1T1
		U[3, :] = np.multiply(pts_array2[1, i], p2T3) - p2T2									# y * p2T3 - p2T2
		U[4, :] = p2T1 - np.multiply(pts_array2[0, i], p2T3)									# p2T1 - x * p2T3
		U[5, :] = np.multiply(pts_array2[0, i], p2T2) - np.multiply(pts_array2[1, i], p2T1)		# x * p2T2 - y * p2T1

		print(U)
		# aa
		print(condition_matrix)
		
		conditioned_U = np.matmul(U, condition_matrix)
		print(conditioned_U)
		# aa

		_, _, V = np.linalg.svd(U)
		# conditioned_pts_3d[i, :] = V[:, -1].transpose()			# 1 x 4
		conditioned_pts_3d = V[:, -1]			# 4 x 1
		pts_3d[i, :] = np.matmul(condition_matrix, conditioned_pts_3d.reshape((4, 1))).transpose()

		# print(pts_3d)

		# aa
		#     b = [pts_array1(i, :)'; 1; pts_array2(i, :)'; 1];    
		#     pts_3d(i, :) = (H * b)'
		pts_3d[i, :] = np.divide(pts_3d[i, :], pts_3d[i, 3])		# N x 4

		# print(pts_3d[i, :].transpose().reshape((4, 1)).shape)
		# print(projection1.shape)
		# print(np.matmul(projection1, pts_3d[i, :].transpose().reshape((4, 1))).shape)

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
	# print(error)
	print(error_list)

	pts_3d = pts_3d[:, 0:3].transpose() 		# 3 x N
	return pts_3d, p1_proj, p2_proj