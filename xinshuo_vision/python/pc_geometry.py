# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np, cv2
from sklearn.neighbors import NearestNeighbors

def __pc_fit_transform(pc1, pc2):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
		A: Nxm numpy array of corresponding points
		B: Nxm numpy array of corresponding points
	Returns:
		T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
		R: mxm rotation matrix
		t: mx1 translation vector
    '''
    # assert pc1.shape == pc2.shape

    # get number of dimensions
    m = pc1.shape[1]

    # translate points to their centroids
    centroid1 = np.mean(pc1, axis=0)
    centroid2 = np.mean(pc2, axis=0)
    centered_pc1 = pc1 - centroid1				# N x 3
    centered_pc2 = pc2 - centroid2				# N x 3

    # rotation matrix
    H = np.dot(centered_pc1.transpose(), centered_pc2)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.transpose(), U.transpose())

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1, :] *= -1
       R = np.dot(Vt.transpose(), U.transpose())			# 3 x 3

    t = centroid2.T - np.dot(R, centroid1.transpose())		# translation, (3, )
    
    # print(centered_pc1.shape)

    tranformed_centered_pc1 = np.dot(R, centered_pc1.transpose()).transpose() + t 			# N x 3
    normalization_factor = sum(a.transpose().dot(a) for a in tranformed_centered_pc1)
    s = sum(b.transpose().dot(a) for a, b in zip(tranformed_centered_pc1, centered_pc2))  		# compute scaling
    s = s / normalization_factor
    
    # t = centroid2.T - s * np.dot(R, centroid1.T)		# translation, (3, )
    # t = centroid2.transpose() - np.dot(R, centroid1.transpose())

    # homogeneous transformation
    T = np.identity(m + 1)			# 4 x 4
    T[:m, :m] = R
    T[:m, m] = t
  
    return T, R, t, s
    # return T, R, t

def __nearest_neighbor(src, dst):
	'''
	Find the nearest (Euclidean) neighbor in dst for each point in src
	Input:
		src: Nxm array of points
		dst: Nxm array of points
	Output:
		distances: Euclidean distances of the nearest neighbor
		indices: dst indices of the nearest neighbor
	'''

	# assert src.shape == dst.shape
	neigh = NearestNeighbors(n_neighbors=1)
	neigh.fit(dst)
	distances, indices = neigh.kneighbors(src, return_distance=True)
	return distances.ravel(), indices.ravel()

def pc_icp(pc1, pc2, init_pose=None, max_iterations=10000, tolerance=0.0000001):
	'''
	The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
	Input:
		A: Nxm numpy array of source mD points
		B: Nxm numpy array of destination mD point
		init_pose: (m+1)x(m+1) homogeneous transformation
		max_iterations: exit algorithm after max_iterations
		tolerance: convergence criteria
	Output:
		T: final homogeneous transformation that maps A on to B, (m+1) X (m+1)
		distances: Euclidean distances (errors) of the nearest neighbor
		iter_tmp: number of iterations to converge
	'''
	# assert pc1.shape == pc2.shape

	# get number of dimensions
	m = pc1.shape[1]

	# make points homogeneous, copy them to maintain the originals
	src = np.ones((m + 1, pc1.shape[0]))
	dst = np.ones((m + 1, pc2.shape[0]))
	src[:m, :] = np.copy(pc1.T)				# 4 x N
	dst[:m, :] = np.copy(pc2.T)

	# apply the initial pose estimation
	if init_pose is not None: src = np.dot(init_pose, src)
	prev_error = 0
	pc_transformed_list = []
	mean_error_list = []
	for iter_tmp in range(max_iterations):
		distances, indices = __nearest_neighbor(src[:m, :].transpose(), dst[:m, :].transpose())	# find the nearest neighbors between the current source and destination points
		
		# check error
		mean_error = np.mean(distances)
		mean_error_list.append(mean_error)
		if np.abs(prev_error - mean_error) < tolerance: break
		prev_error = mean_error

		# compute the transformation between the current source and nearest destination points
		T, _, _, _ = __pc_fit_transform(src[:m, :].transpose(), dst[:m, indices].transpose())
		# T, R, t, s = pc_fit_transform(src[:m, :].transpose(), dst[:m, indices].transpose())
		# T = cv2.estimateRigidTransform(src[:m, :].transpose(), dst[:m, indices].transpose(), False)

		# update the current source
		src = np.dot(T, src)
		# print(src[:, -1])

		# zxc
		# src = np.dot(T, src) * s
		# src = cv2.transform(src, T)
		pc_transformed_list.append(src.copy())

	# calculate final transformation
	# T, _, _, s = pc_fit_transform(pc1, src[:m, :].transpose())
	T, _, _, _ = __pc_fit_transform(pc1, src[:m, :].transpose())

	# return T, iter_tmp, pc_transformed_list, mean_error_list, s
	return T, iter_tmp, pc_transformed_list, mean_error_list

def pc_icp_scale(pc1, pc2, init_pose=None, max_iterations=10000, tolerance=0.0000001):
	'''
	The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
	Input:
		A: Nxm numpy array of source mD points
		B: Nxm numpy array of destination mD point
		init_pose: (m+1)x(m+1) homogeneous transformation
		max_iterations: exit algorithm after max_iterations
		tolerance: convergence criteria
	Output:
		T: final homogeneous transformation that maps A on to B, (m+1) X (m+1)
		distances: Euclidean distances (errors) of the nearest neighbor
		iter_tmp: number of iterations to converge
	'''
	# assert pc1.shape == pc2.shape

	# get number of dimensions
	m = pc1.shape[1]

	# make points homogeneous, copy them to maintain the originals
	src = np.ones((m + 1, pc1.shape[0]))
	dst = np.ones((m + 1, pc2.shape[0]))
	src[:m, :] = np.copy(pc1.T)				# 4 x N
	dst[:m, :] = np.copy(pc2.T)

	# apply the initial pose estimation
	if init_pose is not None: src = np.dot(init_pose, src)
	prev_error = 0
	pc_transformed_list = []
	mean_error_list = []
	for iter_tmp in range(max_iterations):
		distances, indices = __nearest_neighbor(src[:m, :].transpose(), dst[:m, :].transpose())	# find the nearest neighbors between the current source and destination points
		
		# check error
		mean_error = np.mean(distances)
		mean_error_list.append(mean_error)
		if np.abs(prev_error - mean_error) < tolerance: break
		prev_error = mean_error

		# compute the transformation between the current source and nearest destination points
		# T, _, _ = pc_fit_transform(src[:m, :].transpose(), dst[:m, indices].transpose())
		T, _, _, s = __pc_fit_transform(src[:m, :].transpose(), dst[:m, indices].transpose())
		# T = cv2.estimateRigidTransform(src[:m, :].transpose(), dst[:m, indices].transpose(), False)

		# squeeze to one point

		# update the current source
		# src = np.dot(T, src)
		# print(src[:, -1])

		# zxc
		src = np.dot(T, src)
		src[:m, :] = src[:m, :] * s 		
		# src = cv2.transform(src, T)
		pc_transformed_list.append(src.copy())

	# calculate final transformation
	# T, _, _, s = pc_fit_transform(pc1, src[:m, :].transpose())
	T, _, _, s = __pc_fit_transform(pc1, src[:m, :].transpose())

	return T, iter_tmp, pc_transformed_list, mean_error_list, s
	# return T, iter_tmp, pc_transformed_list, mean_error_list