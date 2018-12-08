# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np, init_paths, time
from pc_geometry import pc_icp, pc_icp_scale
from mesh_geometry import obj2ply_trimesh
from sklearn.neighbors import NearestNeighbors
# from open3d import PointCloud, Vector3dVector, draw_geometries, write_point_cloud, read_point_cloud, registration_icp, TransformationEstimationPointToPoint
from open3d import *
from xinshuo_math import construct_3drotation_matrix_rodrigue
from xinshuo_io import load_2dmatrix_from_file

# Constants
N = 1000                                    # number of random points in the dataset
num_tests = 10                               # number of test iterations
dim = 3                                     # number of dimensions of the points
noise_sigma = .0001                           # standard deviation error to be added
translation = 5                            # max translation of the test set
rotation = 1                               # max rotation (radians) of the test set

# def test_pc_fit_transform():
# 	# Generate a random dataset
# 	A = np.random.rand(N, dim)
# 	total_time = 0
# 	for i in range(num_tests):
# 		B = np.copy(A)

# 		# Translate
# 		t = np.random.rand(dim)*translation
# 		B += t

# 		# Rotate
# 		R = construct_3drotation_matrix_rodrigue(np.random.rand(dim), np.random.rand()*rotation)
# 		B = np.dot(R, B.T).T

# 		# Add noise
# 		B += np.random.randn(N, dim) * noise_sigma

# 		# Find best fit transform
# 		start = time.time()
# 		T, R1, t1 = pc_fit_transform(B, A)
# 		# T, R1, t1, s = pc_fit_transform(B, A)
# 		total_time += time.time() - start

# 		# Make C a homogeneous representation of B
# 		C = np.ones((N, 4))
# 		C[:,0:3] = B

# 		# Transform C
# 		C = np.dot(T, C.T).T


# 		# print(C[:, 0:3])
# 		# print(A)

# 		# assert np.allclose(C[:,0:3], A, atol=6*noise_sigma) # T should transform B (or C) to A
# 		# assert np.allclose(-t1, t, atol=6*noise_sigma)      # t and t1 should be inverses
# 		# assert np.allclose(R1.T, R, atol=6*noise_sigma)     # R and R1 should be inverses

# 	print('best fit time: {:.3}'.format(total_time/num_tests))

def test_pc_icp_random():
	total_time = 0
	A = np.random.rand(N, dim)		# Generate a random dataset
	
	# Translate, Rotate and add noise
	B = np.copy(A)					# N x 3
	t = np.random.rand(dim) * translation 		#
	B += t
	R = construct_3drotation_matrix_rodrigue(np.random.rand(dim), np.random.rand() * rotation)
	B = np.dot(R, B.transpose()).transpose()
	B += np.random.randn(N, dim) * noise_sigma
	np.random.shuffle(B)		# Shuffle to disrupt correspondence

	# Run ICP
	start = time.time()
	T, num_iter, mean_error_list, pc_transformed_list, scale = pc_icp_scale(B, A, tolerance=0.001)
	# T, num_iter, mean_error_list, pc_transformed_list = pc_icp(B, A, tolerance=0.0001)
	total_time += time.time() - start
	# print(T.shape)

	# Make C a homogeneous representation of B
	C = np.ones((N, 4))
	C[:, 0:3] = np.copy(B)			# N x 4
	C = np.dot(T, C.transpose()).transpose()
	print(C.shape)
	C = C[:, :4] * scale
	# assert np.mean(distances) < 6*noise_sigma                   # mean error should be small
	# assert np.allclose(T[0:3,0:3].T, R, atol=6*noise_sigma)     # T and R should be inverses
	# assert np.allclose(-T[0:3,3], t, atol=6*noise_sigma)        # T and t should be inverses

	# visualization
	print('number of iteration: %d' % num_iter)
	print('icp time: {:.3}'.format(total_time))
	pcd1 = PointCloud()
	pcd1.points = Vector3dVector(A)
	pcd1.paint_uniform_color([1, 0, 0])
	write_point_cloud('pc_target.ply', pcd1)
	pcd2 = PointCloud()
	pcd2.points = Vector3dVector(B)
	pcd2.paint_uniform_color([0, 1, 0])
	write_point_cloud('pc_source.ply', pcd2)
	pcd3 = PointCloud()
	pcd3.points = Vector3dVector(C[:, 0:3])
	pcd3.paint_uniform_color([0, 1, 0])
	write_point_cloud('pc_registered.ply', pcd3)

def test_pc_icp_given():
	total_time = 0
	filename = '004'
	filepath = 'archive/%s.txt' % filename
	pc1, _ = load_2dmatrix_from_file(filepath)
	# print(pc1.shape)
	np.save(filename+'.npy', pc1)
	pcd1 = PointCloud()
	pcd1.points = Vector3dVector(pc1)
	# obj2ply_trimesh('car.obj', 'car.ply')
	pcd1.paint_uniform_color([0, 1, 0])
	write_point_cloud('partial_car%s.ply' % filename, pcd1)

	pcd2 = read_point_cloud('archive/car.ply')
	pc2 = np.array(pcd2.points)
	pc2 *= 4.2
	print(pc2)
	print(pc1)

	np.save('car.npy', pc2)



	# init_trans = np.identity(4)
	# print(init_trans.shape)
	# reg_p2p = registration_icp(pcd2, pcd1, 0.02, init_trans, TransformationEstimationPointToPoint())
	# print(reg_p2p.transformation)
	# pcd2.transform(reg_p2p.transformation)
	# write_point_cloud('aligned_car.ply', pcd2)
	# why identity??????                      



	# Run ICP
	start = time.time()
	T, num_iter, distances, iterations = pc_icp(pc2, pc1, tolerance=0.0001)
	total_time += time.time() - start

	# Make C a homogeneous representation of B
	pc3 = np.ones((pc2.shape[0], 4))
	pc3[:, 0:3] = np.copy(pc2)
	pc3 = np.dot(T, pc3.T).T

	# visualization
	print('number of iteration: %d' % num_iter)
	print('icp time: {:.3}'.format(total_time))
	pcd3 = PointCloud()
	pcd3.points = Vector3dVector(pc3[:, 0:3])
	pcd3.paint_uniform_color([1, 0, 0])
	write_point_cloud('aligned_car%s.ply' % filename, pcd3)

if __name__ == "__main__":
	# test_pc_fit_transform()
	# test_pc_icp_random()
	test_pc_icp_given()