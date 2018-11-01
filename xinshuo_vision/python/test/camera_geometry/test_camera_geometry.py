# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np, init_paths
from mycamera import MyCamera, load_camera_cluster
from camera_geometry import triangulate_two_views, triangulate_multiple_views
from xinshuo_io import anno_parser, load_image
from xinshuo_visualization import visualize_image_with_pts, visualize_pts_array

def test_triangulate_two_views():
	calibration_file = '../calibration.txt'
	camera_cluster = load_camera_cluster(calibration_file, warning=False)

	pts1_file = '../20180514--handsy--400015--0800.pts'
	pts_array1 = anno_parser(pts1_file)
	pts2_file = '../20180514--handsy--400053--0800.pts'
	pts_array2 = anno_parser(pts2_file)
	projection1 = camera_cluster['400015'].get_projection_matrix()
	projection2 = camera_cluster['400053'].get_projection_matrix()

	# pts_array1 = np.array([[54], [183], [1]]).astype('float32')
	# projection1 = np.array([[1520.4, 0, 302.3, 0], [0, 1525.9, 246.9, 0], [0, 0, 1, 0]])
	# pts_array2 = np.array([[54], [199], [1]])
	# projection2 = np.array([[1520.9, -27.4, 298.7, 5.3], [-49.3, 1410.6, 630.2, -1488.8], [0, -0.3, 1, 0.2]])

	pts_3d, pts1_reproj, pts2_reproj = triangulate_two_views(pts_array1, pts_array2, projection1, projection2, scaling_factor=1)
	img_path = '../20180514--handsy--400015--0800.png'
	img = load_image(img_path)
	debug = False
	fig, ax = visualize_image_with_pts(img, pts_array1, label=True, debug=debug, closefig=False, vis=True)
	visualize_image_with_pts(img, pts1_reproj, label=True, debug=debug, closefig=False, vis=True)
	# visualize_pts_array(pts1_reproj, fig=fig, ax=ax, color_index=1, pts_size=20, label=True, vis=True, debug=debug)
	print('\n\nDONE! SUCCESSFUL!!\n')

def test_triangulate_multiple_views():
	calibration_file = '../calibration.txt'
	camera_cluster = load_camera_cluster(calibration_file, warning=False)

	pts1_file = '../20180514--handsy--400015--0800.pts'
	pts_array1 = anno_parser(pts1_file)
	pts2_file = '../20180514--handsy--400053--0800.pts'
	pts_array2 = anno_parser(pts2_file)
	pts3_file = '../20180514--handsy--400025--0800.pts'
	pts_array3 = anno_parser(pts3_file)
	pts4_file = '../20180514--handsy--400027--0800.pts'
	pts_array4 = anno_parser(pts4_file)
	pts_array = np.zeros((4, 3, 21), dtype='float32')
	# print(pts_array)
	pts_array[0, :, :] = pts_array1
	# print(pts_array)
	pts_array[1, :, :] = pts_array2
	# print(pts_array)
	pts_array[2, :, :] = pts_array3
	pts_array[3, :, :] = pts_array4
	# pts_array = np.vstack((pts_array1, pts_array2, pts_array3, pts_array4))
	projection1 = camera_cluster['400015'].get_projection_matrix()
	projection2 = camera_cluster['400053'].get_projection_matrix()
	projection3 = camera_cluster['400025'].get_projection_matrix()
	projection4 = camera_cluster['400027'].get_projection_matrix()
	projection = np.zeros((4, 3, 4), dtype='float32')
	projection[0, :, :] = projection1
	projection[1, :, :] = projection2
	projection[2, :, :] = projection3
	projection[3, :, :] = projection4
	# projection = np.vstack((projection1, projection2, projection3, projection4))
	# print(pts_array.shape)

	# pts_3d, pts_reproj = triangulate_multiple_views(pts_array, projection, scaling_factor=1000)
	pts_3d, pts_reproj = triangulate_multiple_views(pts_array, projection, scaling_factor=1)
	img_path = '../20180514--handsy--400015--0800.png'
	# img_path = '../20180514--handsy--400053--0800.png'
	# img_path = '../20180514--handsy--400025--0800.png'
	# img_path = '../20180514--handsy--400027--0800.png'
	img = load_image(img_path)
	debug = False
	fig, ax = visualize_image_with_pts(img, pts_array[0, :, :], label=True, debug=debug, closefig=False, vis=True)
	fig, ax = visualize_image_with_pts(img, pts_reproj[0, :, :], label=True, debug=debug, closefig=False, vis=True)
	# visualize_pts_array(pts_reproj[0, :], fig=fig, ax=ax, color_index=1, pts_size=20, label=True, vis=True, debug=debug)

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	# test_triangulate_two_views()
	test_triangulate_multiple_views()