# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import numpy as np

import init_paths
from mycamera import MyCamera, load_camera_cluster
from camera_geometry import triangulate_two_views
from xinshuo_io import anno_parser

def test_triangulate_two_views():
	calibration_file = '../calibration.txt'
	camera_cluster = load_camera_cluster(calibration_file, warning=False)

	pts1_file = '../20180514--handsy--400015--0800.pts'
	pts_array1 = anno_parser(pts1_file)
	pts2_file = '../20180514--handsy--400053--0800.pts'
	pts_array2 = anno_parser(pts2_file)
	# print(pts_array1)
	# print(pts_array2)
	projection1 = camera_cluster['400015'].get_projection_matrix()
	projection2 = camera_cluster['400053'].get_projection_matrix()
	pts_3d, pts1_reproj, pts2_reproj = triangulate_two_views(pts_array1, pts_array2, projection1, projection2)
	print(pts_3d)

	fig, ax = visualization_image_w

	print('\n\nDONE! SUCCESSFUL!!\n')
	
if __name__ == '__main__':
	test_triangulate_two_views()
