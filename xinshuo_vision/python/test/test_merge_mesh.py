# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os

from xinshuo_vision import merge_mesh
from xinshuo_io import load_list_from_folder, mkdir_if_missing
from xinshuo_python import *

sf = 31730
ef = 32270

pts_obj_path = '/mnt/dome/wentao/mark_short/obj'
mesh_root = '/media/xinshuo/disk2/datasets/Mugsy_v2/rom/mark_short/visualization/rendered_mesh_highres/mesh_with_pts'
save_dir = '/media/xinshuo/disk2/datasets/Mugsy_v2/rom/mark_short/wentao/merged_highres'
mkdir_if_missing(save_dir)

for frame_index in range(sf, ef+1):
	print('processing frame %d' % frame_index)
	pts_obj = os.path.join(pts_obj_path, 'pointcloud%05d.obj' % frame_index)
	mesh_path = os.path.join(mesh_root, '%05d.ply' % frame_index)
	save_path = os.path.join(save_dir, '%05d.ply' % frame_index)
	if is_path_exists(save_path):
		continue
	merge_mesh(pts_obj, mesh_path, save_path)

# translated_mesh = '/home/xinshuo/tmp.ply'
# translation = [89.7634, -20.1220, 956.5000]

# translate_mesh(outfile, translated_mesh, translation)
