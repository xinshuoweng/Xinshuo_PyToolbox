# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os, sys, subprocess, inspect, numpy as np, trimesh
from xinshuo_io import load_txt_file, save_txt_file
from xinshuo_miscellaneous import get_timestring
from xinshuo_miscellaneous.python.private import safe_path

def get_merge_script():
	# this mesh downsample method is based on meshlab, please install meshlab first in order to use this
	return """<!DOCTYPE FilterScript>
	<FilterScript>
		<filter name="Flatten Visible Layers">
			<Param type="RichBool" value="true" name="MergeVisible"/>
			<Param type="RichBool" value="true" name="DeleteLayer"/>
			<Param type="RichBool" value="true" name="MergeVertices"/>
			<Param type="RichBool" value="false" name="AlsoUnreferenced"/>
		</filter>
	</FilterScript>
	"""
def create_merge_filter_file(filename='filter_file_tmp.mlx'):
	with open('/tmp/' + filename, 'w') as f: f.write(get_merge_script())
	return '/tmp/' + filename

def mesh_merge(in_file1, in_file2, out_file):
	filename = '%s_filter.mlx' % get_timestring()
	filter_script_path = create_merge_filter_file(filename)  

	command = "meshlabserver -i %s %s" % (in_file1, in_file2)
	command += " -s " + filter_script_path
	command += " -o " + out_file + " -om vn fn vc"
	subprocess.call(command, shell=True, stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT)

def mesh_list_merge(in_file_list, out_file):
	filename = '%s_filter.mlx' % get_timestring()
	filter_script_path = create_merge_filter_file(filename=filename)  

	command = "meshlabserver -i"
	for in_file in in_file_list: command = "%s %s" % (command, in_file)
	command += " -s " + filter_script_path
	command += " -o " + out_file + " -om vn fn vc"
	subprocess.call(command, shell=True, stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT)

# Script taken from doing the needed operation (Filters > Remeshing, Simplification and Reconstruction > Quadric Edge Collapse Decimation,
# with parameters: 0.9 percentage reduction (10%), 0.3 Quality threshold (70%) Target number of faces is ignored with those parameters
# conserving face normals, planar simplification and post-simplimfication cleaning) And going to Filter > Show current filter script
def get_downsample_script(num_faces):
	return """<!DOCTYPE FilterScript>
	<FilterScript>
		<filter name="Quadric Edge Collapse Decimation">
			<Param type="RichInt" value="%d" name="TargetFaceNum"/>
			<Param type="RichFloat" value="0" name="TargetPerc"/>
			<Param type="RichFloat" value="0.3" name="QualityThr"/>
			<Param type="RichBool" value="false" name="PreserveBoundary"/>
			<Param type="RichFloat" value="1" name="BoundaryWeight"/>
			<Param type="RichBool" value="false" name="PreserveNormal"/>
			<Param type="RichBool" value="false" name="PreserveTopology"/>
			<Param type="RichBool" value="true" name="OptimalPlacement"/>
			<Param type="RichBool" value="false" name="PlanarQuadric"/>
			<Param type="RichBool" value="false" name="QualityWeight"/>
			<Param type="RichBool" value="true" name="AutoClean"/>
 			<Param type="RichBool" value="false" name="Selected"/>
		</filter>
	</FilterScript>
	""" % num_faces

def create_downsample_filter_file(num_faces, filename='filter_file_tmp.mlx'):
	with open('/tmp/' + filename, 'w') as f: f.write(get_downsample_script(num_faces))
	return '/tmp/' + filename

def mesh_downsample(in_file, out_file, num_faces):
	filter_script_path = create_downsample_filter_file(num_faces, filename='reduce_tmp_%d.mlx' % num_faces)  
	command = "meshlabserver -i " + in_file
	command += " -s " + filter_script_path
	command += " -o " + out_file + " -om vn fn vc"
	subprocess.call(command, shell=True, stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT)

def __parse_obj_line(obj_line):
	'''
	check if the input line is a coordinate line
	'''
	if obj_line[0:2] == 'v ':
		str_list = obj_line.split(' ')
		x_str, y_str, z_str = str_list[1], str_list[2], str_list[3]
		assert str_list[0] == 'v', 'the line is not coordinate'
		pts = np.zeros((3, ), dtype='float32')
		pts[0] = float(x_str)
		pts[1] = float(y_str)
		pts[2] = float(z_str)

		if len(str_list) > 4:
			remain_str = ''
			for str_index in range(4, len(str_list)):
				remain_str = remain_str + ' %s' % str_list[str_index]

		return pts, 'coordinate', remain_str
	else: return None, ' ', ' '

def __translate_line(obj_line, translation, debug=True):
	pts, line_type, remain_str = __parse_obj_line(obj_line)
	if line_type == 'coordinate':
		pts_translated = pts + np.array(translation)
		translated_line = 'v %f %f %f%s' % (pts_translated[0], pts_translated[1], pts_translated[2], remain_str)
		return translated_line
	else: return obj_line

def mesh_translate_obj(obj_file, out_file, translation, debug=True):
	data, num_lines = load_txt_file(obj_file, debug=debug)
	out_data = []
	for line_index in range(num_lines):
		line_tmp = data[line_index]
		new_line = __translate_line(line_tmp, translation, debug=debug)
		out_data.append(new_line)

	save_txt_file(out_data, out_file, debug=debug)

def mesh_change_color_obj(obj_file, out_file, color, alpha=0.1, debug=True):
	data, num_lines = load_txt_file(obj_file, debug=debug)
	out_data = []
	for line_index in range(num_lines):
		line_tmp = data[line_index]
		pts, line_type, remain_str = __parse_obj_line(line_tmp)

		if line_type == 'coordinate':
			colored_line = 'v %f %f %f %f %f %f %f' % (pts[0], pts[1], pts[2], color[0], color[1], color[2], alpha)
		else: colored_line = line_tmp
		out_data.append(colored_line)

	save_txt_file(out_data, out_file, debug=debug)

def obj2ply_pcl(obj_file, ply_file, debug=True):
	filepath = inspect.getfile(inspect.currentframe())
	filepath = safe_path(filepath)
	parent_list = filepath.split('/')			# works on linux
	parent_list = parent_list[0:-2]
	parent_dir = '/'.join(parent_list)
	lib_dir = os.path.join(parent_dir, 'cplusplus/obj2ply')
	command = '%s/obj2ply %s %s' % (lib_dir, obj_file, ply_file)
	os.system(command)

def obj2ply_trimesh(obj_file, ply_file, debug=True):
	'''
	sometims the pcl function does not work
	'''
	mesh = trimesh.load(obj_file)
	trimesh.io.export.export_mesh(mesh, ply_file)