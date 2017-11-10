# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os, sys, subprocess
import numpy as np

from xinshuo_io import load_txt_file, save_txt_file

################################################################################ mesh related
FNULL = open(os.devnull, 'w')

# this mesh downsample method is based on meshlab, please install meshlab first in order to use this

# Script taken from doing the needed operation
# (Filters > Remeshing, Simplification and Reconstruction >
# Quadric Edge Collapse Decimation, with parameters:
# 0.9 percentage reduction (10%), 0.3 Quality threshold (70%)
# Target number of faces is ignored with those parameters
# conserving face normals, planar simplification and
# post-simplimfication cleaning)
# And going to Filter > Show current filter script
def get_downsample_script(num_faces):

  return """<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Quadric Edge Collapse Decimation">
  <Param type="RichInt" value="500000" name="TargetFaceNum"/>
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
"""

def get_merge_script():

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

def get_translation_script(translation):
	x, y, z = translation[0], translation[1], translation[2]

	return """<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Transform: Move, Translate, Center">
  <Param description="X Axis" value="89.7634" type="RichDynamicFloat" tooltip="Absolute translation amount along the X axis" name="axisX"/>
  <Param description="Y Axis" value="-20.1220" type="RichDynamicFloat" tooltip="Absolute translation amount along the Y axis" name="axisY"/>
  <Param description="Z Axis" value="956.5000" type="RichDynamicFloat" tooltip="Absolute translation amount along the Z axis" name="axisZ"/>
  <Param description="translate center of bbox to the origin" value="false" type="RichBool" tooltip="If selected, the object is scaled to a box whose sides are at most 1 unit lenght" name="centerFlag"/>
  <Param description="Freeze Matrix" value="true" type="RichBool" tooltip="The transformation is explicitly applied and the vertex coords are actually changed" name="Freeze"/>
  <Param description="Apply to all layers" value="false" type="RichBool" tooltip="The transformation is explicitly applied to all the mesh and raster layers in the project" name="ToAll"/>
 </filter>
</FilterScript>
"""

def create_merge_filter_file(filename='filter_file_tmp.mlx'):
	with open('/tmp/' + filename, 'w') as f:
		f.write(get_merge_script())
	return '/tmp/' + filename

def create_translation_filter_file(translation, filename='filter_file_tmp.mlx'):
	with open('/tmp/' + filename, 'w') as f:
		f.write(get_translation_script(translation))
	return '/tmp/' + filename


def create_downsample_filter_file(num_faces, filename='filter_file_tmp.mlx'):
	with open('/tmp/' + filename, 'w') as f:
		f.write(get_downsample_script(num_faces))
	return '/tmp/' + filename

def reduce_faces(in_file, out_file, num_faces):
	filter_script_path = create_downsample_filter_file(num_faces)  

	# Add input mesh
	command = "meshlabserver -i " + in_file
	# Add the filter script
	command += " -s " + filter_script_path
	# Add the output filename and output flags
	command += " -o " + out_file + " -om vn fn"
	# Execute command
	# print "Going to execute: " + command
	subprocess.call(command, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
	# last_line = output.splitlines()[-1]
	# print
	# print "Done:"
	# print in_file + " > " + out_file + ": " + last_line

def merge_mesh(in_file1, in_file2, out_file):
	filter_script_path = create_merge_filter_file()  


	# Add input mesh
	command = "meshlabserver -i %s %s" % (in_file1, in_file2)
	# Add the filter script
	command += " -s " + filter_script_path
	# Add the output filename and output flags
	command += " -o " + out_file + " -om vn fn vc"
	# Execute command
	# print "Going to execute: " + command
	subprocess.call(command, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
	# last_line = output.splitlines()[-1]
	# print
	# print "Done:"
	# print in_file1 + ' + ' + in_file2 + " > " + out_file

def translate_mesh(in_file, out_file, translation):
	filter_script_path = create_translation_filter_file(translation)  

	# Add input mesh
	command = "meshlabserver -i %s" % in_file
	# Add the filter script
	command += " -s " + filter_script_path
	# Add the output filename and output flags
	command += " -o " + out_file + " -om vn fn"
	# Execute command
	print "Going to execute: " + command
	output = subprocess.check_output(command, shell=True)
	last_line = output.splitlines()[-1]
	print
	print "Done:"
	print in_file + " > " + out_file + ": " + last_line


def parse_obj_line(obj_line):
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
	else:
		return None, ' ', ' '

def translate_line(obj_line, translation, debug=True):
	pts, line_type, remain_str = parse_obj_line(obj_line)

	if line_type == 'coordinate':
		pts_translated = pts + np.array(translation)
		# print pts_translated
		# ass
		translated_line = 'v %f %f %f%s' % (pts_translated[0], pts_translated[1], pts_translated[2], remain_str)
		# print translated_line
		return translated_line
	else:
		return obj_line

def translate_obj(obj_file, out_file, translation, debug=True):
	data, num_lines = load_txt_file(obj_file, debug=debug)
	out_data = []

	for line_index in range(num_lines):
		line_tmp = data[line_index]
		new_line = translate_line(line_tmp, translation, debug=debug)
		out_data.append(new_line)

	save_txt_file(out_data, out_file, debug=debug)


def change_color_obj(obj_file, out_file, color, debug=True):
	data, num_lines = load_txt_file(obj_file, debug=debug)
	out_data = []

	for line_index in range(num_lines):
		line_tmp = data[line_index]
		pts, line_type, remain_str = parse_obj_line(line_tmp)

		if line_type == 'coordinate':
			colored_line = 'v %f %f %f %f %f %f' % (pts[0], pts[1], pts[2], color[0], color[1], color[2])
		else:
			colored_line = line_tmp

		# new_line = translate_line(line_tmp, translation, debug=debug)
		out_data.append(colored_line)

	save_txt_file(out_data, out_file, debug=debug)