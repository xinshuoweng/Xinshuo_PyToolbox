# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file contains a set of functions for 2d geometry manipulatio
# Note: all lines in this file are represented as ax + by + c = 0

import numpy as np
import math

from xinshuo_python import *

# TODO: CHECK
def get_line(pts, slope):
	'''
	this function takes a point and a slope to construct a line
	slope is represented by the angle in degree, 
	'''
	if slope == 90 or slope == -90:
		slope = slope + 0.00001
	slope = math.tan(math.radians(slope))
	# print('slope is ' + str(slope))
	dividor = slope * pts[0] - pts[1]
	if dividor == 0:
		dividor += 0.00001
	b = 1.0 / dividor
	a = -b * slope
	assert math.fabs(pts[0]*a + pts[1]*b + 1) < 0.0000001, 'Point is not on the line'
	return np.array([a, b, 1], dtype=float)

# TODO: CHECK
def get_slope(pts1, pts2):
	slope = (pts1[1] - pts2[1]) / (pts1[0] - pts2[0])
	# print(slope)
	slope = np.arctan(slope)
	# print(slope)
	slope = math.degrees(slope)
	# print(slope)
	return slope

# TODO: CHECK
def get_intersection(line1, line2):
	a1 = line1[0]
	b1 = line1[1]
	a2 = line2[0]
	b2 = line2[1]
	dividor = a2 * b1 - a1 * b2
	if dividor == 0:
		dividor += 0.00001
	y = (a1 - a2) / dividor
	if a1 == 0:
		a1 += 0.00001
	x = (-1.0 - b1 * y) / a1
	assert math.fabs(x*line1[0] + y*line1[1] + 1) < 0.0000001, 'Intersection point is not on the line'
	assert math.fabs(x*line2[0] + y*line2[1] + 1) < 0.0000001, 'Intersection point is not on the line'
	return np.array([x, y], dtype=float)


def distance_pts(pts1, pts2, debug=True):
	'''
	this function computes the distance between 2 points
	'''

	if debug:
		assert is2dpts(pts1) and is2dpts(pts2), 'the input point is not correct'

	if not isnparray(pts1):
		pts1 = np.array(pts1)
	if not isnparray(pts2):
		pts2 = np.array(pts2)

	pts1 = np.reshape(pts1, (2, ))
	pts2 = np.reshape(pts2, (2, ))

	distance = np.sqrt(np.sum((pts1 - pts2) * (pts1 - pts2)))
	return float(distance)
