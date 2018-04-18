# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes functions of basic geometry in math
import math, cv2, numpy as np
from numpy.testing import assert_almost_equal

from private import safe_pts
from xinshuo_miscellaneous import print_np_shape, is2dptsarray, is2dpts, is2dline, is3dpts, islist, isscalar, istuple

# all rotation angle is processes in degree

################################################################## 2d math ##################################################################
def get_line(pts, slope, debug=True):
    '''
    # slope is the angle in degree, this function takes a point and a
    '''
    if debug:
        print('debug mode is on during get_line function. Please turn off after debuging')
        assert is2dpts(pts), 'point is not correct'

    if slope == 90 or -90:
        slope = slope + 0.00001
    slope = math.tan(math.radians(slope))
    if debug:
        print('slope is ' + str(slope))
    dividor = slope * pts[0] - pts[1]
    if dividor == 0:
        dividor += 0.00001
    b = 1.0 / dividor
    a = -b * slope
    if debug:
        assert_almost_equal(pts[0]*a + pts[1]*b + 1, 0, err_msg='Point is not on the line')
    return np.array([a, b, 1], dtype=float)

# TODO: check
def get_slope(pts1, pts2, debug=True):
    if debug:
        print('debug mode is on during get_slope function. Please turn off after debuging')
        assert is2dpts(pts1), 'point is not correct'
        assert is2dpts(pts2), 'point is not correct'

    slope = (pts1[1] - pts2[1]) / (pts1[0] - pts2[0])
    slope = np.arctan(slope)
    slope = math.degrees(slope)
    return slope

# TODO: check
def get_intersection(line1, line2, debug=True):
    if debug:
        print('debug mode is on during get_intersection function. Please turn off after debuging')
        assert is2dline(line1), 'line is not correct'
        assert is2dline(line2), 'line is not correct'
    
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

    if debug:
        assert_almost_equal(x*line1[0] + y*line1[1] + 1, 0, err_msg='Intersection point is not on the line')
        assert_almost_equal(x*line2[0] + y*line2[1] + 1, 0, err_msg='Intersection point is not on the line')
    return np.array([x, y], dtype=float)

################################################################## 3d math ##################################################################
def generate_sphere(pts_3d, radius, debug=True):
    '''
    generate a boundary of a 3D shpere point cloud
    '''
    if debug:
        assert is3dpts(pts_3d), 'the input point is not a 3D point'

    num_pts = 100
    u = np.random.rand(num_pts, )
    v = np.random.rand(num_pts, )

    print(u.shape)
    theta = 2 * math.pi * u
    phi = math.acos(2 * v - 1)
    
    pts_shpere = np.zeros((3, num_pts), dtype='float32')
    pts_shpere[0, :] = pts_3d[0] + radius * math.sin(phi) * math.cos(theta)
    pts_sphere[1, :] = pts_3d[1] + radius * math.sin(phi) * math.sin(theta)
    pts_sphere[2, :] = pts_3d[2] + radius * math.cos(phi)

    return pts_sphere
