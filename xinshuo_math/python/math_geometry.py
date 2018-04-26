# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes functions of basic geometry in math
import math, cv2, numpy as np
from numpy.testing import assert_almost_equal

from private import safe_2dptsarray, safe_npdata
from xinshuo_miscellaneous import print_np_shape, is2dptsarray, is2dpts, is2dhomopts, is2dhomoline, is3dpts, islist, isscalar, istuple

# homogenous representation
# 2D line representation:           ax + by + c = 0,            vector representation: (a, b, c)
# 2D pts representation:            (x, y),                     vector representation: (x, y, z)
# 3D plane representation:          ax + by + cz + d = 0,       vector representation: (a, b, c, d)
# 3D pts representation:            (x, y, z),                  vector representation: (x, y, z, t)

################################################################## 2d planar geomemtry ##################################################################
def get_2dline_from_pts(input_pts1, input_pts2, warning=True, debug=True):
    '''
    get the homogenous line representation from two 2d homogenous points

    parameters:
        input_pts1:         a homogenous 2D point, can be a list or tuple or numpy array: (x, y, z)
        input_pts2:         a homogenous 2D point, can be a list or tuple or numpy array: (x, y, z)

    outputs:
        np_line:            a homogenous 2D line,  can be a list or tuple or numpy array: 3 x 1, (a, b, c)
    '''
    np_pts1 = safe_2dptsarray(input_pts1, homogenous=True, warning=warning, debug=debug)
    np_pts2 = safe_2dptsarray(input_pts2, homogenous=True, warning=warning, debug=debug)
    if debug: assert is2dhomopts(np_pts1) and is2dhomopts(np_pts2), 'point is not correct'
    np_line = np.cross(np_pts1.transpose(), np_pts2.transpose()).transpose()

    return np_line

def get_2dpts_from_lines(input_line1, input_line2, warning=True, debug=True):
    '''
    get the homogenous point representation from two 2d homogenous lines

    parameters:
        input_line1:         a homogenous 2D line, can be a list or tuple or numpy array: (a, b, c)
        input_line2:         a homogenous 2D line, can be a list or tuple or numpy array: (a, b, c)

    outputs:
        np_pts:              a homogenous 2D point,  can be a list or tuple or numpy array: 3 x 1, (a, b, c)
    '''    
    # np_line1 = safe_2dptsarray(input_line1, homogenous=True, warning=warning, debug=debug)
    # np_line2 = safe_2dptsarray(input_line2, homogenous=True, warning=warning, debug=debug)
    # if debug: assert is2dhomoline(np_line1) and is2dhomoline(np_line2), 'lines are not correc'
    # np_pts = np.cross(np_line1.transpose(), np_line2.transpose()).transpose()
    np_pts = get_2dline_from_pts(input_line1, input_line2, warning=warning, debug=debug)

    return np_pts

def get_2Dline_from_pts_slope(input_pts, slope, warning=True, debug=True):
    '''
    # slope is the angle in degree, this function takes a point and a
    '''
    np_pts = safe_2dptsarray(input_pts, warning=warning, debug=debug)
    if debug:
        assert is2dpts(np_pts), 'point is not correct'
        assert isscalar(slope), 'the slope is not correct'

    if slope == 90 or -90:
        slope = slope + 0.00001
    slope = math.tan(math.radians(slope))

    dividor = slope * pts[0] - pts[1]
    if dividor == 0:
        dividor += 0.00001
    b = 1.0 / dividor
    a = -b * slope

    if debug: assert_almost_equal(pts[0] * a + pts[1] * b + 1, 0, err_msg='Point is not on the line')
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

################################################################## 3d geometry ##################################################################
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

################################################################## homogenous vs euclidean ##################################################################
def homogenous2euclidean(homo_input, warning=True, debug=True):
    pass