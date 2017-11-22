# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import math, cv2
import numpy as np
from numpy.testing import assert_almost_equal

from xinshuo_python import *
from xinshuo_miscellaneous import print_np_shape

################################################################## 2d math ##################################################################
# TODO: check
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

def saferotation_angle(rotation_angle):
    '''
    ensure the rotation is in [-180, 180] in degree
    '''
    while rotation_angle > 180:
        rotation_angle -= 360

    while rotation_angle < -180:
        rotation_angle += 360

    return rotation_angle

def pts_rotate2D(pts_array, rotation_angle, im_height, im_width, debug=True):
    '''
    rotate the point array in 2D plane counter-clockwise

    parameters:
        pts_array:          2 x num_pts
        rotation_angle:     e.g. 90

    return
        pts_array:          2 x num_pts
    '''
    if debug:
        assert is2dptsarray(pts_array), 'the input point array does not have a good shape'

    rotation_angle = saferotation_angle(rotation_angle)             # ensure to be in [-180, 180]

    if rotation_angle > 0:
        cols2rotated = im_width
        rows2rotated = im_width
    else:
        cols2rotated = im_height
        rows2rotated = im_height
    rotation_matrix = cv2.getRotationMatrix2D((cols2rotated/2, rows2rotated/2), rotation_angle, 1)         # 2 x 3
    
    num_pts = pts_array.shape[1]
    pts_rotate = np.ones((3, num_pts), dtype='float32')             # 3 x num_pts
    pts_rotate[0:2, :] = pts_array         

    return np.dot(rotation_matrix, pts_rotate)         # 2 x num_pts

def pts_euclidean(pts1, pts2, debug=True):
    '''
    calculate the euclidean distance of two sets of points

    parameter:
        pts1, pts2: 2 x N or (2, ) numpy array, (x, y)

    return
        average euclidean distance
    '''
    if debug:
        assert is2dptsarray(pts1) and is2dptsarray(pts2), 'the input points are not correct'
        assert pts1.shape == pts2.shape, 'shape of two points is not equal'
    
    # if the shape of input points is (2, ), reshape them to (2, 1)
    if len(pts1.shape) == 1:
        pts1 = np.reshape(pts1, (2, 1))
        pts2 = np.reshape(pts2, (2, 1))

    eculidean_list = np.zeros((pts1.shape[1], ), dtype='float32')
    # calculate the distance
    num_pts = pts1.shape[1]
    ave_euclidean = 0
    for pts_index in xrange(num_pts):
        pts1_tmp = pts1[:, pts_index]
        pts2_tmp = pts2[:, pts_index]
        n = float(pts_index + 1)
        distance_tmp = math.sqrt((pts1_tmp[0] - pts2_tmp[0])**2 + (pts1_tmp[1] - pts2_tmp[1])**2)               # TODO check the start
        ave_euclidean = (n-1) / n * ave_euclidean + distance_tmp / n
        eculidean_list[pts_index] = distance_tmp

    # ave_euclidean = ave_euclidean / float(num_pts)
    return ave_euclidean, eculidean_list.tolist()

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

def calculate_truncated_mse(error_list, truncated_list, debug=True):
    '''
    calculate the mse truncated by a set of thresholds, and return the truncated MSE and the percentage of how many points' error is lower than the threshold

    parameters:
        error_list:         a list of error
        truncated_list:     a list of threshold

    return
        tmse_dict:          a dictionary where each entry is a dict and has key 'T-MSE' & 'percentage'
    '''
    if debug:
        assert islist(error_list) and all(isscalar(error_tmp) for error_tmp in error_list), 'the input error list is not correct'
        assert islist(truncated_list) and all(isscalar(thres_tmp) for thres_tmp in truncated_list), 'the input truncated list is not correct'
        assert len(truncated_list) > 0, 'there is not entry in truncated list'

    tmse_dict = dict()
    num_entry = len(error_list)
    error_array = np.asarray(error_list)
    
    for threshold in truncated_list:
        error_index = np.where(error_array[:] < threshold)[0].tolist()              # plot visible points in red color
        error_interested = error_array[error_index]
        
        entry = dict()
        entry['T-MSE'] = np.mean(error_interested)
        entry['percentage'] = len(error_index) / float(num_entry)
        tmse_dict[threshold] = entry

    return tmse_dict

################################################################## coordinates ##################################################################
def cart2pol_2d_degree(pts, debug=True):
    '''
    input a 2d point and convert to polar coordinate

    return for degree: [0, 360)
    '''
    if debug:
        assert istuple(pts) or islist(pts) or isnparray(pts), 'input point is not correct'
        assert np.array(pts).size == 2, 'input point is not 2d points'

    x = pts[0]
    y = pts[1]
    rho = np.sqrt(x**2 + y**2)
    phi = math.degrees(np.arctan2(y, x))
    while phi < 0:
        phi += 360
    while phi >= 360.:
        phi -= 360
        
    return (rho, phi)

def pol2cart_2d_degree(pts, debug=True):
    '''
    input point: (rho, phi)

    phi is in degree
    '''
    if debug:
        assert istuple(pts) or islist(pts) or isnparray(pts), 'input point is not correct'
        assert np.array(pts).size == 2, 'input point is not 2d points'

    rho = pts[0]
    phi = math.radians(pts[1])
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)