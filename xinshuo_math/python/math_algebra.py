# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes functions of basic algebra in math
import math, cv2, numpy as np

from private import safe_2dptsarray, safe_angle
from xinshuo_miscellaneous import is2dptsarray, islist, isscalar, isnparray, istuple

# all rotation angle is processes in degree

def pts_euclidean(input_pts1, input_pts2, warning=True, debug=True):
    '''
    calculate the euclidean distance between two sets of points

    parameters:
        input_pts1:     2 x N or (2, ) numpy array, a list of 2 elements, a listoflist of 2 elements: (x, y)
        input_pts2:     same as above

    outputs:
        ave_euclidean:      averaged euclidean distance
        eculidean_list:     a list of the euclidean distance for all data points
    '''
    pts1 = safe_2dptsarray(input_pts1, warning=warning, debug=debug)
    pts2 = safe_2dptsarray(input_pts2, warning=warning, debug=debug)
    if debug:
        assert pts1.shape == pts2.shape, 'the shape of two points is not equal'
        assert is2dptsarray(pts1) and is2dptsarray(pts2), 'the input points are not correct'

    # calculate the distance
    eculidean_list = np.zeros((pts1.shape[1], ), dtype='float32')
    num_pts = pts1.shape[1]
    ave_euclidean = 0
    for pts_index in xrange(num_pts):
        pts1_tmp = pts1[:, pts_index]
        pts2_tmp = pts2[:, pts_index]
        n = float(pts_index + 1)
        distance_tmp = math.sqrt((pts1_tmp[0] - pts2_tmp[0])**2 + (pts1_tmp[1] - pts2_tmp[1])**2)               # TODO check the start
        ave_euclidean = (n - 1) / n * ave_euclidean + distance_tmp / n
        eculidean_list[pts_index] = distance_tmp

    return ave_euclidean, eculidean_list.tolist()
    
def pts_rotate2D(pts_array, rotation_angle, im_height, im_width, warning=True, debug=True):
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

    rotation_angle = safe_angle(rotation_angle, warning=warning, debug=True)             # ensure to be in [-180, 180]

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
