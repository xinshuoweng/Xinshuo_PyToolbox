# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes functions of basic geometry in math
import math, cv2, numpy as np
from numpy.testing import assert_almost_equal

from private import safe_ptsarray, safe_npdata
from xinshuo_miscellaneous import print_np_shape, is2dptsarray, is2dpts, is2dline, is3dpts, islist, isscalar, istuple

# instruction
# 2D line representation:           ax + by + c = 0,            vector representation: (a, b, c)
# 2D pts representation:            (x, y)
# 3D line representation:           ax + by + cz + d = 0,       vector representation: (a, b, c, d)
# 3D pts representation:            (x, y, z)

################################################################## 2d math ##################################################################
def get_2Dline_from_pts_slope(input_pts, slope, warning=True, debug=True):
    '''
    # slope is the angle in degree, this function takes a point and a
    '''
    np_pts = safe_ptsarray(input_pts, warning=warning, debug=debug)
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

def get_line_from_pts(pts1, pts2, warning=True, debug=True):
    pass


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

def nparray_hwc2chw(input_nparray, warning=True, debug=True):
    '''
    this function transpose the channels of an numpy array from HWC to CHW

    parameters:
        input_nparray:  a numpy HWC array

    outputs:
        np_array:       a numpy CHW array
    '''
    np_array = safe_npdata(input_nparray, warning=warning, debug=debug)
    if debug: assert np_array.ndim == 3, 'the input numpy array does not have a good dimension: {}'.format(np_image.shape)

    return np.transpose(np_array, (2, 0, 1)) 

def nparray_chw2hwc(input_nparray, warning=True, debug=True):
    '''
    this function transpose the channels of an numpy array  from CHW to HWC

    parameters:
        input_nparray:  a numpy CHW array

    outputs:
        np_array:       a numpy HWC array
    '''

    if debug: isnparray(input_nparray), 'the input array is not a numpy'
    np_array = input_nparray.copy()
    if debug: assert np_array.ndim == 3, 'the input numpy array does not have a good dimension: {}'.format(np_image.shape)

    return np.transpose(np_array, (1, 2, 0)) 

def nparray_resize(input_nparray, resize_factor=None, target_size=None, interp='bicubic', warning=True, debug=True):
    '''
    resize the numpy array given a resize factor (e.g., 0.25), or given a target size (height, width)
    e.g., the numpy array has 600 x 800:
        1. given a resize factor of 0.25 -> results in an image with 150 x 200
        2. given a target size of (300, 400) -> results in an image with 300 x 400
    note that:
        resize_factor and target_size cannot exist at the same time

    parameters:
        input_nparray:      a numpy array
        resize_factor:      a scalar
        target_size:        a list of tuple or numpy array with 2 elements, representing height and width
        interp:             interpolation methods: bicubic or bilinear

    outputs:
        resized_nparray:    a numpy array
    ''' 
    np_array = safe_npdata(input_nparray, warning=warning, debug=debug)
    if debug:
        assert interp in ['bicubic', 'bilinear'], 'the interpolation method is not correct'
        assert (resize_factor is not None and target_size is None) or (resize_factor is None and target_size is not None), 'resize_factor and target_size cannot co-exist'

    if target_size is not None:
        if debug: assert isimsize(target_size), 'the input target size is not correct'
        target_width, target_height = int(round(target_size[1])), int(round(target_size[0]))
    elif resize_factor is not None:
        if debug: assert isscalar(resize_factor), 'the resize factor is not a scalar'
        height, width = np_array.shape[:2]
        target_width, target_height = int(round(resize_factor * width)), int(round(resize_factor * height))
    else: assert False, 'the target_size and resize_factor do not exist'

    if interp == 'bicubic':
        resized_nparray = cv2.resize(np_array, (target_width, target_height), interpolation = cv2.INTER_CUBIC)
    elif interp == 'bilinear':
        resized_nparray = cv2.resize(np_array, (target_width, target_height), interpolation = cv2.INTER_LINEAR)
    else: assert False, 'interpolation is wrong'

    return resized_nparray