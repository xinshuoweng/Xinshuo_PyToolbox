# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes functions of basic geometry in math
import math, cv2, numpy as np
from numpy.testing import assert_almost_equal

from private import safe_pts
from xinshuo_miscellaneous import print_np_shape, is2dptsarray, is2dpts, is2dline, is3dpts, islist, isscalar, istuple

# all rotation angle is processes in degree

################################################################## 2d math ##################################################################
def pts_euclidean(input_pts1, input_pts2, debug=True):
    '''
    calculate the euclidean distance between two sets of points

    parameters:
        input_pts1:     2 x N or (2, ) numpy array, a list of 2 elements, a listoflist of 2 elements: (x, y)
        input_pts2:     same as above

    outputs:
        ave_euclidean:      averaged euclidean distance
        eculidean_list:     a list of the euclidean distance for all data points
    '''
    pts1 = safe_pts(input_pts1, debug=debug)
    pts2 = safe_pts(input_pts2, debug=debug)
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

    rotation_angle = safe_angle(rotation_angle, debug=True)             # ensure to be in [-180, 180]

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


# centroid
# def find_tensor_peak_batch(heatmap, radius, downsample, threshold = 0.000001):
#   assert heatmap.dim() == 3, 'The dimension of the heatmap is wrong : {}'.format(heatmap.size())
#   assert radius > 0 and isinstance(radius, numbers.Number), 'The radius is not ok : {}'.format(radius)
#   num_pts, H, W = heatmap.size(0), heatmap.size(1), heatmap.size(2)
#   # find the approximate location:
#   score, index = torch.max(heatmap.view(num_pts, -1), 1)
#   index_w = (index % W).float()
#   index_h = (index / W).float()
  
#   def normalize(x, L):
#     return -1. + 2. * x.data / (L-1)
#   boxes = [index_w - radius, index_h - radius, index_w + radius, index_h + radius]
#   boxes[0] = normalize(boxes[0], W)
#   boxes[1] = normalize(boxes[1], H)
#   boxes[2] = normalize(boxes[2], W)
#   boxes[3] = normalize(boxes[3], H)
#   affine_parameter = torch.zeros((num_pts, 2, 3))
#   affine_parameter[:,0,0] = (boxes[2]-boxes[0])/2
#   affine_parameter[:,0,2] = (boxes[2]+boxes[0])/2
#   affine_parameter[:,1,1] = (boxes[3]-boxes[1])/2
#   affine_parameter[:,1,2] = (boxes[3]+boxes[1])/2
  
#   # extract the sub-region heatmap
#   theta = MU.np2variable(affine_parameter,heatmap.is_cuda,False)
#   grid_size = torch.Size([num_pts, 1, radius*2+1, radius*2+1])
#   grid = F.affine_grid(theta, grid_size)
#   sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid).squeeze(1)
#   sub_feature = F.threshold(sub_feature, threshold, np.finfo(float).eps)

#   # slow for speed improvement
#   X = MU.np2variable(torch.arange(-radius, radius+1),heatmap.is_cuda,False).view(1, 1, radius*2+1)
#   Y = MU.np2variable(torch.arange(-radius, radius+1),heatmap.is_cuda,False).view(1, radius*2+1, 1)
  
#   sum_region = torch.sum(sub_feature.view(num_pts,-1),1)
#   x = torch.sum((sub_feature*X).view(num_pts,-1),1) / sum_region + index_w
#   y = torch.sum((sub_feature*Y).view(num_pts,-1),1) / sum_region + index_h
     
#   x = x * downsample + downsample / 2.0 - 0.5
#   y = y * downsample + downsample / 2.0 - 0.5
#   return torch.stack([x, y],1), score

# TODO
def find_peaks(heatmap, thre):
    #filter = fspecial('gaussian', [3 3], 2)
    #map_smooth = conv2(map, filter, 'same')
    
    # variable initialization    

    map_smooth = np.array(heatmap)
    map_smooth[map_smooth < thre] = 0.0


    map_aug = np.zeros([map_smooth.shape[0]+2, map_smooth.shape[1]+2])
    map_aug1 = np.zeros([map_smooth.shape[0]+2, map_smooth.shape[1]+2])
    map_aug2 = np.zeros([map_smooth.shape[0]+2, map_smooth.shape[1]+2])
    map_aug3 = np.zeros([map_smooth.shape[0]+2, map_smooth.shape[1]+2])
    map_aug4 = np.zeros([map_smooth.shape[0]+2, map_smooth.shape[1]+2])
    
    # shift in different directions to find peak, only works for convex blob
    map_aug[1:-1, 1:-1] = map_smooth
    map_aug1[1:-1, 0:-2] = map_smooth
    map_aug2[1:-1, 2:] = map_smooth
    map_aug3[0:-2, 1:-1] = map_smooth
    map_aug4[2:, 2:] = map_smooth

    peakMap = np.multiply(np.multiply(np.multiply((map_aug > map_aug1),(map_aug > map_aug2)),(map_aug > map_aug3)),(map_aug > map_aug4))
    peakMap = peakMap[1:-1, 1:-1]

    idx_tuple = np.nonzero(peakMap)     # find 1
    Y = idx_tuple[0]
    X = idx_tuple[1]

    score = np.zeros([len(Y),1])
    for i in range(len(Y)):
        score[i] = heatmap[Y[i], X[i]]

    return X, Y, score


# def find_tensor_peak(heatmap, radius, downsample):
#   assert heatmap.dim() == 2, 'The dimension of the heatmap is wrong : {}'.format(heatmap.dim())
#   assert radius > 0 and isinstance(radius, numbers.Number), 'The radius is not ok : {}'.format(radius)
#   H, W = heatmap.size(0), heatmap.size(1)
#   # find the approximate location:
#   score, index = torch.max(heatmap.view(-1), 0)
#   index = int(MU.variable2np(index))
#   index_h, index_w = np.unravel_index(index, (H,W))

#   sw, sh = int(index_w - radius),     int(index_h - radius)
#   ew, eh = int(index_w + radius + 1), int(index_h + radius + 1)
#   sw, sh = max(0, sw), max(0, sh)
#   ew, eh = min(W, ew), min(H, eh)
  
#   subregion = heatmap[sh:eh, sw:ew]
#   threshold = 0.000001
#   eps = np.finfo(float).eps

#   with torch.cuda.device_of(subregion):
#     X = MU.np2variable(torch.arange(sw, ew).unsqueeze(0))
#     Y = MU.np2variable(torch.arange(sh, eh).unsqueeze(1))

#   indicator = (subregion > threshold).type( type(subregion.data) )
#   eps = (subregion <= threshold).type( type(subregion.data) ) * eps
#   subregion = subregion * indicator + eps
  
#   x = torch.sum( subregion * X ) / torch.sum( subregion )
#   y = torch.sum( subregion * Y ) / torch.sum( subregion )
     
#   ## calculate the score
#   np_x, np_y = MU.variable2np(x), MU.variable2np(y)
#   x2, y2 = min(W-1, int(np.ceil(np_x))), min(H-1, int(np.ceil(np_y)))
#   x1, y1 = max(0, x2-1), max(0, y2-1)
#   ## Bilinear interpolation
#   if x1 == x2: 
#     R1, R2 = heatmap[y1, x1], heatmap[y1, x2]
#   else:
#     R1 = (x2-x)/(x2-x1)*heatmap[y1, x1] + (x-x1)/(x2-x1)*heatmap[y1, x2]
#     R2 = (x2-x)/(x2-x1)*heatmap[y2, x1] + (x-x1)/(x2-x1)*heatmap[y2, x2]
#   if y1 == y2:
#     score = R1
#   else:
#     score = (y2-y)/(y2-y1)*R1 + (y-y1)/(y2-y1)*R2
     
#   x = x * downsample + downsample / 2.0 - 0.5
#   y = y * downsample + downsample / 2.0 - 0.5
#   return torch.cat([x, y]), score
