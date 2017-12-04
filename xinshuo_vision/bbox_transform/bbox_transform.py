# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file contains all functions related to operation of bounding box
# import __init__paths__
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal
from math import radians as rad


from xinshuo_math import get_line, get_intersection
from xinshuo_images import imagecoor2cartesian, cartesian2imagecoor
from xinshuo_python import isnparray, is2dptsarray, is2dptsarray_occlusion, is2dpts


def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def bbox_transform_inv(boxes, deltas, debug=True):
    '''
    boxes are from RPN, deltas are from boxes regression parameter
    '''
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths          # center of the boxes
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    if debug:
        print(deltas[0, 0:4])
        print(deltas[0, 1::4])
        print(deltas[0, 2::4])
        print(deltas[0, 3::4])

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w     # x1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h     # y1
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w     # x2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h     # y2
    return pred_boxes

def clip_boxes(boxes, im_shape):
    '''
    Clip boxes to image boundaries.
    '''
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)     # x1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)     # y1 >= 0
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)     # x2 < im_shape[1]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)     # y2 < im_shape[0]
    return boxes

def bbox_rotation_inv(bbox_in, angle_in_degree, image_shape, debug=True):
    '''
    bbox_in is two coordinate
    angle is clockwise
    '''
    if debug:
        assert isnparray(bbox_in) and bbox_in.size == 4, 'box is not correct'

    im_width = image_shape[1]
    im_height = image_shape[0]
    coor_in_tl = np.array([(bbox_in[0] - im_width/2)/im_width*2, (bbox_in[1] - im_height/2)/im_height*2, 1]) # normalization
    coor_in_br = np.array([(bbox_in[2] - im_width/2)/im_width*2, (bbox_in[3] - im_height/2)/im_height*2, 1]) # normalization
    # print(coor_in_tl)
    # print(coor_in_br)
    affine = np.array([[math.cos(rad(angle_in_degree)), math.sin(rad(angle_in_degree)), 0], [-math.sin(rad(angle_in_degree)), math.cos(rad(angle_in_degree)), 0]])
    coor_out_tl = np.dot(coor_in_tl, affine.transpose())
    coor_out_br = np.dot(coor_in_br, affine.transpose())
    # print(coor_out_tl)
    # print(coor_out_br)
    bbox_recover = [coor_out_tl[0] * im_width/2 + im_width/2, coor_out_tl[1] * im_height/2 + im_height/2, coor_out_br[0] * im_width/2 + im_width/2, coor_out_br[1] * im_height/2 + im_height/2]
    bbox_recover = np.array(bbox_recover, dtype = float)

    return bbox_recover

def bbox_general2rotated_loose(bbox_in, angle_in_degree, image_shape, debug=True):
    '''
    transfer the general bbox (top left and bottom right points) to represent rotated bbox with loose version including top left and bottom right points
    '''
    bbox = bbox_rotation_inv(bbox_in, angle_in_degree, image_shape, debug=debug) # get top left and bottom right coordinate of the rotated bbox in the image coordinate
    return bbox_rotatedtight2rotatedloose(bbox, angle_in_degree, debug=debug)

def bbox_rotatedtight2rotatedloose(bbox_in, angle_in_degree, debug=True):
    '''
    transfer the rotated bbox with tight version to loose version, both contains only two points (top left and bottom right)
    only a single box is feeded into
    '''
    if debug:
        assert isnparray(bbox_in) and bbox_in.size == 4, 'box is not correct'

    pts_tl = np.array([bbox_in[0], bbox_in[1]])
    pts_br = np.array([bbox_in[2], bbox_in[3]])
    line1 = get_line(imagecoor2cartesian(pts_tl), angle_in_degree + 90.00)
    line2 = get_line(imagecoor2cartesian(pts_br), angle_in_degree)
    pts_bl = cartesian2imagecoor(get_intersection(line1, line2))
    pts_tr = cartesian2imagecoor(get_intersection(get_line(imagecoor2cartesian(pts_tl), angle_in_degree), get_line(imagecoor2cartesian(pts_br), angle_in_degree + 90.00)))
    # assert_almost_equal(np.dot(pts_bl - pts_br, pts_bl - pts_tl), 0, err_msg='The intersection points are wrong')
    # assert_almost_equal(np.dot(pts_tr - pts_br, pts_tr - pts_tl), 0, err_msg='The intersection points are wrong')

    pts_tl_final = np.zeros((2), dtype=np.float32)
    pts_br_final = np.zeros((2), dtype=np.float32)
    pts_tl_final[0] = min({pts_tl[0], pts_br[0], pts_bl[0], pts_tr[0]})
    pts_tl_final[1] = min({pts_tl[1], pts_br[1], pts_bl[1], pts_tr[1]})
    pts_br_final[0] = max({pts_tl[0], pts_br[0], pts_bl[0], pts_tr[0]})
    pts_br_final[1] = max({pts_tl[1], pts_br[1], pts_bl[1], pts_tr[1]})

    # print(pts_tl_final)
    # print(pts_br_final)
    test = np.hstack((pts_tl_final, pts_br_final))
    return test

def apply_rotation_loose(all_boxes, angle_in_degree, image_shape, debug=True):
    '''
    this function takes Nx84 bounding box into account and transfer all of them
    to rotated representation with loose version

    all_boxes support for multiple classes
    '''
    assert all_boxes.shape[1] % 4 == 0, 'The shape of boxes is not multiple of 4\
    while applying rotation with loose version'

    num_classes = all_boxes.shape[1] / 4
    num_proposal = all_boxes.shape[0]

    for row in xrange(num_proposal):
        for cls_ind in xrange(num_classes):
            # print()
            box_tmp = all_boxes[row, cls_ind * 4 : (cls_ind + 1) * 4]
            all_boxes[row, cls_ind * 4 : (cls_ind + 1) * 4] = bbox_general2rotated_loose(box_tmp, angle_in_degree, image_shape, debug=debug)

    return all_boxes

def apply_rotation_tight(bbox_in, angle_in_degree, im_shape, debug=True):
    '''
    return 4 points clockwise
    '''
    if debug:
        assert isnparray(bbox_in) and bbox_in.size == 4, 'box is not correct'

    bbox_in = np.reshape(bbox_in, (4, ))
    bbox_tight = bbox_rotation_inv(bbox_in, angle_in_degree, im_shape, debug=debug) # get top left and bottom right coordinate of the rotated bbox in the image coordinate
    # print('bbox after inverse the rotation')
    # print(bbox_tight)
    pts_total = np.zeros((4, 2), dtype=np.int)
    pts_tl = np.array([bbox_tight[0], bbox_tight[1]])
    pts_br = np.array([bbox_tight[2], bbox_tight[3]])
    line1 = get_line(imagecoor2cartesian(pts_tl, debug=debug), angle_in_degree + 90.00, debug=debug)
    line2 = get_line(imagecoor2cartesian(pts_br, debug=debug), angle_in_degree, debug=debug)
    pts_bl = cartesian2imagecoor(get_intersection(line1, line2, debug=debug), debug=debug)
    pts_tr = cartesian2imagecoor(get_intersection(get_line(imagecoor2cartesian(pts_tl, debug=debug), angle_in_degree, debug=debug), get_line(imagecoor2cartesian(pts_br, debug=debug), angle_in_degree + 90.00, debug=debug), debug=debug), debug=debug)

    # print np.reshape(pts_tl, (1, 2)).shape
    # print pts_total[0, :].shape

    pts_total[0, :] = np.reshape(pts_tl, (1, 2))
    pts_total[1, :] = np.reshape(pts_tr, (1, 2))
    pts_total[2, :] = np.reshape(pts_br, (1, 2))
    pts_total[3, :] = np.reshape(pts_bl, (1, 2))
    return pts_total

def pts2bbox(pts, debug=True, vis=False):
    '''
    convert a set of 2d points to a bounding box

    parameter:  
        pts:    2 x N numpy array, N should >= 2

    return:
        bbox:   1 x 4 numpy array, TLBR format
    '''
    if debug:
        assert is2dptsarray(pts) or is2dptsarray_occlusion(pts), 'the input points should have shape: 2 or 3 x num_pts vs %d x %s' % (pts.shape[0], pts.shape[1])
        assert pts.shape[1] >= 2, 'number of points should be larger or equal than 2'

    bbox = np.zeros((1, 4), dtype='float32')
    bbox[0, 0] = np.min(pts[0, :])          # x coordinate of left top point
    bbox[0, 1] = np.min(pts[1, :])          # y coordinate of left top point
    bbox[0, 2] = np.max(pts[0, :])          # x coordinate of bottom right point
    bbox[0, 3] = np.max(pts[1, :])          # y coordinate of bottom right point
    
    if vis:
        fig = plt.figure()
        pts = imagecoor2cartesian(pts)
        plt.scatter(pts[0, :], pts[1, :], color='r')
        plt.scatter(bbox[0, 0], -bbox[0, 1], color='b')         # -1 is to convert the coordinate from image to cartesian
        plt.scatter(bbox[0, 2], -bbox[0, 3], color='b')
        plt.show()
        plt.close(fig)
    return bbox

def bbox2center(bbox, debug=True, vis=False):
    '''
    convert a bounding box to a point, which is the center of this bounding box

    parameter:
        bbox:   N x 4 numpy array, TLBR format

    return:
        center: 2 x N numpy array, x and y correspond to first and second row respectively
    '''
    if debug:
        assert bboxcheck_TLBR(bbox), 'the input bounding box should be TLBR format'

    num_bbox = bbox.shape[0]        
    center = np.zeros((num_bbox, 2), dtype='float32')
    center[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2.
    center[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2.

    if vis:
        fig = plt.figure()
        plt.scatter(bbox[0, 0], -bbox[0, 1], color='b')         # -1 is to convert the coordinate from image to cartesian
        plt.scatter(bbox[0, 2], -bbox[0, 3], color='b')
        center_show = imagecoor2cartesian(center)
        plt.scatter(center_show[0], center_show[1], color='r')        
        plt.show()
        plt.close(fig)
    return np.transpose(center)

def bboxcheck(bbox, debug=True):
    '''
    check the input to be a bounding box 

    parameter:
        bbox:   N x 4 numpy array, N >= 0
    
    return:
        True or False
    '''    
    return isnparray(bbox) and bbox.shape[1] == 4 and bbox.shape[0] >= 0

def bboxcheck_TLBR(bbox, debug=True):
    '''
    check the input bounding box to be TLBR format

    parameter:
        bbox:   N x 4 numpy array, TLBR format
    
    return:
        True or False
    '''
    if not bboxcheck(bbox):
        return False

    return (bbox[:, 3] >= bbox[:, 1]).all() and (bbox[:, 2] >= bbox[:, 0]).all()      # coordinate of bottom right point should be larger or equal than top left point

def bbox_TLBR2TLWH(bbox, debug=True):
    '''
    transform the input bounding box with TLBR format to TLWH format

    parameter:
        bbox: N X 4 numpy array, TLBR format

    return 
        bbox: N X 4 numpy array, TLWH format
    '''
    if debug:
        assert bboxcheck_TLBR(bbox), 'the input bounding box should be TLBR format'

    bbox_TLWH = np.zeros_like(bbox)
    bbox_TLWH[:, 0] = bbox[:, 0]
    bbox_TLWH[:, 1] = bbox[:, 1]
    bbox_TLWH[:, 2] = bbox[:, 2] - bbox[:, 0]
    bbox_TLWH[:, 3] = bbox[:, 3] - bbox[:, 1]
    return bbox_TLWH


def bbox_enlarge(bbox, ratio=0.2, width_ratio=None, height_ratio=None, min_length=128, min_width=None, min_height=None, debug=True):
    '''
    enlarge the bbox around the edge

    parameters:
        bbox:   N X 4 numpy array, TLBR format
        ratio:  how much to enlarge, for example, the ratio=0.2, then the width and height will be increased by 0.2 times of original width and height
    '''

    if debug:
        assert bboxcheck_TLBR(bbox), 'the input bounding box should be TLBR format'

    if width_ratio is not None and height_ratio is not None:
        width = (bbox[:, 2] - bbox[:, 0]) * width_ratio
        height = (bbox[:, 3] - bbox[:, 1]) * height_ratio
    else:
        width = (bbox[:, 2] - bbox[:, 0]) * ratio
        height = (bbox[:, 3] - bbox[:, 1]) * ratio

    cur_width = bbox[:, 2] - bbox[:, 0]
    cur_height = bbox[:, 3] - bbox[:, 1]
    if min_height is not None and min_width is not None:
        width = max(width, min_width - cur_width)
        height = max(height, min_height - cur_height)
    else:
        width = max(width, min_length - cur_width)
        height = max(height, min_length - cur_height)

    bbox[:, 0] -= width / 2.0
    bbox[:, 2] += width / 2.0
    bbox[:, 3] += height / 2.0
    bbox[:, 1] -= height / 2.0

    return bbox

def pts_conversion_bbox(pts_array, bbox, debug=True):
    '''
    convert pts in the original image to pts in the cropped image

    parameters:
        bbox:       1 X 4 numpy array, TLBR or TLWH format
        pts_array:  2(3) x N numpy array, N should >= 1
    '''

    if debug:
        assert is2dptsarray(pts_array) or is2dptsarray_occlusion(pts_array), 'the input points should have shape: 2 or 3 x num_pts vs %d x %s' % (pts_array.shape[0], pts_array.shape[1])
        assert bboxcheck(bbox), 'the input bounding box is not correct'

    pts_array[0, :] = pts_array[0, :] - bbox[0, 0]
    pts_array[1, :] = pts_array[1, :] - bbox[0, 1]

    return pts_array

def pts_conversion_back_bbox(pts_array, bbox, debug=True):
    '''
    convert pts in the cropped image to the pts in the original image 

    parameters:
        bbox:       1 X 4 numpy array, TLBR or TLWH format
        pts_array:  2(3) x N numpy array, N should >= 1
    '''

    if debug:
        assert is2dptsarray(pts_array) or is2dptsarray_occlusion(pts_array), 'the input points should have shape: 2 or 3 x num_pts vs %d x %s' % (pts_array.shape[0], pts_array.shape[1])
        assert bboxcheck(bbox), 'the input bounding box is not correct'

    pts_array[0, :] = pts_array[0, :] + bbox[0, 0]
    pts_array[1, :] = pts_array[1, :] + bbox[0, 1]

    return pts_array

def get_centered_bbox(pts_array, width, height, debug=True):
    '''
    given a set of points, return a set of bbox which are centered at the points
    
    parameters:
        pts_array:      2 x num_pts

    return:
        bbox:           N x 4 (TLBR format)

    '''
    
    if debug:
        assert is2dpts(pts_array) or is2dptsarray(pts_array) or is2dptsarray_occlusion(pts_array), 'the input points should have shape: 2 or 3 x num_pts vs %d x %s' % (pts_array.shape[0], pts_array.shape[1])

    if is2dpts(pts_array):
        xmin = pts_array[0] - np.ceil(width/2.0) + 1    
        ymin = pts_array[1] - np.ceil(height/2.0) + 1
    else: 
        xmin = pts_array[0, :] - np.ceil(width/2.0) + 1    
        ymin = pts_array[1, :] - np.ceil(height/2.0) + 1

    xmax = xmin + width - 1;
    ymax = ymin + height - 1;
    
    bbox = np.vstack((xmin, ymin, xmax, ymax))
    return np.transpose(bbox)