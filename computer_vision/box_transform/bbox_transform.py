# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import __init__paths__
import numpy as np
import math
import time
import math_function as mf
from numpy.testing import assert_almost_equal
from math import radians as rad

from check import islist

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

# boxes are from RPN, deltas are from boxes regression parameter

def bbox_transform_inv(boxes, deltas):
    # if cfg.DEBUG:
    #     print(boxes.shape)
    #     print(deltas.shape)
    #     print(boxes[0])
    #     print(deltas[0])
        # time.sleep(10000)

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

    # if cfg.DEBUG:
    #     print(deltas[0, 0:4])
    #     print(deltas[0, 1::4])
    #     print(deltas[0, 2::4])
    #     print(deltas[0, 3::4])
        # time.sleep(1000)

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def bbox_rotation_inv(bbox_in, angle_in_degree, image_shape):
    '''
    bbox_in is two coordinate
    angle is clockwise
    '''
    assert islist(bbox_in) and len(bbox_in) == 4, 'box is not correct'

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


# transfer the general bbox (top left and bottom right points) to represent rotated bbox with loose version including top left and bottom right points
def bbox_general2rotated_loose(bbox_in, angle_in_degree, image_shape):
    bbox = bbox_rotation_inv(bbox_in, angle_in_degree, image_shape) # get top left and bottom right coordinate of the rotated bbox in the image coordinate
    return bbox_rotatedtight2rotatedloose(bbox, angle_in_degree)


def bbox_rotatedtight2rotatedloose(bbox_in, angle_in_degree):
    '''
    transfer the rotated bbox with tight version to loose version, both contains only two points (top left and bottom right)
    only a single box is feeded into
    '''
    assert islist(bbox_in) and len(bbox_in) == 4, 'box is not correct'

    pts_tl = np.array([bbox_in[0], bbox_in[1]])
    pts_br = np.array([bbox_in[2], bbox_in[3]])
    line1 = mf.get_line(mf.convert_pts(pts_tl), angle_in_degree + 90.00)
    line2 = mf.get_line(mf.convert_pts(pts_br), angle_in_degree)
    pts_bl = mf.convert_pts_back2image(mf.get_intersection(line1, line2))
    pts_tr = mf.convert_pts_back2image(mf.get_intersection(mf.get_line(mf.convert_pts(pts_tl), angle_in_degree), mf.get_line(mf.convert_pts(pts_br), angle_in_degree + 90.00)))
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
    # print(test)

    # time.sleep(1000)
    return test


def apply_rotation_loose(all_boxes, angle_in_degree, image_shape):
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
            all_boxes[row, cls_ind * 4 : (cls_ind + 1) * 4] = bbox_general2rotated_loose(box_tmp, angle_in_degree, image_shape)

    return all_boxes