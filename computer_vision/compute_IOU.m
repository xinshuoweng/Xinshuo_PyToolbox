function [IOU] = compute_IOU(pre_bbox, gt_bbox)

% description:
% pre_bbox is a maitrx has dimension N x 4, N is number of examples, the
% bbox has a format [top_left_x, top_left_y, width, height]
% gt_bbox is a maitrx has dimension N x 4, N is number of examples, the
% bbox has a format [top_left_x, top_left_y, width, height]

assert(isequal(size(pre_bbox), size(gt_bbox)), 'Error! The size of prediction is not same as ground truth');
IOU = diag(bboxOverlapRatio(pre_bbox, gt_bbox, 'Union'));

end