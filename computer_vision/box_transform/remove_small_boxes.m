% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function takes boxes in and remove very small boxes
% input format could be a 1x1 cell, which contains Nx4 
% or input boxes could be a Nx4 matrix or Nx5 matrix
% input format: LTRB 
% output format: LTRB 
function [boxes, scores] = remove_small_boxes(min_box_size, min_box_height, boxes, scores)
	boxcheck_LTBR(boxes);
    widths = boxes(:, 3) - boxes(:, 1) + 1;
    heights = boxes(:, 4) - boxes(:, 2) + 1;
    
    valid_ind = widths >= min_box_size & heights >= min_box_size & heights >= min_box_height;
    boxes = boxes(valid_ind, :);
    scores = scores(valid_ind, :);
end
