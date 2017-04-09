% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function takes boxes in and process it to make them inside the image
% input format could be a 1x1 cell, which contains Nx4 
% or input boxes could be a Nx4 matrix or Nx5 matrix
% input format: LTRB (x, y)
% output format: LTRB (x, y)
function boxes = clip_boxes(boxes, im_width, im_height)
	boxcheck_LTBR(boxes);
    % x1 >= 1 & <= im_width
    boxes(:, 1:4:end) = max(min(boxes(:, 1:4:end), im_width), 1);
    % y1 >= 1 & <= im_height
    boxes(:, 2:4:end) = max(min(boxes(:, 2:4:end), im_height), 1);
    % x2 >= 1 & <= im_width
    boxes(:, 3:4:end) = max(min(boxes(:, 3:4:end), im_width), 1);
    % y2 >= 1 & <= im_height
    boxes(:, 4:4:end) = max(min(boxes(:, 4:4:end), im_height), 1);
end