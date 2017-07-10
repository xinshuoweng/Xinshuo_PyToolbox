% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function takes boxes in and process it to make them inside the image
% input format could be a 1x1 cell, which contains Nx4 
% or input boxes could be a Nx4 matrix or Nx5 matrix
% input format: TLWH (x, y)
% output format: TLWH (x, y)
function boxes = clip_bboxes_TLWH(boxes, im_width, im_height, debug_mode)
	if nargin < 4
		debug_mode = true;
	end

	if debug_mode
		boxes = boxcheck_TLWH(boxes);
	end		

    % x1 >= 1 & <= im_width
    boxes(:, 1) = max(min(boxes(:, 1), im_width), 1);
    % y1 >= 1 & <= im_height
    boxes(:, 2) = max(min(boxes(:, 2), im_height), 1);
    
    % width
    boxes(:, 3) = max(min(boxes(:, 3), im_width - boxes(:, 1) + 1), 1);
    % height
    boxes(:, 4) = max(min(boxes(:, 4), im_height - boxes(:, 2) + 1), 1);
end