% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function transfer the format of the input boxes
% input format could be a 1x1 cell, which contains Nx4 (>= 4)
% or input boxes could be a Nx4 (>= 4) matrix or a .mat
% input format: LTBR (x, y)
% output format: LTWH (x, y)

function boxes = RectLTRB2LTWH(boxes)
	boxes = boxcheck_LTRB(boxes);
	boxes = [boxes(:, 1), boxes(:, 2), boxes(:, 3) - boxes(:,1) + 1, boxes(:,4) - boxes(:,2) + 1, boxes(:, 5:end)];
end

