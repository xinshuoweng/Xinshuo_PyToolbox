% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function check the format of the input boxes
% input format could be a 1x1 cell, which contains Nx4 (>= 4)
% or input boxes could be a Nx4 (>= 4) matrix or a .mat
% input format: LTWH (x, y)
% output format: LTWH (x, y)
function boxes = bboxcheck_LTWH(boxes, im_width, im_height)
	if ischar(boxes)
		try
			fprintf('loading input boxes for clipping.');
			boxes = load(boxes);
		catch
			assert(false, 'fail to load input boxes. Please check the path.');
		end
	end

	while iscell(boxes)
		assert(numel(boxes) == 1, 'For the cell input, the cell should have 1x1 shape');
		boxes = boxes{1};
	end

	assert(ismatrix(boxes) && size(boxes, 2) >= 4, 'input boxes should at least have 4 columns')
	test_width = boxes(:, 3) + boxes(:, 1);
	test_height = boxes(:, 4) + boxes(:, 2);
	assert(sum(test_width > im_width) == 0, 'The input format for boxes should be LTWH, left top point with width and height.');
	assert(sum(test_height > im_height) == 0, 'The input format for boxes should be LTWH, left top point with width and height.');
end