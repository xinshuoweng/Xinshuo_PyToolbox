% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function check the format of the input boxes
% input format could be a 1x1 cell, which contains Nx4 (>= 4)
% or input boxes could be a Nx4 (>= 4) matrix or a .mat
% input format: LTBR 
% output format: LTBR 
function boxes = boxcheck_LTBR(boxes)
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
	test_LTRB = [boxes(:, 3) - boxes(:, 1), boxes(:, 4) - boxes(:, 2)];
	assert(sum(sum(test_LTRB < 0)) == 0, 'The input format for boxes should be LTBR, two points (left top and bottom right).');

end