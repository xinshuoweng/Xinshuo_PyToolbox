% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function check the format of the input boxes
% input format could be a 1x1 cell, which contains Nx5 or a .mat

function boxes = bboxcheck_score(boxes)
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

	assert(ismatrix(boxes) && size(boxes, 2) == 5, 'input boxes with scores should have 5 columns')
	score = boxes(:, 5);
	assert(sum(score > 1) == 0, 'score of boxes should be less or equal than 1.');
	assert(sum(score < 0) == 0, 'score of boxes should be larger or equal than 0.')
end