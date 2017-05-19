% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function check the format of the input boxes
% input format could be a 1x1 cell, which contains Nx5 or a .mat
% and also the score should be sorted

function boxes = bboxcheck_sortedscore(boxes)
	boxes = bboxcheck_score(boxes);
	score = boxes(:, 5);
	assert(issorted(fliplr(score')), 'score is not sorted in descend order.');
end