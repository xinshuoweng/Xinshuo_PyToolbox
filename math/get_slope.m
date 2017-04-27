% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function takes two 2d points as input and return the slope of that line
function slope = get_slope(pts1, pts2, debug_mode)
	if debug_mode
		fprintf('debug mode is on during get_slope function. Please turn off after debuging\n');
		assert(is2dpts(pts1), 'point is not correct');
		assert(is2dpts(pts2), 'point is not correct');
	end

	if iscell(pts1)
		pts1 = cell2mat(pts1);
	end
	if iscell(pts2)
		pts2 = cell2mat(pts2);
	end

	slope = (double(pts1(2)) - double(pts2(2))) / (double(pts1(1)) - double(pts2(1)));
end