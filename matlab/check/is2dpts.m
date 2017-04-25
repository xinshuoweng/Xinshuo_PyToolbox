% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function checks if the input is a 2d array
function valid = is2dpts(pts_test)
	if (isvector(pts_test) && numel(pts_test) == 2)
		valid = true;
	else
		valid = false;
end