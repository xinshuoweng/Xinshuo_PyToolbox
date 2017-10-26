% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function checks if the input is a 2d array
% 2 x num_pts
function valid = is2dPtsArray(pts_array_test)
	if (ismatrix(pts_array_test) && size(pts_array_test, 1) == 2 && size(pts_array_test, 2) > 0)
		valid = true;
	else
		valid = false;
end