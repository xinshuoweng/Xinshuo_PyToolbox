% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function checks if the input is a 3d array
% 3 x num_pts
function valid = is3dPtsArray(pts_array_test)
	if (ismatrix(pts_array_test) && size(pts_array_test, 1) == 3 && size(pts_array_test, 2) > 0)
		valid = true;
	else
		valid = false;
end