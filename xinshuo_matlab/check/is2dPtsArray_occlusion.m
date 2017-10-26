% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function checks if the input is a 2d array

function valid = is2dPtsArray_occlusion(pts_array_test)
	if (ismatrix(pts_array_test) && size(pts_array_test, 1) == 3 && size(pts_array_test, 2) > 0)
		valid = true;
	else
		valid = false;
end