% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function checks if the input is a 3d array
% num_plane x 4
function valid = is3dPlaneArray(plane_array_test)
	if (ismatrix(plane_array_test) && size(plane_array_test, 2) == 4 && size(plane_array_test, 1) > 0)
		valid = true;
	else
		valid = false;
end