% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function checks if the input is an array of 2d line
% num_line x 3
function valid = is2dLineArray(line_array_test)
	valid = is3dPtsArray(line_array_test');
end