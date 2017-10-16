% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function checks if the input is a 2d array
function valid = is2dLine(line_test)
	if (isvector(line_test) && numel(line_test) == 3)
		valid = true;
	else
		valid = false;
end