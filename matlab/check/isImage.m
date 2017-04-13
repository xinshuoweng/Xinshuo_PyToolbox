% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function takes a multidimensional array as input and check if it is
% an image
function valid = isImage(image_test)
	valid = (ismatrix(image_test) || (length(size(image_test)) == 3 && size(image_test, 3) == 3)) && ~iscell(image_test); 
end