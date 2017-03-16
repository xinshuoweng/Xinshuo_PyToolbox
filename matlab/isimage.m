% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function takes a multidimensional array as input and check if it is
% an image
function logical = isimage(image)
	logical = (ismatrix(image) || (length(size(image)) == 3 && size(image, 3) == 3)) && ~iscell(image); 
end