% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function is to judge an image to be a float image or not
function valid = isfloatimage(image_test)
	if ~isImage(image_test)
		valid = false;
	else
		% double type and within range [0, 1]
		logical_matrix = arrayfun(@(x) isscalar(x) && x<=1.0 && x>=0.0, image_test);	
		valid = all(logical_matrix(:));
	end
end