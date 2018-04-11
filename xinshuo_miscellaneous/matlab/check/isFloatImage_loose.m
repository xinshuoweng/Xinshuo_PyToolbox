% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function is to judge an image to be a float image or not in a loose way
% which means the value inside could not be strictly within [0.0, 1.0], could exceed within a certain tolerance
function valid = isFloatImage_loose(image_test, tolerance)
	if nargin < 2
		tolerance = 0.2;
	end

	if ~isImage(image_test)
		valid = false;
	else
		% double type and within range [0, 1]
		logical_matrix = arrayfun(@(x) isscalar(x) && x<=1 + tolerance && x>=0.0 - tolerance, image_test);	
		valid = all(logical_matrix(:));
	end
end