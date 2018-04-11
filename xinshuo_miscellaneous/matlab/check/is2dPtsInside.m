% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function is to check if the point is inside the image
function valid = is2dPtsInside(pts_test, im_size, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(is2dPts(pts_test), 'input point is not correct.');
	end
	im_size = check_imageSize(im_size, debug_mode);
	im_width = im_size(2);
	im_height = im_size(1);
	
	x = pts_test(1);
	y = pts_test(2);
	if x > 0 && x <= im_width && y > 0 && y <= im_height
		valid = true;
	else
		valid = false;
	end
end
