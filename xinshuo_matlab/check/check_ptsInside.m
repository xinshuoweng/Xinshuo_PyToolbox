% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function obtain the point inside an image
function pts = check_ptsInside(pts_test, im_size, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(is2dPts(pts_test), 'input point is not correct.');
	end
	im_size = check_imageSize(im_size, debug_mode);
	im_width = im_size(2);
	im_height = im_size(1);
	
	pts(1) = min(max(pts_test(1), 0), im_width);
	pts(2) = min(max(pts_test(2), 0), im_height);
end