% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function obtain the point inside an image
function pts = checkptsinside(pts_test, im_width, im_height)
	pts(1) = min(max(pts_test(1), 0), im_width);
	pts(2) = min(max(pts_test(2), 0), im_height);
end