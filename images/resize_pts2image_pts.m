% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function converts a set of point locations in a resized image to the point locations in the original image
% parameter:
%	pts_locations:	2 x num_pts, matrix, (x, y) locations in a resized image
%	im_size:		height x width of original image
%	resize_shape:	HW format
%
% return:
%	image_pts:		2 x num_pts, matrix, (x, y) locations in original image
function pts_locations = resize_pts2image_pts(pts_locations, im_size, resize_shape, debug_mode)
	if nargin < 4
		debug_mode = true;
	end

	if debug_mode
		assert(size(pts_locations, 1) == 2, 'shape of point locations is not correct!');
	end
	im_size = check_imageSize(im_size, debug_mode);
	resize_shape = check_imageSize(resize_shape, debug_mode);

	%% convert point coordinate
	num_pts = size(pts_locations, 2);
	for pts_index = 1:num_pts
		pts_locations(1, pts_index) = pts_locations(1, pts_index) * im_size(2) / resize_shape(2); % for x
		pts_locations(2, pts_index) = pts_locations(2, pts_index) * im_size(1) / resize_shape(1); % for y
	end
end