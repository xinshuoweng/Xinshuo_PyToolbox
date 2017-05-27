% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function convert a set of point locations within a bounding box to the point locations in the original image
% parameter:
%	pts_locations:	2 x num_pts, matrix, (x, y) locations within bounding box
%	im_size:		height x width of original image
%	bbox:			LTWH format
%
% return:
%	image_pts:		2 x num_pts, matrix, (x, y) locations in original image
function pts_locations = bbox_pts2image_pts(pts_locations, im_size, bbox, debug_mode)
	if nargin < 4
		debug_mode = true;
	end

	if debug_mode
		assert(size(pts_locations, 1) == 2, 'shape of point locations is not correct!');
	end

	[bbox, im_size] = bboxcheck_TLWH(bbox, im_size, debug_mode);

	%% convert point coordinate
	num_pts = size(pts_locations, 2);
	for pts_index = 1:num_pts
		pts_locations(1, pts_index) = pts_locations(1, pts_index) + bbox(1);
		pts_locations(2, pts_index) = pts_locations(2, pts_index) + bbox(2);
	end
end