% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function return the dominant planes from 3d points
% parameters
%		pts_3d  			3 x num_pts, (x, y, z) in euclidean space
%
% ouputs
%		planes 				num_planes x 4, each row has format of ax + by + cz + d = 0
%		plane_index			num_pts x 1,    the final assignment of plane for every point
function [planes, plane_index] = get_dominant_3dplanes(pts_3d, num_planes, debug_mode, max_iter, err_threshold)
	if nargin < 3
		debug_mode = true;
	end

	if nargin < 4
		max_iter = 1000;
	end

	if nargin < 5 
		err_threshold = 0.02;
	end

	if debug_mode
		assert(is3dPtsArray(pts_3d), 'the input points is not correct');
		assert(isInteger(num_planes), 'the input number of planes is not correct');
	end

	num_pts = size(pts_3d, 2);
	plane_index = zeros(num_pts, 1);
	for plane_index = 1:num_planes
		

	end
end