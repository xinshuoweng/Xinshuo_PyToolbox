% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function return the distance of a set of points to a set of 3d planes
% parameters
%		pts_3d 			3 x num_pts, (x, y, z)
%		plane_3d		num_plane x 3, 		ax + by + cz + d = 0
%
% ouput
%		dist_matrix		num_plane x num_pts, 	the distance of every point to every plane
function dist_matrix = get_distance_from_pts_3dplane(pts_3d, plane_3d, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(is3dPtsArray(pts_3d), 'the input points is not correct');
		assert(is3dPlaneArray(plane_3d), 'the input 3d plane is not correct');
	end

	num_pts = size(pts_3d, 2);
	num_planes = size(plane_3d, 1);

	% convert point to projective space with normalization, 	
	pts_homo = ones(4, num_pts);
	pts_homo(1:3, :) = pts_3d;
	pts_3d = pts_homo;					% 4 x num_pts

	% compute the distance matrix
	dist_matrix = abs(plane_3d * pts_3d);						% num_plane x num_pts
	for plane_index = 1:num_planes
		normalization_factor = sqrt(plane_3d(plane_index, 1) ^ 2 + plane_3d(plane_index, 2) ^ 2 + plane_3d(plane_index, 3) ^ 2);
		dist_matrix(plane_index, :) = dist_matrix(plane_index, :) ./ normalization_factor;
	end
end