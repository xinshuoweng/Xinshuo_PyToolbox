% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function return the distance of a point to a 2d line
% parameters
%		pts  			2 or 3 x num_pts, (x, y), or (x, y, t) in projective space
%		line_2d			num_line x 3, 		ax + by + c = 0
%
% ouput
%		dist_matrix		num_line x num_pts, 	the distance of every point to every line
function dist_matrix = get_distance_from_pts_2dline(pts, line_2d, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(is2dPtsArray(pts) || is3dPtsArray(pts), 'the input points is not correct');
		assert(is2dLineArray(line_2d), 'the input 2d line is not correct');
	end

	num_pts = size(pts, 2);
	num_lines = size(line_2d, 1);

	% convert point to projective space with normalization, 	3 x num_pts
	if is3dPtsArray(pts)
		for pts_index = 1:num_pts
			pts(:, pts_index) = pts(:, pts_index) ./ pts(end, pts_index);
		end
	else
		pts_homo = ones(3, num_pts);
		pts_homo(1:2, :) = pts;
		pts = pts_homo;
	end	

	% compute the distance matrix
	dist_matrix = abs(line_2d * pts);						% num_line x num_pts
	for line_index = 1:num_lines
		normalization_factor = sqrt(line_2d(line_index, 1) ^ 2 + line_2d(line_index, 2) ^ 2);
		dist_matrix(line_index, :) = dist_matrix(line_index, :) ./ normalization_factor;
	end
end