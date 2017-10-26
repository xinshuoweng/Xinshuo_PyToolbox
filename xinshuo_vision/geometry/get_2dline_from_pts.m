% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% this function takes two points in and output the line in 2d space
% parameters
%	pts1:	num_pts x 2
%	pts2: 	num_pts x 2
%
% output
%	line_2d:	num_pts x 3 vector [a, b, c] -> represent ax + by + c = 0

function line_2d = get_2dline_from_pts(pts1, pts2, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	epsilon = 1e-5;
	if debug_mode
		assert(size(pts1, 2) == 2 && size(pts1, 1) > 0, 'the size of input points is not correct');
		assert(all(size(pts1) == size(pts2)), 'the size of input points is not correct');
	end

	% get homogeneous coordinate
	num_pts = size(pts1, 1);
	pts1_homo = ones(num_pts, 3);					% num_pts x 3
	pts2_homo = ones(num_pts, 3);
	pts1_homo(:, 1:2) = pts1;
	pts2_homo(:, 1:2) = pts2;

	line_2d = cross(pts1_homo, pts2_homo);			% num_pts x 3

	% normalize the line
	for line_index = 1:num_pts
		line_2d(line_index, :) = line_2d(line_index, :) ./ line_2d(line_index, end);
	end

	if debug_mode
		assert(all(diag(pts1_homo * line_2d') < 1e-5), 'the point is not on the line');
		assert(all(diag(pts2_homo * line_2d') < 1e-5), 'the point is not on the line');
	end
end