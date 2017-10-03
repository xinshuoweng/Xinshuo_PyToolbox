% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% this function takes two points in and output the line in 2d space
% parameters
%	pts1:	1 x 2
%	pts2: 	1 x 2
%
% output
%	line:	1 x 3 vector [a, b, c] -> represent ax + by + c = 0

function line_2d = get_2dline_from_pts(pts1, pts2, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(all(size(pts1) == [1, 2]), 'the size of input points is not correct');
	end

	% get homogeneous coordinate
	pts1_homo = [pts1, 1];
	pts2_homo = [pts2, 1];

	line_2d = cross(pts1_homo, pts2_homo);
end