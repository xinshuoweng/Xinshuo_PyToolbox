% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% this function takes two points in and compute the distance
% parameters
%	pts1:	1 x 2 or 1 x 3
%	pts2: 	1 x 2 or 1 x 3
%
% output
%	distance:	scalar

function distance = get_pts_distance(pts1, pts2, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(size(pts1, 2) == size(pts2, 2), 'the size of input points are not equal');
		assert(size(pts1, 1) == 1, 'the size of input points are not correct');
		assert(size(pts2, 1) == 1, 'the size of input points are not correct');
	end

	distance = norm(pts1 - pts2);
end