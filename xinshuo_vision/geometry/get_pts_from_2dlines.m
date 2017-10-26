% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% this function takes two lines in and output the point in 2d space
% parameters
%	line1:	1 x 3			ax + by + c = 0
%	line2: 	1 x 3
%
% output
%	pts:	1 x 2 

function pts = get_pts_from_2dlines(line1, line2, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	epsilon = 1e-10;
	if debug_mode
		assert(all(size(line1) == [1, 3]), 'the size of input points is not correct');
		assert(all(size(line2) == [1, 3]), 'the size of input points is not correct');
	end

	% normalization
	line1 = line1 ./ line1(1, 3);
	line2 = line2 ./ line2(1, 3);

	if debug_mode
		residual = line1 - line2;
		assert(norm(residual) > epsilon, 'the input two lines are too similar');
	end

	pts = cross(line1, line2);
	pts = pts / pts(3);

	if debug_mode
		assert(pts * line1' < epsilon, 'the point is not on the line');
		assert(pts * line2' < epsilon, 'the point is not on the line');
	end
end