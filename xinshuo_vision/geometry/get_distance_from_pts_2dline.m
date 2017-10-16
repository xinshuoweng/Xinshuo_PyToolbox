% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function return the distance of a point to a 2d line
% parameters
%		pts  			(x, y), or (x, y, t) in projective space
%		line_2d			ax + by + c = 0
function distance = get_distance_from_pts_2dline(pts, line_2d, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(is2dPts(pts) || is3dPts(pts), 'the input points is not correct');
		assert(is2dLine(line_2d), 'the input 2d line is not correct');
	end

	% make sure the dimension is row vector
	if size(pts, 2) == 1
		pts = pts';						% 1 x 3
	end
	if size(line_2d, 2) == 1
		line_2d = line_2d';				% 1 x 3
	end

	% convert point in projective space to 2d space
	if is3dPts(pts)
		pts = pts / pts(3);
	else
		pts = [pts, 1];
	end

	distance = abs(pts * line_2d');
	distance = distance / sqrt(line_2d(1) ^ 2 + line_2d(2) ^ 2);
end