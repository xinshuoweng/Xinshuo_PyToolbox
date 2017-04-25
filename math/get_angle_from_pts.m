% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function takes two 2d points as input, and calculate the angle (counterclockwise) of the line
% two points don't have order
function [angle_in_degree] = get_angle_from_pts(pts1, pts2, debug_mode)
	if ~exist('debug_mode', 'var')
		debug_mode = true;
	end

	slope = get_slope(pts1, pts2, debug_mode);
    slope = atan(slope);				%(-pi/2, pi/2)
    slope = rad2deg(slope);			%(-90, 90)
end