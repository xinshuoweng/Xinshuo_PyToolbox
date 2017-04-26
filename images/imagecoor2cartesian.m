% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% change the coordinate system from image coordinate system to normal cartesian system, basically reverse the y coordinate
function pts = imagecoor2cartesian(pts, debug_mode)
	if ~exist('debug_mode', 'var')
		debug_mode = true;
	end

	if debug_mode
		fprintf('debug mode is on during convert_pts function. Please turn off after debuging\n');
		assert(is2dpts(pts), 'point is not correct');
	end

	if iscell(pts)
		pts = cell2mat(pts);
	end
	pts(2) = -pts(2);
end