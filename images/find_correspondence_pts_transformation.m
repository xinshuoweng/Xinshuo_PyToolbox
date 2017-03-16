% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function takes an image and the angle in degree as input to find the correspondent point 
% in the transformed image referred to the input point in the original image
% this function assume the rotation is used as imrotate or imtransform
% the input pts_src should be Nx2 matrix, each row is [x, y] coordinate
% when the debug flag is set false, no warning is provided
function pts_dst = find_correspondence_pts_transformation(size_in, size_out, angle_degree, pts_src, debug_flag)
	assert(~iscell(pts_src), 'The input points should be matrix instead of cell.');
	assert(ismatrix(pts_src) && length(size(pts_src)) == 2, 'The input points should be matrix with dimension Nx2.');
	assert(~iscell(size_in) && ismatrix(size_in) && length(size(size_in)) == 2, 'The input size should be matrix with dimension 1x2 or 2x1.');
	assert(~iscell(size_out) && ismatrix(size_out) && length(size(size_out)) == 2, 'The input size should be matrix with dimension 1x2 or 2x1.');
	if ~exist('debug_flag', 'var')
		disp('Warning: only rotation transformation is supported right now.');
	else
		assert(islogical(debug_flag), 'The argument for debug should be logical while finding correspondence.');
		if debug_flag
			disp('Warning: only rotation transformation is supported right now.');
		end
	end

	% normalize the input coordinate based on center of the image
	size_in = size_in / 2;
	orig_x = pts_src(:, 1) - size_in(2);
	orig_y = pts_src(:, 2) - size_in(1);

	rotation = [cosd(-angle_degree), sind(-angle_degree); -sind(-angle_degree), cosd(-angle_degree)];
	old_orig = [orig_x, orig_y];
	new_orig = old_orig * rotation;
	
	% inverse normalization
	size_out = size_out / 2;
	pts_dst(:, 1) = new_orig(:, 1) + size_out(2);
	pts_dst(:, 2) = new_orig(:, 2) + size_out(1);
end

