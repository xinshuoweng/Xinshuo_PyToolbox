% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% this functions take two pairs of parallel lines in the image plane, and remove the projective distortion
% parameters
%		line_pairs:		should not be none when label mode is false, 
%						format:	4 x 3 matrix, each row is a line represented as (a, b, c), first two and last two are physically "parallel" in the input image
function [rectified_img, H] = affine_rectification(img, line_pairs, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	% sanity check
	if debug_mode
		assert(isImage(img), 'the input image is incorrect');
		assert(all(size(line_pairs) == [4, 3]), 'the pairs of parallel lines are not correct');
	end

	% get point at infinity
	p1_infinity = cross(line_pairs(1, :), line_pairs(2, :));
	p2_infinity = cross(line_pairs(3, :), line_pairs(4, :));

	% get line at infinity
	line_infinity = cross(p1_infinity, p2_infinity);

	H = [1, 0, 0; ...
		 0, 1, 0; ...
		 line_infinity];

	if debug_mode
		line_prime = inv(H') * line_infinity';				% 3 x 3 	3 x 1
		line_infinity_canonical = [0, 0, 1];
		distance = get_pts_distance(line_prime', line_infinity_canonical, debug_mode);
		assert(distance < 1e-3, sprintf('the computed homography is not precise enough, the corrected line at infinity is [%.3f, %.3f, %.3f]\n', line_prime(1), line_prime(2), line_prime(3)));
	end

	rectified_img = applyH(img, H);
end
