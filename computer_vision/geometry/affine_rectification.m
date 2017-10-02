% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% this functions take two pairs of parallel lines in the image plane, and remove the projective distortion
% If label_mode = 1, use ginput(8) to get 8 points (4 lines) 
% If label_mode = 0, you should load whatever information you manually entered and return [rectI, H].

% parameters
%		line_pairs:		should not be none when label mode is false, 
%						format:	4 x 3 matrix, each row is a line represented as (a, b, c), first two and last two are physically "parallel" in the input image
function [rectified_img, H] = affine_rectification(img, label_mode, debug_mode, line_pairs)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(isImage(img), 'the input image is incorrect');
		if ~label_mode
			assert()
		end
	end
end
