% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function check the dimension of image size, usually convert (H, W, D) to (H, W)
function im_size = check_imageSize(imsize_check, debug_mode) 
	if nargin < 2
		debug_mode = true;
	end
	
	if debug_mode
		assert(isvector(imsize_check) && (length(imsize_check) == 2 || length(imsize_check) == 3), 'input image size is not correct.');
	end

	if length(imsize_check) == 3
		im_size = imsize_check(1:2);
	else
		im_size = imsize_check;
	end
end