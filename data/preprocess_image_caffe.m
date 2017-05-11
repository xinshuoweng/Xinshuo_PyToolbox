% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% this file preprocess an image for caffe model
function img_out = preprocess_image_caffe(img, mean_value, debug_mode)
	if ~exist('debug_mode', 'var')
		debug_mode = true;
	end

	if debug_mode
		img = isImageorPath(img);
		assert(isintegerimage(img), 'image should be in integer format.');
		assert(length(mean_value) == 1 || length(mean_value) == 3, 'mean value should be length 1 or 3!');
		assert(all(mean_value <= 1.0 && mean_value >= 0.0), 'mean value should be in range [0, 1].');
	end

	img_out = im2double(img);
    img_out = img_out - mean_value;
    img_out = permute(img_out, [2 1 3]);	% permute to width x height x channel
    
    if size(img_out,3) == 1
        img_out(:, :, 3) = img_out(:, :, 1);	% broadcast to color image
        img_out(:, :, 2) = img_out(:, :, 1);	
    end
    
    img_out = img_out(:, :, [3 2 1]); 		% swap channel to bgr
end