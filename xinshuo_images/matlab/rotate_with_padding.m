% Author: Xinshuo Weng
% Email: xinshuo.weng@gmail.com

% this function takes an image and rotate it with additional padding (random
% noise or random background)
% padding = 'noise' or 'background'
% angle is degree
% edge_eliminate is for removing the edge in the original image while adding padding
% scale is for controling how much content will be deleted from original image after rotation.
% the bigger the scale is, more content will be removed. Generally scale
% range is [0, 10]
function im = rotate_with_padding(im, angle_degree, padding, background, edge_eliminate, edge_scale, debug_mode)
    im = check_imageorPath(im);
    if ~exist('debug_mode', 'var')
    	debug_mode = true;
    end
    if ~exist('padding', 'var')
		padding = 'noise';
	end
    if ~exist('edge_eliminate', 'var')
		edge_eliminate = false;
	end
    if ~edge_eliminate         % if no edge elimination is considered
        edge_scale = 1;
    elseif ~exist('edge_scale', 'var') 
		edge_scale = 0.1;
    end

	if debug_mode
		assert(strcmp(padding, 'noise') || strcmp(padding, 'background'), 'Only random noise or background padding are supported right now.');
		assert(isscalar(edge_scale) && ~iscell(edge_scale), 'The input argument for scale of edge should be scalar.');
		assert(islogical(edge_eliminate), 'The input argument for edge elimination should be logical.');
		assert(islogical(debug_mode), 'The format of debug mode is not correct');
	end

	% rotating the image
	im = im2double(im);
	original_size = size(im);
	im = imrotate(im, angle_degree);
    
	% padding template
    if strcmp(padding, 'noise')
		template_padding = rand(size(im));
	else
		if debug_mode
			assert(exist('background', 'var') == 1, 'The background must be provided if one want to add background as padding.');
		end
		
        template_padding = im2double(check_imageorPath(background));
        shape = size(im);
        shape = shape(1:2);
        template_padding = imresize(template_padding, shape);	% resize the background to the same scale
    end
    
	% fill the blank with padding
	threshold = 0.8;					% threshold for eliminating the edge
	dummy = ones(original_size);		% set dummy image to find the blank corner
	dummy = imrotate(dummy, angle_degree);
    dummy = imgaussfilt(dummy, edge_scale);
	im(dummy < threshold) = template_padding(dummy < threshold);

	% resize the image as output
    im = imresize(im, original_size(1:2));	
end