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
function im = rotate_with_padding(im, angle_degree, padding, background, edge_eliminate, edge_scale)
    if ischar(im)
    	im = imread(im);
    end
	assert(isimage(im), 'The input matrix is not as same as format of image while rotating.');
    
    if ~exist('padding', 'var')
		padding = 'noise';
	else
		assert(strcmp(padding, 'noise') || strcmp(padding, 'background'), 'Only random noise or background padding are supported right now.');
    end
    
    if ~exist('edge_eliminate', 'var')
		edge_eliminate = false;
	else
		assert(islogical(edge_eliminate), 'The input argument for edge elimination should be logical.');
    end
    
    if ~edge_eliminate         % if no edge elimination is considered
        edge_scale = 1;
    elseif ~exist('edge_scale', 'var') 
		edge_scale = 0.1;
    else
		assert(isscalar(edge_scale) && ~iscell(edge_scale), 'The input argument for scale of edge should be scalar.');
    end

	% rotating the image
	im = im2double(im);
	original_size = size(im);
	im = imrotate(im, angle_degree);
    
	% padding template
    if strcmp(padding, 'noise')
		template_padding = rand(size(im));
	else
		assert(exist('background', 'var'), 'The background must be provided if one want to add background as padding.');
        if ischar(background)
			template_padding = imread(background);
		else
			assert(isimage(background), 'The background image given is not as same as format of image while rotating.');
			template_padding = background;
        end
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

