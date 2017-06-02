% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function is to pad given value to an image in provided region
% parameters:
%   pad_rect:   4 element array, which describes where to pad the value. The order is [left, top, right, bottom]
%   pad_value:  a scalar defines what value we should pad
function padded = pad_around(img, pad_rect, pad_value, debug_mode)
    if ~exist('debug_mode', 'var')
        debug_mode = true;
    end

    if debug_mode
        img = check_imageorPath(img);
        assert(size(pad_rect, 1) == 1 && size(pad_rect, 2) == 4, 'the shape of padding array is wrong');
        assert(~iscell(pad_rect), 'The input of rectangular should be a matrix.');
        assert(all(pad_rect >= 0), 'the padding array should be non-negative value');
        assert(all(arrayfun(@(x) isInteger(x), pad_rect)), 'the padding array should be all integers.');

        % handle edge case for padding value
        if isIntegerImage(img)
            if ~exist('pad_value', 'var')
                pad_value = 128;
            else
                assert(pad_value >= 0 && pad_value <= 255, sprintf('padding value: %d is not correct for an integer image.', pad_value));
            end  
        else
            if ~exist('pad_value', 'var')
                pad_value = 0.5;
            else
                assert(pad_value >= 0 && pad_value <= 1, sprintf('padding value: %f is not correct for a floating image.', pad_value));
            end
        end
    end

    im_size = size(img);
    im_width = size(img, 2);
    im_height = size(img, 1);
    
    % calculate the new size of image
    pad_left    = pad_rect(1);
    pad_top     = pad_rect(2);
    pad_right   = pad_rect(3);
    pad_bottom  = pad_rect(4);
    new_height  = im_height + pad_top + pad_bottom;
    new_width   = im_width + pad_left + pad_right;
    
    % pad
    channel = size(img, 3);
    padded = zeros(new_height, new_width, channel);
    padded(:) = pad_value;
    padded(pad_top+1 : new_height-pad_bottom, pad_left+1 : new_width-pad_right, :) = img;
    padded = padded / 255.;         % convert to float image (since zeros() create double matrix)
end