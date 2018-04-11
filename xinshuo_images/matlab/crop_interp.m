% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function is to crop the image around given a bounding box, the difference of this function compared to the crop_center is that
% this function can take a floating bbox and computer the interpolated cropped patch
% parameters:
%   img:                    a floating image
%   input_format:           'hw', 'hwc', 'chw'
%   crop_rect:              4 x 1 or 1 x 4

% TODO: handle negative rect
function cropped_patch = crop_interp(img, input_format, crop_rect, debug_mode)
    if nargin < 4
        debug_mode = true;
    end

    if debug_mode
        assert(all(size(crop_rect) == [1, 4]), 'the input rect size is not correct');
    end

    height_kernel = round(crop_rect(4) - crop_rect(2) + 1);
    width_kernel = round(crop_rect(3) - crop_rect(1) + 1);

    % convert bbox
    x = crop_rect(1) : crop_rect(3);
    y = crop_rect(2) : crop_rect(4);
    if round(length(x)) < width_kernel
        x = [x, crop_rect(3)];
    elseif round(length(x)) > width_kernel
        x = x(1:end-1);
    end
    if round(length(y)) < height_kernel
        y = [y, crop_rect(4)];
    elseif round(length(y)) > height_kernel
        y = y(1:end-1);
    end

    if strcmp(input_format, 'hw') || strcmp(input_format, 'hwc')
        im_height = size(img, 1);
        im_width = size(img, 2);
        if strcmp(input_format, 'hwc')
            im_channel = size(img, 3);
        else
            im_channel = 1;
        end
    elseif strcmp(input_format, 'chw')
        im_channel = size(img, 1);
        im_height = size(img, 2);
        im_width = size(img, 3);
    else    
        assert(false, 'error');
    end

    X_range = 1 : im_width;
    Y_range = 1 : im_height;
    [X, Y] = meshgrid(X_range, Y_range);
    [x_temp, y_temp] = meshgrid(x, y);

    % generate a cropped patch with dimension of C x H x W
    if strcmp(input_format, 'hw')
        cropped_patch = interp2(X, Y, img, x_temp, y_temp, 'linear');    
        cropped_patch = reshape(cropped_patch, 1, height_kernel, width_kernel);
    elseif strcmp(input_format, 'hwc')
        cropped_patch = zeros(height_kernel, width_kernel, im_channel);                     
        for channel_index = 1:im_channel
            cropped_patch(:, :, channel_index) = interp3(X, Y, img(channel_index, :, :), x_temp, y_temp, 'linear');    
        end
        cropped_patch = permute(cropped_patch, [3, 1, 2]);
    elseif strcmp(input_format, 'chw')
        cropped_patch = zeros(im_channel, height_kernel, width_kernel);

        for channel_index = 1:im_channel
            cropped_patch(channel_index, :, :) = interp2(X, Y, squeeze(img(channel_index, :, :)), x_temp, y_temp, 'linear');    
        end
    end

end

