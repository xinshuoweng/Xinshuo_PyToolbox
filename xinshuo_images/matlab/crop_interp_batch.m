% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function is to crop the image around given a bounding box, the difference of this function compared to the crop_center is that
% this function can take a floating bbox and computer the interpolated cropped patch
% parameters:
%   img:                    a floating image
%   input_format:           'hw', 'hwc', 'chw'
%   crop_rect:              num_pts x 4

% TODO: handle negative rect
function cropped_patch_batch = crop_interp_batch(img, input_format, crop_rect, debug_mode)
    if nargin < 4
        debug_mode = true;
    end

    if debug_mode
        assert(size(crop_rect, 2) == 4 && size(crop_rect, 1) > 0, 'the input rect size is not correct');
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

    height_kernel = round(crop_rect(1, 4) - crop_rect(1, 2) + 1);
    width_kernel = round(crop_rect(1, 3) - crop_rect(1, 1) + 1);

    num_rect = size(crop_rect, 1);
    cropped_patch_batch = zeros(num_rect, im_channel, height_kernel, width_kernel);
    for rect_index = 1:num_rect
        height_kernel_tmp = round(crop_rect(rect_index, 4) - crop_rect(rect_index, 2) + 1);
        width_kernel_tmp = round(crop_rect(rect_index, 3) - crop_rect(rect_index, 1) + 1);
        assert(height_kernel_tmp == height_kernel, 'the height should be equal in all samples');
        assert(width_kernel_tmp == width_kernel, 'the height should be equal in all samples');

        % convert bbox
        x = crop_rect(rect_index, 1) : crop_rect(rect_index, 3);
        y = crop_rect(rect_index, 2) : crop_rect(rect_index, 4);
        if round(length(x)) < width_kernel
            x = [x, crop_rect(rect_index, 3)];
        elseif round(length(x)) > width_kernel
            x = x(1:end-1);
        end
        if round(length(y)) < height_kernel
            y = [y, crop_rect(rect_index, 4)];
        elseif round(length(y)) > height_kernel
            y = y(1:end-1);
        end
        [x_temp, y_temp] = meshgrid(x, y);

        % generate a cropped patch with dimension of num_pts x C x H x W
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

        cropped_patch_batch(rect_index, :, :, :) = cropped_patch;
    end
end

