% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function is to crop the image around a specific center with padded value around the empty area, all images in this function are floating images
% parameters:
% rect is an array, which defines how to crop the image around center
%       1. rect with WH format, then crop around the center of image
%       2. rect with XYWH format, then crop around (X, Y) with given height and width
% Note that if the cropped region is out of boundary, we pad gray value around outside
% Note that the cropping is right aligned, which means, if the crop width or height is even, we crop one more pixel right to the center
function cropped = crop_center(img, rect, pad_value, debug_mode)
    if ~exist('debug_mode', 'var')
        debug_mode = true;
    end

    if ~exist('pad_value', 'var')
        pad_value = 0.5;
    end

    if debug_mode
        img = check_imageorPath(img, debug_mode);
        assert(size(rect, 1) == 1 && (size(rect, 2) == 2 || size(rect, 2) == 4), 'the shape of crop array is wrong');
        assert(~iscell(rect), 'The input of rectangular should be a matrix.');
        assert(all(arrayfun(@(x) isInteger(x), rect)), 'the padding array should be all integers.');
    end

    % convert image format
    if isIntegerImage(img)
        img = im2double(img);
        pad_value = pad_value / 255;
    end

    im_size = size(img);
    im_width = size(img, 2);
    im_height = size(img, 1);
    if length(rect) == 4            % crop around the given center and width and height
        center_x = rect(1);
        center_y = rect(2);
        crop_width = rect(3);
        crop_height = rect(4);
        
        if debug_mode
            center_pts = [center_x, center_y];
            assert(is2dPtsInside(center_pts, im_size, debug_mode), 'center point is not in the image');
        end

        % calculate cropped region
        xmin = center_x - ceil(crop_width/2) + 1;
        ymin = center_y - ceil(crop_height/2) + 1;
        xmax = xmin + crop_width - 1;
        ymax = ymin + crop_height - 1;
        crop_rect = [xmin, ymin, crop_width - 1, crop_height - 1];

        % crop
        cropped = imcrop(img, crop_rect);

        % if original image is not enough to cover the crop area, we pad value around outside after cropping
        if xmin < 1 || ymin < 1 || xmax > im_width || ymax > im_height
            pad_left    = max(1 - xmin, 0);
            pad_top     = max(1 - ymin, 0);
            pad_right   = max(xmax - im_width, 0);
            pad_bottom  = max(ymax - im_height, 0);
            pad_rect    = [pad_left, pad_top, pad_right, pad_bottom];

            if debug_mode
                assert(pad_value >= 0 && pad_value <= 1, sprintf('padding value: %f is not correct for a floating image.', pad_value));
            end

            % padding
            cropped = pad_around(cropped, pad_rect, pad_value, debug_mode);
        end

    else                            % crop around the center of the image
        crop_width = rect(1);
        crop_height = rect(2);
        xmin = (im_width - crop_width) / 2;
        ymin = (im_height - crop_height) / 2;
        if debug_mode
            assert(xmin >= 0 && ymin >= 0, 'the size of crop region is out of range');
        end
        crop_rect = [xmin, ymin, crop_width - 1, crop_height - 1];
        cropped = imcrop(img, crop_rect);
    end
end