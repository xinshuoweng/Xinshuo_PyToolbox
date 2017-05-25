% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

%% parameter description:
% image is the original input
% rect is an array, which defines the way of cropping the image
%       1. rect with WH format, then crop the center of image, so the output image has specific height and width
%       2. rect with TLWH format, then crop image within specific rectangular region
function cropped = crop_center(img, rect, debug_mode)
    if nargin < 3
        debug_mode = true;
    end

    if debug_mode
        img = check_imageorPath(img);
        assert(length(rect) == 2 || length(rect) == 4, 'the format of rect is wrong');
        assert(~iscell(rect), 'The input of rectangular should be a matrix.');
    end

    rect = int16(rect);
    if length(rect) == 4            %% crop the specific region
        xmin = rect(1);
        ymin = rect(2);
        im_width = rect(3);
        im_height = rect(4);
        assert(xmin > 0 && ymin > 0 && (xmin + im_width) <= size(img, 2) && (ymin + im_height) <= size(img, 1), 'the size of crop region is out of range');
        new_rect = [xmin, ymin, im_width - 1, im_height - 1];
        cropped = imcrop(img, new_rect);
    else                            %% crop the center of the image
        im_width = rect(1);
        im_height = rect(2);
        xmin = (size(img, 2) - im_width) / 2;
        ymin = (size(img, 1) - im_height) / 2;
        assert(xmin >= 0 && ymin >= 0, 'the size of crop region is out of range');
        new_rect = [xmin, ymin, im_width-1, im_height-1];
        cropped = imcrop(img, new_rect);
    end
end