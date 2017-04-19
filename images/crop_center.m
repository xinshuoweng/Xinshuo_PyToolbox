% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

%% parameter description:
% image is the original input
% rect is an array, which defines the way of cropping the image
%       1. rect = [width, height], then crop the center of image, so the
%       output image has specific height and width
%       2. rect = [xmin, ymin, width, height], then crop image within 
%       specific rectangular region
function cropped = crop_center(img, rect, debug_mode)
    if debug_mode
        img = isImageorPath(img);
        assert(length(rect) == 2 || length(rect) == 4, 'the format of rect is wrong');
        assert(~iscell(rect), 'The input of rectangular should be a matrix.');
    end

    %% crop the specific region
    if length(rect) == 4
        xmin = rect(1);
        ymin = rect(2);
        width = rect(3);
        height = rect(4);
        assert(xmin >= 0 && ymin >= 0 && (xmin + width) <= size(img, 2) && ...
            (ymin + height) <= size(img, 1), 'the size of crop region is out of range');
        new_rect = [rect(1), rect(2), rect(3)-1, rect(4)-1];
        cropped = imcrop(img, new_rect);
    else
        %% crop the center of the image
        width = rect(1);
        height = rect(2);
        xmin = (size(img, 2) - width) / 2;
        ymin = (size(img, 1) - height) / 2;
        assert(xmin >= 0 && ymin >= 0, 'the size of crop region is out of range');
        new_rect = [xmin, ymin, width-1, height-1];
        cropped = imcrop(img, new_rect);
    end
end