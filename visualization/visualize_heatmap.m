% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% visualize a heatmap on top of an image
function visualize_heatmap(img, heatmap, debug_mode)
    if nargin < 3
        debug_mode = true;
    end

    if isIntegerImage(img)
        img = im2double(img);
    end

    if debug_mode
        assert(isFloatImage_loose(img), 'input image is not a float image.');
        assert(isFloatImage_loose(heatmap), 'input heatmap is not a float image.');
    end
    max_value = max(max(heatmap));
    im_to_disp = (img + mat2im(heatmap, jet(100), [0 max_value])) / 2;  % apply heatmap on the original image
    figure; imshow(im_to_disp);
end
