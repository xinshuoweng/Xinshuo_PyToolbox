% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% visualize a heatmap on top of an image
function visualize_segment(pts_array, fig, debug_mode)
    if nargin < 3
        debug_mode = true;
    end

    if debug_mode
        assert(all(size(pts_array) == [2, 2]), 'the input point array is not correct');
    end

    figure(fig);
    plot(pts_array(:, 1), pts_array(:, 2), '-rx');

    % max_value = max(max(heatmap));
    % im_to_disp = (img + mat2im(heatmap, jet(100), [0 max_value])) / 2;  % apply heatmap on the original image
    % figure; imshow(im_to_disp);
end
