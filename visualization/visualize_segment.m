% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% visualize a heatmap on top of an image
function visualize_segment(pts_array, fig, color_index, debug_mode)
    if nargin < 3
        debug_mode = true;
    end

    color_set = ['r', 'g', 'b', 'y', 'm', 'c', 'w', 'k'];

    if debug_mode
        assert(all(size(pts_array) == [2, 2]), 'the input point array is not correct');
    end

    % figure(fig);
    plot(pts_array(:, 1), pts_array(:, 2), 'Marker', 'x', 'Color', color_set(color_index));
end
