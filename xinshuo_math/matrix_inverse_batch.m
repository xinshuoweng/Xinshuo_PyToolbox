% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% inverse a batch of matrix
% e.g. matrix has shape of 3 x 2 x 2
function weight_map = matrix_inverse_batch(matrix, debug_mode)
    if nargin < 3
        debug_mode = true;
    end

    if nargin < 2 
        sigma = min(patch_size(1), patch_size(2)) / 2.;
    end

    if debug_mode
        assert(isvector(patch_size) && numel(patch_size) == 2, 'the format of patch size is not correct');
        assert(patch_size(1) > 0 && patch_size(2) > 0, 'the patch size must > 0');
    end
    
    center = [(patch_size(1) + 1.) / 2, (patch_size(2) + 1.) / 2];
    for x = 1:patch_size(1)
        for y = 1:patch_size(2)
            weight_map(x, y) = (x - center(1)) ^ 2 + (y - center(2)) ^ 2;
        end
    end

    weight_map = exp(weight_map / -2.0 / sigma / sigma);
    weight_map(1, :) = 0;
    weight_map(end, :) = 0;
    weight_map(:, 1) = 0;
    weight_map(:, end) = 0;
end
