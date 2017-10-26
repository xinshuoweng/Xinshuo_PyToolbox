% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% inverse a batch of 2d matrix
% e.g. matrix has shape of 3 x 2 x 2
%     output has shape of 3 x 2 x 2, where the 2 x 2 matrix is the inverse of the input
function inverse_batch = matrix_inverse_batch(matrix, debug_mode)
    if nargin < 2
        debug_mode = true;
    end

    if debug_mode
        assert(length(size(matrix)) == 3, 'the input matrix should have a batch of 2D matrix');
    end

    num_batch = size(matrix, 1);
    inverse_batch = zeros(size(matrix));
    for batch_index = 1:num_batch
        inverse_batch(batch_index) = inv(matrix(batch_index));
    end

end
