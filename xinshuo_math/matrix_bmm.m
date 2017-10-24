% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% the batch matrix multiplication 
% eg. matrix1 has shape of 3 x 2 x 4
%     matrix2 has shape of 3 x 4 x 5
%     output has shape of 3 x 2 x 5
function batch_matrix = matrix_bmm(matrix1, matrix2, debug_mode)
    if nargin < 3
        debug_mode = true;
    end

    if debug_mode
        assert(length(size(matrix1)) == 3, 'the input matrix does not have a good shape.');
        assert(length(size(matrix2)) == 2 || length(size(matrix2)) == 3, 'the input matrix does not have a good shape.');
        assert(size(matrix1, 1) == size(matrix2, 1), 'the batch number is not correct');
        assert(size(matrix1, 3) == size(matrix2, 2), 'the matrix is not good for multiplication');
    end

    % handle singleton dimension as the last dimension
    if length(size(matrix2)) == 2
        handle_singleton = true;
        matrix2 = repmat(matrix2, 1, 1, 2);
    else
        handle_singleton = false;
    end

    num_batch = size(matrix1, 1);
    batch_matrix = zeros(num_batch, size(matrix1, 2), size(matrix2, 3));
    for batch_index = 1:num_batch
        batch_matrix(batch_index, :, :) = squeeze(matrix1(batch_index, :, :)) * squeeze(matrix2(batch_index, :, :));
    end

    if handle_singleton
        batch_matrix = batch_matrix(:, :, 1);
    end
end