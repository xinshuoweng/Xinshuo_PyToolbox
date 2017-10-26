% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% this functions compute the norm in row-wise for a matrix
function normed_vector = matrix_norm_rowwise(matrix, debug_mode)
    if nargin < 2
        debug_mode = true;
    end

    if debug_mode
        assert(ismatrix(matrix) && length(size(matrix)) == 2, 'the input matrix is not correct');
    end
    
    normed_vector = sqrt(sum(matrix.^2, 2));
end