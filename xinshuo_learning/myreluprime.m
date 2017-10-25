% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu


% input is the matrix, which contains all output from subsequent layer
% output is the matrix with same dimension
function grad = myreluprime(output, debug_mode)
	if nargin < 2
		debug_mode = true;
	end

	if debug_mode
		assert(isvector(output) || ismatrix(output), 'the format of input to the relu function is nor right\n');
	end

	grad = zeros(size(output));
	logical_matrix = (output > 0);

	grad(logical_matrix) = 1;
end