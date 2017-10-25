% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu


% input is the matrix or vector, which contains all linear combination from previous layer
% output is the matrix or vector, by passing the input into the Relu function
function output = myrelu(in, debug_mode)
	if nargin < 2
		debug_mode = true;
	end

	if debug_mode
		assert(isvector(in) || ismatrix(in), 'the format of input to the relu function is nor right\n');
	end

	output = zeros(size(in));
	logical_matrix = (in > 0);

	output(logical_matrix) = in(logical_matrix);
end