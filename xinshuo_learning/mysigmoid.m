% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu


% input is the matrix or vector, which contains all linear combination from previous layer
% output is the matrix or vector, by passing the input into the sigmoid function
function output = mysigmoid(in, debug_mode)
	if nargin < 2
		debug_mode = true;
	end

	if debug_mode
		assert(isvector(in) || ismatrix(in), 'the format of input to the sigmoid function is nor right\n');
	end

	output = 1./(1 + exp(-in));
end