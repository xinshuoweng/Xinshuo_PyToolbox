% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu


% input is the matrix or vector, which contains all linear combination from previous layer
% output is the matrix or vector, by passing the input into the tanh function

% tanh(x) = 2 * sigmoid(2x) - 1
function output = mytanh(in, debug_mode)
	if nargin < 2
		debug_mode = true;
	end

	if debug_mode
		assert(isvector(in) || ismatrix(in), 'the format of input to the relu function is nor right\n');
	end

	in_new = 2 .* in;
	out_tmp = mysigmoid(in_new, debug_mode);
	output = 2 .* out_tmp - 1;

	% output
	% pause;
end