% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% input is the vector, output the softmax of the input vector
function output = mysoftmax(in, debug_mode)
	if nargin < 2
		debug_mode = true;
	end

	if debug_mode
		assert(isvector(in), 'the format of input to the sigmoid function is nor right\n');
	end

	total = sum(exp(in));

	if debug_mode
		assert(isscalar(total), 'the summation of softmax function is not right\n');
	end

	output = exp(in)./total;
end