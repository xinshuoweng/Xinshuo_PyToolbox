% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% input is the matrix, output the softmax of the input matrix
%
% parameters:
%		in:		num_class x batch_size
%		output:	num_class x batch_size
function output = mysoftmax(in, debug_mode)
	if nargin < 2
		debug_mode = true;
	end

	if debug_mode
		assert(ismatrix(in), 'the format of input to the softmax function is nor right\n');
	end

	total = sum(exp(in));

	if debug_mode
		assert(isvector(total), 'the summation of softmax function is not right\n');
	end

	output = exp(in)./total;
end