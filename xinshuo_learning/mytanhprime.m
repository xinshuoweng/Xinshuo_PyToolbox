% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu


% input is the matrix, which contains all output from subsequent layer
% output is the matrix with same dimension

% tanhprime(x) = 4 * sigmoidprime(2 * x)
function grad = mytanhprime(output, debug_mode)
	if nargin < 2
		debug_mode = true;
	end

	if debug_mode
		assert(isvector(output) || ismatrix(output), 'the format of input to the sigmoid function is nor right\n');
	end

	% out_new = 2 .* output;
	% grad_tmp = mysigmoidprime(out_new);
	% grad = 4 .* grad_tmp;

	% size(output)
	% pause

	grad = 1 - output .* output;
	% grad
	% pause
end