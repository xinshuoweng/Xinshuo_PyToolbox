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
	epsilon = 1e-12;
% 
	% max(in(:))
	% min(in(:))

	input_exp = exp(in);
	% mean(input_exp(:))
	% max(input_exp(:))

	% min(input_exp(:))
	

	total = sum(input_exp);
	% size()
	total = repmat(total, size(in, 1), 1);
	% min(total(:))
	% max(total(:))

	if debug_mode
		assert(ismatrix(total), 'the summation of softmax function is not right\n');
	end

	output = input_exp ./ total;

	index = find(isnan(output(:)));
	% length(index)
	if ~isempty(index)
		index_selected = index(1:min(10, length(index)))
		input_exp(index_selected)
		total(index_selected)
	end

	assert(~any(isinf(input_exp(:))), 'input is nan');
	% assert(~any(isinf(total(:))), 'total is nan');
	% assert(~any(isinf(output(:))), 'output is nan');
	% assert(~any(isnan(input_exp(:))), 'input is nan');
	% assert(~any(isnan(total(:))), 'total is nan');
	% assert(~any(isnan(output(:))), 'output is nan');
end