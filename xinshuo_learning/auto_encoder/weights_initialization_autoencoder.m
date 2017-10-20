% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu


% given a list of numbers for Auto-Encoder, initialize the weight
function aenn_weight = weights_initialization_autoencoder(aenn, method, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(isfield(aenn, 'num_hidden'),  'the input Auto-Encoder should have a number of hidden variables');
		assert(isfield(aenn, 'num_input'), 'the input Auto-Encoder should have a number of input variables');
		assert(ischar(method) && strcmp(method, 'gaussian'), sprintf('the initialization method %s is not available', method));
	end

	% compute bias and weight for RBM
	if strcmp(method, 'gaussian')
		W = normrnd(0, 0.1, [aenn.num_hidden, aenn.num_input]);
		bias_encode = normrnd(0, 0.1, [aenn.num_hidden, 1]);
		bias_decode = normrnd(0, 0.1, [aenn.num_input, 1]);
	else
		assert(false, sprintf('%s initialization method is not supported in xinshuo''s library', method));
	end

	aenn_weight = struct();
	aenn_weight.W = W;
	aenn_weight.bias_encode = bias_encode;
	aenn_weight.bias_decode = bias_decode;
end
