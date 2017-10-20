% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% this function encode the input sample to the hidden space
function [reconstructed_data] = decode_autoencoder(aenn_weight, var_hidden, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(isstruct(aenn_weight), 'the weight should be a struct\n');
		assert(isfield(aenn_weight, 'W'), 'the weights in fully connected do not exist');
		assert(isfield(aenn_weight, 'bias_decode'), 'the bias in fully connected do not exist');
	end

	% W, b are cells of matrix to store the weight and bias
	W = aenn_weight.W;				% num_hidden x num_input
	bias_decode = aenn_weight.bias_decode;			% num_hidden x 1

	if debug_mode
		assert(size(W, 2) == length(bias_decode), 'the shape of weight and bias is not good\n');
	end

	pre_activation = W' * var_hidden + bias_decode;
	reconstructed_data = mysigmoid(pre_activation);
end
