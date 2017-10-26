% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% this function computes the average reconstructed error
function [reconstructed_error_avg, reconstructed_data] = eval_reconstruction_error_autoencoder(aenn_weights, data, debug_mode)
	if nargin < 4
		debug_mode = true;
	end

	if debug_mode
		assert(isfield(aenn_weights, 'W'), 'the weights in fully connected do not exist');
		assert(isfield(aenn_weights, 'bias_encode'), 'the bias in hidden variables do not exist');
		assert(isfield(aenn_weights, 'bias_decode'), 'the bias in visible variables do not exist');
	end

	% inference
	num_data = size(data, 1);
	reconstructed_data = zeros(size(data));			% num_data x num_class
	for data_index = 1 : num_data
	    data_tmp = data(data_index, :)';

	    var_hidden = encode_autoencoder(aenn_weights, data_tmp, debug_mode);
	    reconstructed_data_tmp = decode_autoencoder(aenn_weights, var_hidden, debug_mode);

		reconstructed_data(data_index, :) = reconstructed_data_tmp';
	end

	% compute the cross-entropy reconstructed error
	cross_entropy = data .* log(reconstructed_data + 1e-9) + (1 - data) .* log(1 - reconstructed_data + 1e-9);
	cross_entropy = mean(cross_entropy, 2);
	reconstructed_error_avg = -mean(cross_entropy);
end
