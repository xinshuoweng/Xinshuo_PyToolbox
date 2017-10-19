% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% this function computes the average reconstructed error
function [reconstructed_error_avg] = eval_reconstruction_error(rbm_weights, data, config, debug_mode)
	if nargin < 5
		debug_mode = true;
	end

	if debug_mode
		assert(isfield(rbm_weights, 'W'), 'the weights in fully connected do not exist');
		assert(isfield(rbm_weights, 'bias_hidden'), 'the bias in hidden variables do not exist');
		assert(isfield(rbm_weights, 'bias_visible'), 'the bias in visible variables do not exist');
	end

	% inference
	num_data = size(data, 1);
	reconstructed_data = zeros(size(data));			% num_data x num_class
	for data_index = 1 : num_data
	    data_tmp = data(data_index, :)';

		% get negative visible sample
		var_visible = data_tmp;
		for iter_index = 1:config.sampling_step
			hidden_sample = gibbs_sampling_hidden_from_visible(rbm_weights, var_visible, debug_mode);
			var_visible = gibbs_sampling_visible_from_hidden(rbm_weights, hidden_sample, debug_mode);
		end

		reconstructed_data_prab_tmp = forward_hidden_rbm(rbm_weights, hidden_sample, debug_mode);			% num_visible x 1
		reconstructed_data(data_index, :) = reconstructed_data_prab_tmp';
	end

	% compute reconstructed root of square error
	% difference_square = (reconstructed_data - data) .^ 2;
	% reconstructed_error = sqrt(sum(difference_square, 2));
	% reconstructed_error_avg = mean(reconstructed_error);

	% compute the cross-entropy reconstructed error
	cross_entropy = data .* log(reconstructed_data + 1e-9) + (1 - data) .* log(1 - reconstructed_data + 1e-9);
	cross_entropy = mean(cross_entropy, 2);
	reconstructed_error_avg = -mean(cross_entropy);
end
