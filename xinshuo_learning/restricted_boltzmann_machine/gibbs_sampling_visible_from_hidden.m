% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% given the visible variable, sample a hidden variable based on probability
function [sample_visible] = gibbs_sampling_visible_from_hidden(rbm_weight, var_hidden, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	post_activation = forward_hidden_rbm(rbm_weight, var_hidden, debug_mode);			% num_visible x 1
	if debug_mode
		assert(all(size(post_activation) == [size(rbm_weight.W, 2), 1]), 'The shape of post activation is not correct\n');
	end

	N = ones(size(post_activation));
	sample_visible = binornd(N, post_activation); 		% since all hidden variables are independent given visible variables, we sample them independently
end
