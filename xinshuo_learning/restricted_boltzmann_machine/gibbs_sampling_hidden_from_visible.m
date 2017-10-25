% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% given the visible variable, sample a hidden variable based on probability
function [sample_hidden] = gibbs_sampling_hidden_from_visible(rbm_weight, var_visible, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	post_activation = forward_visible_rbm(rbm_weight, var_visible, debug_mode);			% num_hidden x 1
	if debug_mode
		assert(all(size(post_activation) == [size(rbm_weight.W, 1), 1]), 'The shape of post activation is not correct\n');
	end

	N = ones(size(post_activation));
	sample_hidden = binornd(N, post_activation); 		% since all hidden variables are independent given visible variables, we sample them independently

	% post_activation(1:10)
	% sample_hidden(1:10)
	% pause;
end
