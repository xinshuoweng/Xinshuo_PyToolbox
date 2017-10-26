% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu


% compute the gradient of RBM
% inputs
%		positive_visible_sample:		sample of a input data
%		negative_visible_sample:		sample from gibbs sampling
function gradients = compute_gradient_rbm(rbm_weight, positive_visible_sample, negative_visible_sample, debug_mode)
	if nargin < 4
		debug_mode = true;
	end

	if debug_mode
		assert(isstruct(rbm_weight), 'the weight should be a struct \n');
		assert(isfield(rbm_weight, 'W'), 'the weights in RBM do not exist');
		assert(isfield(rbm_weight, 'bias_visible'), 'the bias of visible variable in RBM do not exist');
		assert(isfield(rbm_weight, 'bias_hidden'),  'the bias of hidden variable in RBM do not exist');
	end

	hidden_prab_positive = forward_visible_rbm(rbm_weight, positive_visible_sample, debug_mode);		% num_hidden x 1
	hidden_prab_negative = forward_visible_rbm(rbm_weight, negative_visible_sample, debug_mode);		% num_hidden x 1

	grad_W = hidden_prab_positive * positive_visible_sample' - hidden_prab_negative * negative_visible_sample';			% num_hidden x num_visible
	grad_bias_hidden = hidden_prab_positive - hidden_prab_negative;
	grad_bias_visible = positive_visible_sample - negative_visible_sample;

	% hidden_prab_positive(1:10)
	% positive_visible_sample(1:10)
	% pause

	gradients = struct();
	gradients.W = -grad_W;
	gradients.bias_visible = -grad_bias_visible;
	gradients.bias_hidden = -grad_bias_hidden;

	% grad_W(1:10, 1:10)
	% grad_bias_visible(1:10)
	% pause;
end
