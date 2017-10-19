% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu


% given a list of numbers for restricted Boltzmann machine, initialize the weight
function rbm_weight = weights_initialization_rbm(rbm, method, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(isfield(rbm, 'num_hidden'),  'the input restricted Boltzmann machine should have a number of hidden variables');
		assert(isfield(rbm, 'num_visible'), 'the input restricted Boltzmann machine should have a number of visible variables');
		assert(ischar(method) && strcmp(method, 'gaussian'), sprintf('the initialization method %s is not available', method));
	end

	% compute bias and weight for RBM
	if strcmp(method, 'gaussian')
		W = normrnd(0, 0.1, [rbm.num_hidden, rbm.num_visible]);
		bias_hidden = normrnd(0, 0.1, [rbm.num_hidden, 1]);
		bias_visible = normrnd(0, 0.1, [rbm.num_visible, 1]);
	else
		assert(false, sprintf('%s initialization method is not supported in xinshuo''s library', method));
	end

	rbm_weight = struct();
	rbm_weight.W = W;
	rbm_weight.bias_hidden = bias_hidden;
	rbm_weight.bias_visible = bias_visible;
end
