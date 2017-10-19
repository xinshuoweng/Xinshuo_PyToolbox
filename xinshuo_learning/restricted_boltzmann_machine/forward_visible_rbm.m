% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% given the state of visible variable and current weights of RBM, inferring the probability of hidden variable
function post_activation = forward_visible_rbm(rbm_weight, var_visible, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(isstruct(rbm_weight), 'the weight should be a struct \n');
		assert(isfield(rbm_weight, 'W'), 'the weights in RBM do not exist');
		assert(isfield(rbm_weight, 'bias_hidden'), 'the bias of hidden variable in RBM do not exist');
	end

	% W, b are cells of matrix to store the weight and bias
	W = rbm_weight.W;					% num_hidden x num_visible
	bias_hidden = rbm_weight.bias_hidden;

	if debug_mode
		assert(size(W, 1) == size(bias_hidden, 1), sprintf('the input number of neurons should be equal in weight and bias: %d vs %d\n', size(W, 2), size(bias_hidden, 1)));
		assert(1 == size(bias_hidden, 2), 'the second dimension of bias in hidden variable should be equal 1\n');
		assert(all(size(var_visible) == [size(W, 2), 1]), 'The shape of input visible variable is not correct\n');
	end

	pre_activation = bias_hidden + W * var_visible;		% num_hidden x 1
	post_activation = mysigmoid(pre_activation);

	% bias_hidden(1:10)
	% W(1:10)
	% pause
end
