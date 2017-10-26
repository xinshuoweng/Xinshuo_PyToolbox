% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu


% compute the gradient of aenn
% inputs
%		positive_decode_sample:		sample of a input data
%		negative_decode_sample:		sample from gibbs sampling
function gradients = compute_gradient_autoencoder(aenn_weight, data_temp, var_hidden, reconstucted_data, debug_mode)
	if nargin < 5
		debug_mode = true;
	end

	if debug_mode
		assert(isstruct(aenn_weight), 'the weight should be a struct \n');
		assert(isfield(aenn_weight, 'W'), 'the weights in aenn do not exist');
		assert(isfield(aenn_weight, 'bias_decode'), 'the bias of decode variable in aenn do not exist');
		assert(isfield(aenn_weight, 'bias_encode'),  'the bias of encode variable in aenn do not exist');
	end

	% compute the gradient of the final output layer
	grad_b_decode = reconstucted_data - data_temp;          													% 784 x 1, including the final sigmoid layer
	grad_W_decode = (grad_b_decode * var_hidden')';   													% 784 x 1  *  1 x 100 = 784 x 100

	grad_b_encode = aenn_weight.W * grad_b_decode .* mysigmoidprime(var_hidden);						% 100 x 1
	grad_W_encode = grad_b_encode * data_temp';															% 100 x 784
	

	% % iteratively compute the gradient for all hidden layers
	% for i = num_hidden : -1 : 1
	% 	weight_cur = W{i + 1}';																% 100 x 10

	% 	% post_activation{i}(1:10)
	% 	pre_activation = activation(post_activation{i});									% 1 x 100
	% 	% pre_activation(1:10)
	% 	% pause

	% 	delta_cur = pre_activation .* (weight_cur * delta);    								% 100 x 1

	% 	if i == 1
	% 		grad_W{i} = delta_cur * X';                                                 
	% 	else
	% 		grad_W{i} = delta_cur * post_activation{i-1}';                         			% 5x1 * (100x1)'
	% 	end

	% 	grad_b{i} = delta_cur;
	% 	delta = delta_cur;
	% end

	gradients = struct();
	gradients.W = grad_W_decode + grad_W_encode;
	gradients.bias_encode = grad_b_encode;
	gradients.bias_decode = grad_b_decode;

	% gradients = struct();
	% gradients.W = -grad_W;
	% gradients.bias_decode = -grad_bias_decode;
	% gradients.bias_encode = -grad_bias_encode;

	% grad_W(1:10, 1:10)
	% grad_bias_decode(1:10)
	% pause;
end
