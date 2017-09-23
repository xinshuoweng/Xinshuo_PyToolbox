% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% this function computes the gradient of fully connected layer with stored activation
% inputs:
%   		'X' and 'Y' the single input data sample and ground truth output vector of sizes N x 1 and C x 1 respectively
function gradients = Backward(fc_weight, X, Y, post_activation, debug_mode)
	if nargin < 6
		debug_mode = true;
	end

	if debug_mode
		assert(size(X, 2) == 1, 'the dimension of input sample is not correct');
		assert(size(Y, 2) == 1, 'the dimension of input sample is not correct');
		assert(isfield(fc_weight, 'W'), 'the weights in fully connected do not exist');
		assert(isfield(fc_weight, 'b'), 'the bias in fully connected do not exist');
	end

	W = fc_weight.W;
	b = fc_weight.b;

	num_hidden = length(post_activation);
	grad_W = cell(size(W));
	grad_b = cell(size(b));

	% number_layer = number_hidden + 1;
	output = mysoftmax(W{num_hidden + 1} * post_activation{num_hidden} + b{num_hidden + 1});

	% compute the gradient of the final output layer
	delta = output - Y;          % 9x1
	grad_W{num_hidden + 1} = delta * post_activation{num_hidden}';   			% 7x1 * 1x9
	grad_b{num_hidden + 1} = delta;                              	% 9x1

	% iteratively compute the gradient for all hidden layers
	for i = num_hidden : -1 : 1
		weight_cur = W{i + 1}';					% 7x9
		delta_cur = mysigmoidprime(post_activation{i}) .* (weight_cur * delta);    % 7x1

		if i == 1
			grad_W{i} = delta_cur * X';                                                 
		else
			grad_W{i} = delta_cur * post_activation{i-1}';                         % 5x1 * (7x1)'
		end

		grad_b{i} = delta_cur;
		delta = delta_cur;

		% grad_b{2}
		% grad_W{2}
		% pause
	end

	gradients = struct();
	gradients.W = grad_W;
	gradients.b = grad_b;
end
