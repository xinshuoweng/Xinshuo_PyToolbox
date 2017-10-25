% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% this function computes the gradient of fully connected layer with stored activation
% inputs:
%   		'X' and 'Y' the single input data sample and ground truth output vector of sizes N x 1 and C x 1 respectively
function gradients = backward_fc(fc_weight, X, Y, post_activation, config, debug_mode)
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

	% define the activation function
	if isfield(config, 'activation')
		if strcmp(config.activation, 'sigmoid')
			activation = @mysigmoidprime;
		elseif strcmp(config.activation, 'relu')
			activation = @myreluprime;
		elseif strcmp(config.activation, 'tanh')
			activation = @mytanhprime;
		else
			assert(false, sprintf('the activation function is not correct in the configuration: %s', config.activation));
		end	
	else
		fprintf('No activation function is specified. Sigmoid is used here.');
		activation = @mysigmoidprime;
	end

	% number_layer = number_hidden + 1;
	pre_output = W{num_hidden + 1} * post_activation{num_hidden} + b{num_hidden + 1};		% 10 x 1
	output = mysoftmax(pre_output);															% 10 x 1

	% compute the gradient of the final output layer
	delta = output - Y;          															% 10 x 1
	grad_W{num_hidden + 1} = delta * post_activation{num_hidden}';   						% 10 x 1  *  1 x 100
	grad_b{num_hidden + 1} = delta;                              							% 10 x 1

	% iteratively compute the gradient for all hidden layers
	for i = num_hidden : -1 : 1
		weight_cur = W{i + 1}';																% 100 x 10

		% post_activation{i}(1:10)
		pre_activation = activation(post_activation{i});									% 1 x 100
		% pre_activation(1:10)
		% pause

		delta_cur = pre_activation .* (weight_cur * delta);    								% 100 x 1

		if i == 1
			grad_W{i} = delta_cur * X';                                                 
		else
			grad_W{i} = delta_cur * post_activation{i-1}';                         			% 5x1 * (100x1)'
		end

		grad_b{i} = delta_cur;
		delta = delta_cur;
	end

	gradients = struct();
	gradients.W = grad_W;
	gradients.b = grad_b;
end
