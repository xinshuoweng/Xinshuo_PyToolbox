% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% this function computes the gradient of fully connected layer with stored activation
% inputs:
%   		'X' and 'Y' the single input data sample and ground truth output vector of sizes N x 1 and C x 1 respectively
%			X:			input_size x batch_size
%			Y:			num_class x batch_size
function gradients = backward_fc(fc_weight, X, Y, post_activation, config, debug_mode)
	if nargin < 6
		debug_mode = true;
	end

	if debug_mode
		assert(size(X, 2) == config.batch_size, 'the batch size of input sample is not correct');
		assert(size(Y, 2) == config.batch_size, 'the batch size of input label is not correct');
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
		activation_flag = true;
		if strcmp(config.activation, 'sigmoid')
			activation = @mysigmoidprime;
		elseif strcmp(config.activation, 'relu')
			activation = @myreluprime;
		elseif strcmp(config.activation, 'tanh')
			activation = @mytanhprime;
		elseif strcmp(config.activation, 'none')
			activation_flag = false;
		else
			assert(false, sprintf('the activation function is not correct in the configuration: %s', config.activation));
		end	
	else
		fprintf('No activation function is specified. Sigmoid is used here.');
		activation = @mysigmoidprime;
	end

	pre_output = W{num_hidden + 1} * post_activation{num_hidden} + repmat(b{num_hidden + 1}, 1, config.batch_size);		% num_class x batch_size
	% assert(~any(isnan(pre_output(:))), 'pre output is nan');
	% assert(~any(isinf(pre_output(:))), 'pre output is nan');

	output = mysoftmax(pre_output);															% num_class x batch_size
	% assert(~any(isnan(output(:))), 'output is nan');
	% assert(~any(isinf(output(:))), 'output is nan');

	% compute the gradient of the final output layer
	delta = output - Y;          															% num_class x batch_size
	grad_W{num_hidden + 1} = (delta * post_activation{num_hidden}') ./ config.batch_size;   						% num_class x batch_size  *  batch_size x num_hidden
	grad_b{num_hidden + 1} = mean(delta, 2);                              					% num_class x 1

	% iteratively compute the gradient for all hidden layers
	for i = num_hidden : -1 : 1
		weight_cur = W{i + 1}';																% num_hidden x num_class

		% post_activation{i}(1:10)
		if activation_flag
			pre_activation = activation(post_activation{i});								% num_hidden x batch_size
		else
			pre_activation = post_activation{i};											
		end
		delta_cur = pre_activation .* (weight_cur * delta);    								% num_hidden x batch_size

		if i == 1
			grad_W{i} = (delta_cur * X') ./ config.batch_size;                                                 	% num_hidden x input_size
			grad_input = delta_cur' * W{1};													% batch_size x input_size

			% assert(~any(isnan(grad_input(:))), 'gradients of w is nan');
			% assert(~any(isinf(grad_input(:))), 'gradients of w is nan');
		else
			grad_W{i} = (delta_cur * post_activation{i-1}') ./ config.batch_size;                         			% num_hidden x num_hidden_previous
		end

		grad_b{i} = mean(delta_cur, 2);
		delta = delta_cur;

		% assert(~any(isnan(grad_W{i}(:))), 'gradients of w is nan');
		% assert(~any(isinf(grad_W{i}(:))), 'gradients of w is nan');
		% assert(~any(isnan(grad_b{i}(:))), 'gradients of b is nan');
		% assert(~any(isinf(grad_b{i}(:))), 'gradients of b is nan');
	end

	gradients = struct();
	gradients.W = grad_W;
	gradients.b = grad_b;
	gradients.input = grad_input;

	% mean(mean(grad_input))
	max(grad_W{1}(:))
	max(grad_W{2}(:))
	max(grad_b{2}(:))
	max(grad_b{1}(:))
	max(grad_input(:))
	% mean(mean(grad_b{1}))
	% pause;
end