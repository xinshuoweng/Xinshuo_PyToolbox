% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu


% output is the one from softmax layer
% as well as the hidden layer pre activations, and the hidden layer post
% activations
function [output, post_activation, pre_activation] = forward_hidden_rbm(rbm_weight, var_hidden, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(isstruct(rbm_weight), 'the weight should be a struct \n');
		assert(isfield(rbm_weight, 'W'), 'the weights in RBM do not exist');
		assert(isfield(rbm_weight, 'bias_hidden'), 'the bias of visible variable in RBM do not exist');
	end

	% W, b are cells of matrix to store the weight and bias
	W = rbm_weight.W;
	bias_hidden = rbm_weight.bias_hidden;

	if debug_mode
		assert(size(W, 1) == size(bias_hidden, 1), sprintf('the input number of neurons should be equal in weight and bias: %d vs %d\n', size(W, 1), size(b, 1)));
		assert(1 == size(bias_hidden, 2), 'the second dimension of bias in visible variable should be equal 1\n');
	end

	pre_activation = cell(num_layer-1, 1);
	post_activation = cell(num_layer-1, 1);

	for i = 1:num_layer
		weight = W{i};			% 100 x 784
		bias = b{i};			% 100 x 1
		% size(train_sample)	% 784 x 1

		output_pre = weight * train_sample + bias;				% 100 x 1			

		% output_pre
		if i < num_layer
			pre_activation{i} = output_pre;     				% 100 x 1
		end

		% output_pre(1:10)
		output_pos = activation(output_pre);
		% output_pos(1:10)
		% pause

		if i < num_layer
			post_activation{i} = output_pos;     				% 100 x 1
		elseif i == num_layer      % output the final softmax result
			output = mysoftmax(output_pre);						% 10 x 1
			break;
		end
		train_sample = output_pos;
	end

	if debug_mode
		assert(length(pre_activation) == length(W) - 1, 'the dimension of pre-activation is not correct');
		assert(length(post_activation) == length(W) - 1, 'the dimension of post-activation is not correct');
		for layer_index = 1:num_layer-1
			assert(all(size(pre_activation{layer_index}) == size(fc_weight.b{layer_index})), 'the dimension of pre-activation is not correct');
			assert(all(size(post_activation{layer_index}) == size(fc_weight.b{layer_index})), 'the dimension of post-activation is not correct');
		end
	end

end
