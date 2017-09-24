% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% trains the network for one epoch
% This % function should return the updated network parameters 'W' and 'b' after
% performing back-propagation on every data sample.
function weight = train(weight, train_data, train_label, config, debug_mode)
	if nargin < 5
		debug_mode = true;
	end

	if debug_mode
		assert(isfield(config, 'lr'), 'the configuration needs to have a field of learning rate\n');
		assert(isfield(config, 'shuffle'), 'the configuration needs to have a field of shuffle\n');
	end

	% set previous gradients as zero before optimization
	gradients_old = weight;
	num_layer = length(gradients_old.W);
	for layer_index = 1:num_layer
		gradients_old.W{layer_index}(:) = 0;
	end
	for layer_index = 1:num_layer
		gradients_old.b{layer_index}(:) = 0;
	end

	num_data = size(train_data, 1);
	% shuffle the data
	if config.shuffle
		shuffle_id = randperm(num_data);
		train_data = train_data(shuffle_id, :);
		train_label = train_label(shuffle_id, :);
	end

	for i = 1:num_data
		data_temp = train_data(i, :)';  		% N x 1
		label_temp = train_label(i, :)';      	% C x 1 

		[~, post_activation, pre_activation] = forward_fc(weight, data_temp, debug_mode);
		gradients = backward_fc(weight, data_temp, label_temp, post_activation, debug_mode);
		[weight, gradients_old] = update_parameters(weight, gradients, gradients_old, config, debug_mode);

		if mod(i, 100) == 0
			fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b');
			fprintf('Done %.2f %%', i/size(train_data, 1) * 100);
		end
	end
	fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b');

end
