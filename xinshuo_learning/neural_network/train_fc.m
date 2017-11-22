% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% trains the network for one epoch
% This % function should return the updated network parameters 'W' and 'b' after
% performing back-propagation on every data sample.
function weight = train_fc(weight, train_data, train_label, config, debug_mode)
	if nargin < 5
		debug_mode = true;
	end

	if debug_mode
		assert(isfield(config, 'lr'), 'the configuration needs to have a field of learning rate\n');
		assert(isfield(config, 'shuffle'), 'the configuration needs to have a field of shuffle\n');
		assert(isfield(config, 'batch_size'), 'the configuration needs to have a field of batch size\n');
	end

	% set previous gradients as zero before optimization
	gradients_old = weight;						% num_hidden x num_visible
	num_layer = length(gradients_old.W);
	for layer_index = 1:num_layer
		gradients_old.W{layer_index}(:) = 0;
	end
	for layer_index = 1:num_layer
		gradients_old.b{layer_index}(:) = 0;
	end
	% if isfield(gradients_old, 'input')
	% 	gradients_old.input(:) = 0;
	% end
	if isfield(weight, 'input')
		back_input = true;
		gradients_old = rmfield(gradients_old, 'input');
	else
		back_input = false;
	end
	% weight.input

	num_data = size(train_data, 1);
	% shuffle the data
	if config.shuffle
		shuffle_id = randperm(num_data);
		% shuffle_id(1:10)
		train_data = train_data(shuffle_id, :);					% N x 48
		train_label = train_label(shuffle_id, :);				% N x num_class		(one-hot vector)
	end	

	for i = 1:config.batch_size:num_data - config.batch_size
		data_temp = train_data(i:i+config.batch_size-1, :)';  			% N x batch_size
		label_temp = train_label(i:i+config.batch_size-1, :)';      	% C x batch_size

		%% preprocessing the input data sample
		if back_input
			volcabulary = weight.input;
			train_sample_parsed = zeros(config.batch_size, 16 * (4 - 1));		% 8000 x 48
			for data_index = 1:config.batch_size
				train_sample_parsed(data_index, 1:16) = volcabulary(data_temp(1, data_index), :);
				train_sample_parsed(data_index, 17:32) = volcabulary(data_temp(2, data_index), :);
				train_sample_parsed(data_index, 33:48) = volcabulary(data_temp(3, data_index), :);
			end
			weight = rmfield(weight, 'input');
		else
			train_sample_parsed = data_temp';				% batch_size x input_size
		end

		% size(train_sample_parsed)

		[~, post_activation, ~] = forward_fc(weight, train_sample_parsed', config, debug_mode);
		gradients = backward_fc(weight, train_sample_parsed', label_temp, post_activation, config, debug_mode);
		% fieldnames(gradients)

		% if back_input
			% assert(isfield(gradients_old, 'input'), 'the gradients of inputs does not exist');
		assert(isfield(gradients, 'input'), 'the gradients of inputs does not exist');
		grad_input = gradients.input;				% batch_size x 48
		grad_input_tmp = - grad_input;				% batch_size x 48		
		gradients = rmfield(gradients, 'input');	
		% else
		% end	

		[weight, gradients_old] = update_parameters_fc(weight, gradients, gradients_old, config, debug_mode);

		%% hard code for updating the inputs
		if back_input
			% value_input = weight.input;				% num_class x 16
			for batch_size_index = 1:config.batch_size
				grad_vol_tmp1 = grad_input_tmp(batch_size_index, 1:16);
				grad_vol_tmp2 = grad_input_tmp(batch_size_index, 17:32);
				grad_vol_tmp3 = grad_input_tmp(batch_size_index, 33:48);

				velocity1 = config.lr .* grad_vol_tmp1;
				velocity2 = config.lr .* grad_vol_tmp2;
				velocity3 = config.lr .* grad_vol_tmp3;

				volcabulary(data_temp(1, batch_size_index), :) = volcabulary(data_temp(1, batch_size_index), :) + velocity1;
				volcabulary(data_temp(2, batch_size_index), :) = volcabulary(data_temp(2, batch_size_index), :) + velocity2;
				volcabulary(data_temp(3, batch_size_index), :) = volcabulary(data_temp(3, batch_size_index), :) + velocity3;
				% value_input = value_input + config.lr .* grad_input_tmp;

				% gradients_old1 = velocity1;
				% gradients_old2 = velocity2;
				% gradients_old3 = velocity3;
			end
			weight.input = volcabulary;
		end

		if mod(i, 10*config.batch_size) == 1
			fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b');
			fprintf('Done %.2f %%', i/num_data * 100);
		end
	end
	fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b');

end
