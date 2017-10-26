% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% trains the restricted Boltzmann machine for one epoch
% This function should return the updated network parameters after
% performing back-propagation on every data sample.
function weight = train_rbm(weight, train_data, config, debug_mode)
	if nargin < 4
		debug_mode = true;
	end

	if debug_mode
		assert(isfield(config, 'lr'), 'the configuration needs to have a field of learning rate\n');
		assert(isfield(config, 'shuffle'), 'the configuration needs to have a field of shuffle\n');
	end

	% set previous gradients as zero before optimization
	gradients_old = weight;				% num_hidden x num_visible
	gradients_old.W(:) = 0;
	gradients_old.bias_hidden(:) = 0;
	gradients_old.bias_visible(:) = 0;

	% shuffle the data
	num_data = size(train_data, 1);
	if config.shuffle
		shuffle_id = randperm(num_data);
		train_data = train_data(shuffle_id, :);
		% train_label = train_label(shuffle_id, :);
	end

	if ~isfield(config, 'batch_size')
		config.batch_size = 1;
	end

	for i = 1:num_data
		data_temp = train_data(i, :)';  		% num_visible x 1
		% label_temp = train_label(i, :)';      	% C x 1 

		positive_visible_sample = data_temp;
		
		% get negative visible sample
		var_visible = data_temp;
		for iter_index = 1:config.sampling_step
			hidden_sample = gibbs_sampling_hidden_from_visible(weight, var_visible, debug_mode);
			var_visible = gibbs_sampling_visible_from_hidden(weight, hidden_sample, debug_mode);
		end
		negative_visible_sample = var_visible;

		% imshow(reshape(positive_visible_sample, 28, 28))
		% positive_visible_sample
		% pause;
		% negative_visible_sample
		% imshow(reshape(negative_visible_sample, 28, 28))
		% pause;

		gradients = compute_gradient_rbm(weight, positive_visible_sample, negative_visible_sample, debug_mode);
		% gradients.W
		[weight, gradients_old] = update_parameters_rbm(weight, gradients, gradients_old, config, debug_mode);

		if mod(i, 100) == 0
			fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b');
			fprintf('Done %.2f %%', i/size(train_data, 1) * 100);
		end
	end
	fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b');

end