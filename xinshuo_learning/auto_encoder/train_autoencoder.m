% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% trains the restricted Boltzmann machine for one epoch
% This function should return the updated network parameters after
% performing back-propagation on every data sample.
function weight = train_autoencoder(weight, train_data, config, debug_mode)
	if nargin < 4
		debug_mode = true;
	end

	if debug_mode
		assert(isfield(config, 'lr'), 'the configuration needs to have a field of learning rate\n');
		assert(isfield(config, 'shuffle'), 'the configuration needs to have a field of shuffle\n');
	end

	if ~isfield(config, 'denoising_level')
		config.denoising_level = 0;
	else
		assert(config.denoising_level >= 0 && config.denoising_level <= 1, 'the denoising level of autoencoder should be [0, 1]');
	end

	% set previous gradients as zero before optimization
	gradients_old = weight;				% num_hidden x num_visible
	gradients_old.W(:) = 0;
	gradients_old.bias_encode(:) = 0;
	gradients_old.bias_decode(:) = 0;

	% shuffle the data
	num_data = size(train_data, 1);
	if config.shuffle
		shuffle_id = randperm(num_data);
		train_data = train_data(shuffle_id, :);
	end

	if ~isfield(config, 'batch_size')
		config.batch_size = 1;
	end

	for i = 1:num_data
		data_temp = train_data(i, :)';  		% num_visible x 1
		data_corrupted = get_corrupted_data(data_temp, config.denoising_level, debug_mode);
		% imshow(reshape(data_corrupted, 28, 28))
		% pause

		var_hidden = encode_autoencoder(weight, data_corrupted, debug_mode);
		reconstucted_data = decode_autoencoder(weight, var_hidden, debug_mode);

		gradients = compute_gradient_autoencoder(weight, data_temp, var_hidden, reconstucted_data, debug_mode);
		[weight, gradients_old] = update_parameters_autoencoder(weight, gradients, gradients_old, config, debug_mode);

		if mod(i, 100) == 0
			fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b');
			fprintf('Done %.2f %%', i/size(train_data, 1) * 100);
		end
	end
	fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b');

end