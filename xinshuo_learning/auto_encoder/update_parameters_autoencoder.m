% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% this function updates the parameters from previous stored weights and the gradients computed in the current iteration
function [weights_updated, gradients_old] = update_parameters_autoencoder(weight, gradients, gradients_old, config, debug_mode)
	if nargin < 5
		debug_mode = true;
	end

	if debug_mode
		assert(length(fieldnames(weight)) == length(fieldnames(gradients)), 'dimension of weights and gradients should be equal');
		assert(length(fieldnames(weight)) == length(fieldnames(gradients_old)), 'dimension of weights and previous gradients should be equal');
		assert(isfield(config, 'lr'), 'the configuration does not contain learning rate');
		assert(isfield(config, 'optim'), 'the configuration does not have a selected optimization method');
		assert(isfield(config, 'weight_decay'), 'the configuration does not have a weight decay parameter');
		assert(isfield(weight, 'W'), 'the weights in fully connected do not exist');
		assert(isfield(weight, 'bias_encode'), 'the bias in fully connected do not exist');
		assert(isfield(weight, 'bias_decode'), 'the bias in fully connected do not exist');
		assert(isfield(gradients, 'W'), 'the weights in gradients do not exist');
		assert(isfield(gradients, 'bias_encode'), 'the bias in gradients do not exist');
		assert(isfield(gradients, 'bias_decode'), 'the bias in gradients do not exist');
	end

	W = weight.W;
	bias_encode = weight.bias_encode;
	bias_decode = weight.bias_decode;
	grad_W = gradients.W;
	grad_bias_encode = gradients.bias_encode;
	grad_bias_decode = gradients.bias_decode;

	gradients_cur = struct();
	gradients_cur.W = grad_W;			% this need to be updated if there is weight decay
	gradients_cur.bias_encode = grad_bias_encode;
	gradients_cur.bias_decode = grad_bias_decode;
	
	% update the parameters in all layers by using a pre-defined optimization method
	grad_w_tmp = - grad_W - config.weight_decay .* W;
	grad_bias_encode_tmp = - grad_bias_encode;
	grad_bias_decode_tmp = - grad_bias_decode;

	% gradients_cur.W{i} = grad_w_tmp;		% update with weight decay
	if strcmp(config.optim, 'sgd')
		W = W + config.lr .* grad_w_tmp;
		bias_encode = bias_encode + config.lr .* grad_bias_encode_tmp;
		bias_decode = bias_decode + config.lr .* grad_bias_decode_tmp;

	elseif strcmp(config.optim, 'momentum')
		if debug_mode
			assert(isfield(config, 'momentum'), 'the configuration does not have a momentum parameter');
		end

		velocity_W = config.lr .* grad_w_tmp + config.momentum .* gradients_old.W;
		velocity_bias_encode = config.lr .* grad_bias_encode_tmp + config.momentum .* gradients_old.bias_encode;
		velocity_bias_decode = config.lr .* grad_bias_decode_tmp + config.momentum .* gradients_old.bias_decode;

		W = W + velocity_W;
		bias_encode = bias_encode + velocity_bias_encode;
		bias_decode = bias_decode + velocity_bias_decode;

		gradients_cur.W = velocity_W;
		gradients_cur.bias_encode = velocity_bias_encode;
		gradients_cur.bias_decode = velocity_bias_decode;
	else
		assert(false, sprintf('%s is not supported in xinshuo''s library', config.optim));
	end

	weights_updated = struct();
	weights_updated.bias_encode = bias_encode;
	weights_updated.bias_decode = bias_decode;
	weights_updated.W = W;

	gradients_old = gradients_cur;
end