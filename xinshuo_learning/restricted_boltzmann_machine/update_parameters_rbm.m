% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% this function updates the parameters from previous stored weights and the gradients computed in the current iteration
function [weights_updated, gradients_old] = update_parameters_rbm(weight, gradients, gradients_old, config, debug_mode)
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
		assert(isfield(weight, 'bias_visible'), 'the bias in fully connected do not exist');
		assert(isfield(weight, 'bias_hidden'), 'the bias in fully connected do not exist');
		assert(isfield(gradients, 'W'), 'the weights in gradients do not exist');
		assert(isfield(gradients, 'bias_visible'), 'the bias in gradients do not exist');
		assert(isfield(gradients, 'bias_hidden'), 'the bias in gradients do not exist');
	end

	W = weight.W;
	bias_hidden = weight.bias_hidden;
	bias_visible = weight.bias_visible;
	grad_W = gradients.W;
	grad_bias_visible = gradients.bias_visible;
	grad_bias_hidden = gradients.bias_hidden;

	gradients_cur = struct();
	gradients_cur.W = grad_W;			% this need to be updated if there is weight decay
	gradients_cur.bias_hidden = grad_bias_hidden;
	gradients_cur.bias_visible = grad_bias_visible;
	
	% update the parameters in all layers by using a pre-defined optimization method
	grad_w_tmp = - grad_W - config.weight_decay .* W;
	grad_bias_hidden_tmp = - grad_bias_hidden;
	grad_bias_visible_tmp = - grad_bias_visible;

	% gradients_cur.W{i} = grad_w_tmp;		% update with weight decay
	if strcmp(config.optim, 'sgd')
		W = W + config.lr .* grad_w_tmp;
		bias_hidden = bias_hidden + config.lr .* grad_bias_hidden_tmp;
		bias_visible = bias_visible + config.lr .* grad_bias_visible_tmp;

	elseif strcmp(config.optim, 'momentum')
		if debug_mode
			assert(isfield(config, 'momentum'), 'the configuration does not have a momentum parameter');
		end

		velocity_W = config.lr .* grad_w_tmp + config.momentum .* gradients_old.W;
		velocity_bias_hidden = config.lr .* grad_bias_hidden_tmp + config.momentum .* gradients_old.bias_hidden;
		velocity_bias_visible = config.lr .* grad_bias_visible_tmp + config.momentum .* gradients_old.bias_visible;

		W = W + velocity_W;
		bias_hidden = bias_hidden + velocity_bias_hidden;
		bias_visible = bias_visible + velocity_bias_visible;

		gradients_cur.W = velocity_W;
		gradients_cur.bias_hidden = velocity_bias_hidden;
		gradients_cur.bias_visible = velocity_bias_visible;
	else
		assert(false, sprintf('%s is not supported in xinshuo''s library', config.optim));
	end

	weights_updated = struct();
	weights_updated.bias_hidden = bias_hidden;
	weights_updated.bias_visible = bias_visible;
	weights_updated.W = W;

	gradients_old = gradients_cur;
end