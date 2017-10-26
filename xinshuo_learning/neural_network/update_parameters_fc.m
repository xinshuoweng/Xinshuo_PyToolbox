% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% this function updates the parameters from previous stored weights and the gradients computed in the current iteration
function [weights_updated, gradients_old] = update_parameters_fc(weight, gradients, gradients_old, config, debug_mode)
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
		assert(isfield(weight, 'b'), 'the bias in fully connected do not exist');
		assert(isfield(gradients, 'W'), 'the weights in gradients do not exist');
		assert(isfield(gradients, 'b'), 'the bias in gradients do not exist');
	end

	W = weight.W;
	b = weight.b;
	grad_W = gradients.W;
	grad_b = gradients.b;

	gradients_cur = struct();
	gradients_cur.W = grad_W;			% this need to be updated if there is weight decay
	gradients_cur.b = grad_b;
	% update the parameters in all layers by using a pre-defined optimization method
	for i = 1 : length(W)
		grad_w_tmp = - grad_W{i} - config.weight_decay .* W{i};
		grad_b_tmp = - grad_b{i};

		% gradients_cur.W{i} = grad_w_tmp;		% update with weight decay
		if strcmp(config.optim, 'sgd')
			W{i} = W{i} + config.lr .* grad_w_tmp;
			b{i} = b{i} + config.lr .* grad_b_tmp;
		elseif strcmp(config.optim, 'momentum')
			if debug_mode
				assert(isfield(config, 'momentum'), 'the configuration does not have a momentum parameter');
			end

			velocity_W = config.lr .* grad_w_tmp + config.momentum .* gradients_old.W{i};
			velocity_b = config.lr .* grad_b_tmp + config.momentum .* gradients_old.b{i};
			W{i} = W{i} + velocity_W;
			b{i} = b{i} + velocity_b;

			gradients_cur.W{i} = velocity_W;
			gradients_cur.b{i} = velocity_b;
		else
			assert(false, sprintf('%s is not supported in xinshuo''s library', config.optim));
		end
	end

	weights_updated = struct();
	weights_updated.b = b;
	weights_updated.W = W;

	gradients_old = gradients_cur;
end