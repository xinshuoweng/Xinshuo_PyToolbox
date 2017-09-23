% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% this function updates the parameters from previous stored weights and the gradients computed in the current iteration
function weights_updated = update_parameters(weights, gradients, config, debug_mode)
	if nargin < 4
		debug_mode = true;
	end

	if debug_mode
		assert(length(fieldnames(weights)) == length(fieldnames(gradients)), 'dimension of weights and gradients should be equal');
		assert(isfield(config, 'lr'), 'the configuration does not contain learning rate');
		assert(isfield(config, 'optim'), 'the configuration does not have a selected optimization method');
		assert(isfield(weights, 'W'), 'the weights in fully connected do not exist');
		assert(isfield(weights, 'b'), 'the bias in fully connected do not exist');
		assert(isfield(gradients, 'W'), 'the weights in gradients do not exist');
		assert(isfield(gradients, 'b'), 'the bias in gradients do not exist');
	end

	W = weights.W;
	b = weights.b;

	grad_W = gradients.W;
	grad_b = gradients.b;

	% update the parameters in all layers by using a pre-defined optimization method
	for i = 1 : length(W)
		if strcmp(config.optim, 'sgd')
			W{i} = W{i} - config.lr .* grad_W{i};
			b{i} = b{i} - config.lr .* grad_b{i};
		else
			assert(false, sprintf('%s is not supported in xinshuo''s library', config.optim));
		end
	end

	weights_updated = struct();
	weights_updated.b = b;
	weights_updated.W = W;

	% weights_updated.W{2}
end