% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu


% given a list of numbers for fully connected layers, initialize the weight with Xavier initialization
function fc_weight = weights_initialization(net, method, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(isvector(net), 'the input fully connected layers should be a list');
		assert(all(net > 0), 'all fully connected layers should be larger than 0');
	end

	num_layer = length(net);
	W = cell(1, num_layer-1);
	b = cell(1, num_layer-1);

	% compute N_in and N_out for W at each layer
	for i = 1:num_layer-1
		if strcmp(method, 'xavier')
			W{i} = normrnd(0, 2/(net(i) + net(i+1)), [net(i+1), net(i)]);
			b{i} = normrnd(0, 2/(net(i+1) + 1), [net(i+1), 1]);
		elseif strcmp(method, 'gaussian')
			W{i} = normrnd(0, 0.01, [net(i+1), net(i)]);
			b{i} = normrnd(0, 0.01, [net(i+1), 1]);
		else
			assert(false, sprintf('%s initialization method is not supported in xinshuo''s library', method));
		end
	end

	fc_weight = struct();
	fc_weight.W = W;
	fc_weight.b = b;
end
