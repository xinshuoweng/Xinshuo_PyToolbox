% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu


% given a list of numbers for fully connected layers, initialize the weight with Xavier initialization
function [W, b] = xavier_initialize_fc(layers, debug_mode)
	if debug_mode
		assert(isvector(layers), 'the input fully connected layers should be a list');
		assert(all(layers > 0), 'all fully connected layers should be larger than 0');
	end

	number_layer = length(layers);
	W = cell(1, number_layer-1);
	b = cell(1, number_layer-1);

	% compute N_in and N_out for W at each layer
	for i = 1:number_layer-1
		W{i} = normrnd(0, 2/(layers(i) + layers(i+1)), [layers(i+1), layers(i)]);
		b{i} = normrnd(0, 2/(layers(i+1) + 1), [layers(i+1), 1]);
	end
end
