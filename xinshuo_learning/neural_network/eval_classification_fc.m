% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% this function computes the loss for multi-class classification problem
% accuracy is the percentage of correct classification
function [accuracy, loss_avg] = eval_classification_fc(fc_weights, data, labels, config, debug_mode)
	if nargin < 5
		debug_mode = true;
	end

	if debug_mode
		assert(isfield(fc_weights, 'W'), 'the weights in fully connected do not exist');
		assert(isfield(fc_weights, 'b'), 'the bias in fully connected do not exist');
		assert(size(data, 1) == size(labels, 1), 'the number of data samples should be equal to number of labels during evaluation');
		assert(size(fc_weights.W{end}, 1) == config.num_class, 'the number of output in the last layer is not equal to number of classes');
	end

	% inference
	num_data = size(data, 1);
	config_tmp = config.train;
	config_tmp.batch_size = num_data;
	predictions = forward_fc(fc_weights, data', config_tmp, debug_mode)';		% num_data x num_class

	% predictions = zeros(num_data, config.num_class);			% num_data x num_class
	% for data_index = 1 : num_data
	% 	data_tmp = data(data_index, :)';
	% 	predictions_tmp = forward_fc(fc_weights, data_tmp, config.train, debug_mode)';

	% 	size(predictions_tmp)
	% 	predictions(data_index, :) = predictions_tmp;
	% end

	% compute loss and classification rate
	loss_matrix = predictions .* labels;						% num_data x num_class
	loss_vector = sum(loss_matrix, 2);								% num_data x 1
	loss_total = -log(loss_vector);
	loss_avg = sum(loss_total) / num_data;
	[~, ex_id] = max(predictions, [], 2);
	[~, gt_id] = max(labels, [], 2);
	accuracy = (sum(ex_id - gt_id == 0)) / num_data;
end
