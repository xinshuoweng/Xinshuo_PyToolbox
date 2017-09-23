% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% [accuracy, loss] = ComputeAccuracyAndLoss(W, b, X, Y) computes the networks
% classification accuracy and cross entropy loss with respect to the data samples
% and ground truth labels provided in 'data' and labels'. The function should return
% the overall accuracy and the average cross-entropy loss.

% this function computes the loss for multi-class classification problem
% accuracy is the percentage of correct classification
function [accuracy, loss] = eval_classification(fc_weights, data, labels, num_class, debug_mode)
	if nargin < 5
		debug_mode = true;
	end

	if debug_mode
		assert(isfield(fc_weights, 'W'), 'the weights in fully connected do not exist');
		assert(isfield(fc_weights, 'b'), 'the bias in fully connected do not exist');
		assert(size(data, 1) == size(labels, 1), 'the number of data samples should be equal to number of labels during evaluation');
		assert(size(fc_weights.W{end}, 1) == num_class, 'the number of output in the last layer is not equal to number of classes');
	end
	
	% fc_weights.W{2}
	% pause

	num_data = size(data, 1);
	predictions = zeros(num_data, num_class);			% num_data x num_class
	for data_index = 1 : num_data
	    data_tmp = data(data_index, :)';
		predictions_tmp = forward_fc(fc_weights, data_tmp)';
		% predictions_tmp
		% pause;
		predictions(data_index, :) = predictions_tmp;
	end

	% predictions
	temp1 = predictions .* labels;
	temp2 = sum(temp1, 2);
	loss_total = -log(temp2);
	loss = sum(loss_total) / num_data;
	[~, ex_id] = max(predictions, [], 2);
	[~, gt_id] = max(labels, [], 2);
	accuracy = (sum(ex_id - gt_id == 0))/size(labels, 1);
end
