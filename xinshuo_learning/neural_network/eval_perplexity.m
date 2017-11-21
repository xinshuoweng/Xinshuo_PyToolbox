% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% this function evaluate the perplexity of a language model
function [perplexity, loss_avg] = eval_perplexity(fc_weights, data, labels, config, debug_mode)
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

	% compute loss and classification rate
	prob_matrix = predictions .* labels;						% num_data x num_class
	prob_vector = sum(prob_matrix, 2);							% num_data x 1
	loss_vector = -log(prob_vector);
	loss_avg = sum(loss_vector) / num_data;

	[~, ex_id] = max(predictions, [], 2);
	[~, gt_id] = max(labels, [], 2);
	accuracy = (sum(ex_id - gt_id == 0)) / num_data;
end


function perplexity = calculatePerplexity(p_s, line_count)
	% for validation set
	M = size(p_s, 1);
	sum_ = 0;
    base = 0;
	for i = 1:size(line_count, 1)
        this_sentence_len = line_count(i);
        p = 1;
        for j = 1:this_sentence_len
            p = p * p_s(j+base);
        end
        sum_ = sum_ + log2(p);
        base = base + this_sentence_len;
    end
	l = (sum_)/M;
	perplexity = 2^(-l);
end