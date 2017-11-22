% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% this function evaluate the perplexity of a language model
function [perplexity, loss_avg] = eval_perplexity(fc_weights, data, labels, length_lines, config, debug_mode)
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
	% prob_matrix(1:2, :)
	% pause

	prob_vector = sum(prob_matrix, 2);							% num_data x 1
	loss_vector = -log(prob_vector);
	loss_avg = sum(loss_vector) / num_data;

	% perplexity = 0;
	perplexity = compute_perplexity(prob_vector, length_lines, debug_mode);
end


function perplexity = compute_perplexity(prob_vector, length_lines, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(isvector(prob_vector), 'the input probability is not a vector');
		assert(isvector(length_lines), 'the length of lines is not a vector');
	end

	num_gram = length(prob_vector);	
	num_lines = length(length_lines);
	accu_num_gram_check = 0;

	% sum_perplexity = 0;
	ave_perplexity = 0;
	gram_index_offset = 0;
	for line_index = 1:num_lines
		length_line_tmp = length_lines(line_index);
		num_gram_line_tmp = length_line_tmp - 3;			% number of 4-gram in the current line
		accu_num_gram_check = accu_num_gram_check + num_gram_line_tmp;

		prob_line = double(1);
		for gram_index = 1:num_gram_line_tmp
			prob_line = prob_line * double(prob_vector(gram_index + gram_index_offset));
		end
		% prob_line
		perplexity_tmp = -log2(prob_line);
		% sum_perplexity = sum_perplexity + perplexity_tmp;
		% sum_perplexity
		% pause
		% n = line_index;
		

		gram_index_offset = gram_index_offset + num_gram_line_tmp;

		past_part = (gram_index_offset - num_gram_line_tmp) / gram_index_offset * ave_perplexity;
		current_part = perplexity_tmp / gram_index_offset;

		if isinf(current_part)
			ave_perplexity = past_part;
		else
			ave_perplexity = past_part + current_part;
		end

		% line_index
		% past_part
		% current_part
		% ave_perplexity
		% pause
		% sum_perplexity / gram_index_offset
		% pause
	end

	assert(gram_index_offset == num_gram, 'the length of gram is not correct');
	assert(accu_num_gram_check == num_gram, 'the length of lines is not correct');

	% ave_perplexity_accu = sum_perplexity / num_gram;

	% ave_perplexity_accu
	% ave_perplexity
	% pause
	% perplexity = 2 ^ ave_perplexity_accu;
	perplexity = 2 ^ ave_perplexity;
end