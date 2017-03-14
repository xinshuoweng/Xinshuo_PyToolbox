% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function takes an range of value as input, and output the parameter for normalizing this set of value
% the normalization should be value = (input + add_para) / div_para
% the normalization range is [0, 1]
function [add_para, div_para] = get_normalize_para(input_range)
	assert(isvector(input_range), 'The input should be a vector denote a set of value could be taken.');

	MAX_VALUE = max(input_range);
	MIN_VALUE = min(input_range);
	add_para = (-1) * MIN_VALUE;
	div_para = MAX_VALUE - MIN_VALUE;
end