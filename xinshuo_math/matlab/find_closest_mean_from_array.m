% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function takes an array as input, and output the mean of the array, the element in the array which is the closest to the mean and its index
% input:	array, 			N x 1
function [cloest_value, index, mean_value] = find_closest_mean_from_array(array, debug_mode)
	if nargin < 2
		debug_mode = true;
	end

	if debug_mode
		assert(isvector(array) && size(array, 2) == 1 && length(size(array)) == 2, 'the input array is not correct');
	end

	mean_value = mean(array);
	min_diff = realmax;
	best_index = 0;
	for index = 1:length(array)
	    difference = abs(array(index) - mean_value);
	    if difference < min_diff
			min_diff = difference;
			best_index = index;
		end
	end

	if debug_mode
		assert(best_index > 0 && best_index <= length(array), 'the index is wrong');
	end

	index = best_index;
	cloest_value = array(index, 1);
end