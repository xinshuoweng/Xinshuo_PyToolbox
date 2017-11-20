% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function convert an matrix to a one hot matrix
% inputs:
%			matrix:			an integer
%			ranges:			[min, max], inclusive, both are integers
function onehot = vector2onehot(vector, ranges, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(isvector(vector), 'input vector is not an integer');
		assert(length(ranges) == 2, 'input range is not correct');
		assert(isInteger(ranges(1)) && isInteger(ranges(2)), 'the input range should be integer');
		assert(ranges(1) <= ranges(2), 'the input range is not correct');
		assert(min(vector) >= ranges(1) && max(vector) <= ranges(2), 'the input vector should be in the range of the given ranges');
	end
	num_integers = ranges(2) - ranges(1) + 1;
	index_vector = vector - ranges(1) + 1;

	onehot = bsxfun(@eq, index_vector(:), 1:num_integers);
end