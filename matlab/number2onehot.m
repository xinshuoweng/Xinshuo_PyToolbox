% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function convert an integer number to a one hot vector
% inputs:
%			number:			an integer
%			ranges:			[min, max], inclusive, both are integers
function onehot = number2onehot(number, ranges, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(isInteger(number), 'input number is not an integer');
		assert(length(ranges) == 2, 'input range is not correct');
		assert(isInteger(ranges(1)) && isInteger(ranges(2)), 'the input range should be integer');
		assert(ranges(1) <= ranges(2), 'the input range is not correct');
	end

	num_integers = ranges(2) - ranges(1) + 1;
	index = number - ranges(1) + 1;
	onehot = zeros(num_integers, 1);
	onehot(index) = 1;
end