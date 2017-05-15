% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function initialize a variable with value
function init_var(var_name, var_value)
	assert(ischar(var_name), 'variable name should be a string.');
	% assert(exist('var_value', 'var'), 'please provide a value to initialize the variable.');

	if ~exist(var_name, 'var')
		var_name = var_value;
	end
end