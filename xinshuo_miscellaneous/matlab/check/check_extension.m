% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function is to check the extension of a file, to ensure the format is .***
function ext = check_extension(ext_check, debug_mode) 
	if nargin < 2
		debug_mode = true;
	end
	
	if debug_mode
		assert(ischar(ext_check), 'input extension should be a string.');
	end

	if ext_check(1) == '.'
		ext = ext_check;
	else
		ext = strcat('.', ext_check);
	end
end