% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function assume the source string include a substring, and return the removed one.
% is the substring is inside the source string, then concatenate the two sub-parts after removing the substring
function [removed, valid, pre_part, pos_part] = remove_str_from_str(src_str, substr, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(ischar(src_str), 'the source string is not correct.');
		assert(ischar(substr), 'the sub-string is not correct.');
	end

	start_ind = findstr(substr, src_str);
	if isempty(start_ind)		% substring is not found in the source
		removed = src_str;
		valid = false;
		pre_part = [];
		pos_part = [];
	else
		if start_ind > 1
			pre_part = src_str(1:start_ind-1);
		else
			pre_part = [];
		end

		pos_part = src_str(start_ind + length(substr) : end);
		removed = [pre_part, pos_part];
		valid = true;
	end
end