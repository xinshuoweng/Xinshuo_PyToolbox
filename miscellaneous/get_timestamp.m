% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function return a string to represent the current timestamp
function time = get_timestamp()
	time = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
end