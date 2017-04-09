% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function return the last string of current path
% which is useful during running multiple experiments
function s = process_id()
	d = pwd();
	i = strfind(d, filesep);
	d = d(i(end)+1:end);
	s = d;
end