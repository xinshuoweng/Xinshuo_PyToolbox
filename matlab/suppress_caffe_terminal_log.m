% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function prevent caffe output lots of information to terminal log
%	0 - debug
%	1 - info (still a LOT of outputs)
%	2 - warnings
%	3 - errors

function suppress_caffe_terminal_log()
	setenv('GLOG_minloglevel', '2');
end