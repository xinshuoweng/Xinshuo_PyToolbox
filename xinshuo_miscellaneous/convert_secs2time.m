% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% format second to human readable way
function time_str = convert_secs2time(seconds)
	assert(isscalar(seconds), 'the input second is not correct');

	s = mod(int32(seconds), 60);
	m = floor(seconds/60);

	m = mod(m, 60);
	h = floor(m/60);
    % m, s = divmod(int(seconds), 60)
    % h, m = divmod(m, 60)
    time_str = sprintf('%d:%02d:%02d', h, m, s);
end
