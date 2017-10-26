% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% compose the projection matrix
function M = compose_projection(K, R, t, debug_mode);
	if nargin < 4
		debug_mode = true;
	end

	if debug_mode
		assert(all(size(K) == [3, 3]), 'the input intrinsic matrix does not have a good shape');
		assert(all(size(R) == [3, 3]), 'the input rotation matrix does not have a good shape');
		assert(all(size(t) == [3, 1]), 'the input translation vector does not have a good shape');
	end

	M = K * [R, t];
end