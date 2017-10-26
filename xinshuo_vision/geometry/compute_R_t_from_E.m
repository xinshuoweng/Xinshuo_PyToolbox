% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% decompose the essential matrix to rotation and translation 
function [R, t] = compute_R_t_from_E(E, debug_mode);
	if nargin < 2
		debug_mode = true;
	end

<<<<<<< HEAD
	epsilon = 1e-5;
	if debug_mode
		assert(all(size(E) == [3, 3]), 'the input essential matrix does not have a good shape');
		assert(det(E) < epsilon, 'the determinant of essential matrix is not close to 0');
=======
	if debug_mode
		assert(all(size(E) == [3, 3]), 'the input essential matrix does not have a good shape');
>>>>>>> 4ae903eec3bec55dc48915cc61cc47602218f971
	end

	[U S V] = svd(E);
	W = [0, -1, 0; 1, 0, 0; 0, 0, 1];

	t_skew = U * W * S * U';
	R = U * W' * V'; 
<<<<<<< HEAD

	t_skew
=======
>>>>>>> 4ae903eec3bec55dc48915cc61cc47602218f971
end