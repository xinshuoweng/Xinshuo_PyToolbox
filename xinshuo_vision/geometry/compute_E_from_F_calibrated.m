% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% compute essential matrix from fudamental matrix after calibration
%	det(E) = 0
% 	2EE'E - trace(EE')E = 0
function E = compute_E_from_F_calibrated(F, K1, K2, debug_mode);
	if nargin < 4
		debug_mode = true;
	end

	epsilon = 1e-5;
	if debug_mode
		assert(all(size(F) == [3, 3]), 'the input fudamental matrix does not have a good shape');
		assert(all(size(K1) == [3, 3]), 'the input intrinsic matrix1 does not have a good shape');
		assert(all(size(K2) == [3, 3]), 'the input intrinsic matrix2 does not have a good shape');
		assert(isUpper(K1) && isUpper(K2), 'the input intrinsic matrix is not upper triangular');
		assert(det(F) < epsilon, 'the determinant of fundamental matrix is not close to 0')
	end

	E = K2' * F * K1;

	% enforce the essential matrix to have two identical eigenvalue of 1
	[U, S, V] = svd(E);
	diag_value = (S(1, 1) + S(2, 2)) / 2.0;
	E = U * [diag_value, 0, 0; 0, diag_value, 0; 0, 0, 0] * V';		

	% ensure singularity
	if debug_mode
		assert(det(E) < epsilon, 'the determinant of essential matrix is not close to 0');
	end

	% ensure the property of essential matrix
	if debug_mode
		resdual = 2 * E * E' * E - trace(E * E') * E;
		assert(norm(resdual) < epsilon, 'the essential matrix is not good');
	end
end