% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% this functions remove affine distortion from a affine rectified image
% inputs
%	line_pairs:			2 x 3 matrix, each row represent a line, two lines are orthogonal to each other on the affine rectified image
%
% solve
%	l1 * m1 * s1 + (m1 * l2 + l1 * m2) * s2 = -l2 * m2
%
% 	[a11	a12] * [s1] = [m]
%	[a21	a22]   [s2]	  [n]
function [rectified_img, H] = metric_rectification_affine(img, line_pairs, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	% sanity check
	if debug_mode
		assert(isImage(img), 'the input image is incorrect');
		assert(all(size(line_pairs) == [4, 3]), 'the pairs of parallel lines are not correct');
	end

	% solve equation set
	a11 = line_pairs(1, 1) * line_pairs(2, 1);
	a12 = line_pairs(1, 1) * line_pairs(2, 2) + line_pairs(1, 2) * line_pairs(2, 1);
	a21 = line_pairs(3, 1) * line_pairs(4, 1);
	a22 = line_pairs(3, 1) * line_pairs(4, 2) + line_pairs(3, 2) * line_pairs(4, 1);
	m = -line_pairs(1, 2) * line_pairs(2, 2);
	n = -line_pairs(3, 2) * line_pairs(4, 2);

	params = [a11, a12; a21, a22];
	right_side = [m; n];
	s = params \ right_side;
	conic_dual = [s(1), s(2), 0; s(2), 1, 0; 0, 0, 0];

	% check orthogonality in dual space with affine distortion
	if debug_mode
		dot_product1 = line_pairs(1, :) * conic_dual * line_pairs(2, :)';
		dot_product2 = line_pairs(3, :) * conic_dual * line_pairs(4, :)';
		assert(dot_product1 < 1e-4 && dot_product2 < 1e-4, 'the line in the plane with affine distortion is not orthogonal');
	end

	% check if the computed conic dual is positive definite
	% if debug_mode
	% 	eigen = eig(conic_dual);
	% 	assert(all(eigen >= 0), sprintf('the computed K * K'' is not positive definite, the eigen values are %.3f, %.3f\n', eigen(1), eigen(2)));
	% end

	[L, D] = ldl(conic_dual);
	K = L * sqrt(sqrt(D^2));
	% K = chol(conic_dual(1:2, 1:2), 'lower')														% 2 x 2

	% check the decomposition such that K*K' = conic_dual
	% if debug_mode
	% 	residual = K * K' - conic_dual(1:2, 1:2);
	% 	assert(norm(residual) < 1e-4, 'the Cholesky decomposition is not correct');
	% end
	
	K = [K(1, 1), K(1, 2), 0; K(2, 1), K(2, 2), 0; 0, 0, 1];					% 3 X 3
	H = inv(K);																	% 3 x 3 matrix, this homography remove the affine distortion
	
	% if debug_mode
	% 	conic = [1, 0, 0; 0, 1, 0; 0, 0, 0];

	% 	% % check the transformation between conic and conic in the dual space
	% 	% residual = K * conic * K' - conic_dual;
	% 	% assert(norm(residual) < 1e-4, 'the affine rectification homography is wrong');
		
	% 	% check the orthogonality in the image after affine distortion is removed
	% 	line1_ori = K' * line_pairs(1, :)';			% 3 x 1
	% 	line2_ori = K' * line_pairs(2, :)';
	% 	line3_ori = K' * line_pairs(3, :)';
	% 	line4_ori = K' * line_pairs(4, :)';
	% 	residual1 = line1_ori' * conic * line2_ori;
	% 	residual2 = line3_ori' * conic * line4_ori;
	% 	assert(residual1 < 1e-4 && residual2 < 1e-4, 'two pairs of lines are not orthogonal in the original space');
	% end
	rectified_img = applyH(img, H);
end