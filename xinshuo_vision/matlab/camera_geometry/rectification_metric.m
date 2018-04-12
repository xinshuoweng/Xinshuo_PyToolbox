% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% this functions remove projective distortion
% inputs
%	line_pairs:			10 x 3 matrix, each row represent a line, two lines are orthogonal to each other on the affine rectified image
%
% solve
%	l1 * m1 * s1 + (m1 * l2 + l1 * m2) * s2 = -l2 * m2
%
% 	[a11	a12] * [s1] = [m]
%	[a21	a22]   [s2]	  [n]
function [rectified_img, H] = metric_rectification(img, line_pairs, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	% sanity check
	if debug_mode
		assert(isImage(img), 'the input image is incorrect');
		assert(all(size(line_pairs) == [10, 3]), 'the pairs of parallel lines are not correct');
	end

	% solve equation set
	params = zeros(5, 5);
	right_side = zeros(5, 1);
	for line_index = 1:5
		line1_index = 1 + (line_index - 1) * 2;
		line2_index = line1_index + 1;

		params(line_index, 1) = line_pairs(line1_index, 1) * line_pairs(line2_index, 1);
		params(line_index, 2) = (line_pairs(line1_index, 1) * line_pairs(line2_index, 2) + line_pairs(line1_index, 2) * line_pairs(line2_index, 1)) / 2;
		params(line_index, 3) = line_pairs(line1_index, 2) * line_pairs(line2_index, 2);
		params(line_index, 4) = (line_pairs(line1_index, 1) * line_pairs(line2_index, 3) + line_pairs(line1_index, 3) * line_pairs(line2_index, 1)) / 2;
		params(line_index, 5) = (line_pairs(line1_index, 2) * line_pairs(line2_index, 3) + line_pairs(line1_index, 3) * line_pairs(line2_index, 2)) / 2;
		right_side(line_index, 1) = -line_pairs(line1_index, 3) * line_pairs(line2_index, 3);
	end

	s = (params' * params) \ (params' * right_side);
	conic_dual = [s(1), s(2)/2, s(4)/2; s(2)/2, s(3), s(5)/2; s(4)/2, s(5)/2, 1];
	% S = [s1, s2; s2, 1];
	conic_dual
	eig(conic_dual)

	% check orthogonality in dual space with affine distortion
	if debug_mode
		% conic_dual = [s1, s2, 0; s2, 1, 0; 0, 0, 0];
		dot_product1 = line_pairs(1, :) * conic_dual * line_pairs(2, :)'
		dot_product2 = line_pairs(3, :) * conic_dual * line_pairs(4, :)'
		dot_product3 = line_pairs(5, :) * conic_dual * line_pairs(6, :)'
		% dot_product4 = line_pairs(7, :) * conic_dual * line_pairs(8, :)'
		% dot_product5 = line_pairs(9, :) * conic_dual * line_pairs(10, :)'
		% assert(dot_product1 < 1e-4 && dot_product2 < 1e-4 && dot_product3 < 1e-4 && dot_product4 < 1e-4 && dot_product5 < 1e-4, 'the line in the plane with affine distortion is not orthogonal');
	end

	% check if the computed S is positive definite
	% if debug_mode
		% eigen = eig(conic_dual);
		% assert(all(eigen > 0), 'the computed K * K'' is not positive definite');
	% end
	% [U, S, V] = svd(conic_dual)
	% inv(U)


	% K = chol(conic_dual, 'lower');														% 2 x 2

	% K
	% check the decomposition such that K*K' = S
	if debug_mode
		% residual = K * K' - S;
		% assert(norm(residual) < 1e-4, 'the Cholesky decomposition is not correct');
	end
	
	% K = [K(1, 1), K(1, 2), 0; K(2, 1), K(2, 2), 0; 0, 0, 1];					% 3 X 3
	H = inv(U * sqrt(S));																	% 3 x 3 matrix, this homography remove the affine distortion
	
	if debug_mode
		conic = [1, 0, 0; 0, 1, 0; 0, 0, 0];

		% check the transformation between conic and conic in the dual space
		% residual = K * conic * K' - conic_dual;
		% assert(norm(residual) < 1e-4, 'the affine rectification homography is wrong');
		
		% % check the orthogonality in the image after affine distortion is removed
		% line1_ori = K' * line_pairs(1, :)';			% 3 x 1
		% line2_ori = K' * line_pairs(2, :)';
		% line3_ori = K' * line_pairs(3, :)';
		% line4_ori = K' * line_pairs(4, :)';
		% residual1 = line1_ori' * conic * line2_ori;
		% residual2 = line3_ori' * conic * line4_ori;
		% assert(residual1 < 1e-4 && residual2 < 1e-4, 'two pairs of lines are not orthogonal in the original space');
	end
	rectified_img = applyH(img, H);
	imshow(rectified_img)
end