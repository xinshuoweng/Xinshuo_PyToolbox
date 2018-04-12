% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% use DLT and determinant to compute the fundamental matrix with 7 pts correspondence
% eightpoint:
%   pts1 - Nx2 matrix of (x,y) coordinates
%   pts2 - Nx2 matrix of (x,y) coordinates
%   normlize_factor    - max (imwidth, imheight)
%	
%	det(F) = 0
function F = compute_F_from_7pts(pts1, pts2, normlize_factor, debug_mode)
	if nargin < 4
		debug_mode = true;
	end

	if debug_mode
		assert(all(size(pts1) == size(pts2)), 'the input point correspondence is not good');
		assert(size(pts1, 2) == 2 && size(pts1, 1) > 0 && length(size(pts1)) == 2, 'the input point does not have a good shape');
	end

	num_pts = size(pts1, 1);
	epsilon = 1e-5;

	% normlize the coordinate
	pts1_norm = pts1 / normlize_factor;
	pts2_norm = pts2 / normlize_factor;

	% construct the U matrix
	U(:, [1, 5]) = pts1_norm .* pts2_norm;
	U(:, 2) = pts1_norm(:, 2) .* pts2_norm(:, 1);
	U(:, 3) = pts2_norm(:, 1);
	U(:, 4) = pts1_norm(:, 1) .* pts2_norm(:, 2);
	U(:, 6) = pts2_norm(:, 2);
	U(:, 7) = pts1_norm(:, 1);
	U(:, 8) = pts1_norm(:, 2);
	% U(:, 9) = ones(7, 1);
	U(:, 9) = ones(num_pts, 1);

	% solve the homogenuous least square system Uf = 0
	[E, S, V] = svd(U);
	f1 = V(:, end);
	f2 = V(:, end-1);
	F1 = reshape(f1, 3, 3);
	F2 = reshape(f2, 3, 3);

	syms lambda
	func = symfun(det((1-lambda)*F1 + lambda*F2), lambda);
	res = double(root(func, lambda));
	F_check = (1-res(1))*F1 + res(1)*F2;
	assert(abs(det(F_check)) < epsilon, 'the F is not good');
	% for i = 1:length(res)
	%     if isreal(res(i))
	%         continue;
	%     else
	%         res(i) = [];    
	%     end
	% end

	% ensure the singularity of F
	[W, S, V] = svd(F_check);
	S(3, 3) = 0;
	F = W*S*V';

	% refine the solution
	F_refine = refineF(F, pts1_norm, pts2_norm);

	% unscaling the F
	T = [1/normlize_factor, 0, 0; 0, 1/normlize_factor, 0; 0, 0, 1];
	F = T'*F_refine*T;
end