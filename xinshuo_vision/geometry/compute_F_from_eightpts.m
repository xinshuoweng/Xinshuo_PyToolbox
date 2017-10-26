% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% use DLT to compute the fundamental matrix
% eightpoint:
%   pts1 - Nx2 matrix of (x,y) coordinates
%   pts2 - Nx2 matrix of (x,y) coordinates
%   M    - max (imwidth, imheight)
%	
%	det(F) = 0
function F = compute_F_from_eightpts(pts1, pts2, M, debug_mode)
	if nargin < 4
		debug_mode = true;
	end

	if debug_mode
		assert(all(size(pts1) == size(pts2)), 'the input point correspondence is not good');
		assert(size(pts1, 2) == 2 && size(pts1, 1) > 0 && length(size(pts1)) == 2, 'the input point does not have a good shape');
	end

	% initialization
	num_pts = size(pts1, 1);

	% normlize the coordinate
	pts1_norm = pts1 ./ M;
	pts2_norm = pts2 ./ M;

	% construct the U matrix
	% U(:, [1, 5]) = pts1_norm.*pts2_norm;            % x1x2 y1y2
	% U(:, 2) = pts1_norm(:, 2).*pts2_norm(:, 1);     % y1*x2
	% U(:, 3) = pts2_norm(:, 1);                      % x2
	% U(:, 4) = pts1_norm(:, 1).*pts2_norm(:, 2);     % x1*y2
	% U(:, 6) = pts2_norm(:, 2);                      % y2
	% U(:, 7) = pts1_norm(:, 1);                      % x1
	% U(:, 8) = pts1_norm(:, 2);                      % y1
	% U(:, 9) = ones(num_pts, 1);               % 1

	U(:, [1, 5]) = pts1_norm.*pts2_norm;            % x1x2 y1y2
	U(:, 2) = pts1_norm(:, 2).*pts2_norm(:, 1);     % y1*x2
	U(:, 3) = pts2_norm(:, 1);                      % x2
	U(:, 4) = pts1_norm(:, 1).*pts2_norm(:, 2);     % x1*y2
	U(:, 6) = pts2_norm(:, 2);                      % y2
	U(:, 7) = pts1_norm(:, 1);                      % x1
	U(:, 8) = pts1_norm(:, 2);                      % y1
	U(:, 9) = ones(num_pts, 1);               % 1

	% U(:, [1, 5]) = pts1_norm.*pts2_norm;            % x1x2 y1y2
	% U(:, 2) = pts1_norm(:, 1).*pts2_norm(:, 2);     % x1*y2
	% U(:, 3) = pts1_norm(:, 1);                      % x1
	% U(:, 4) = pts1_norm(:, 2).*pts2_norm(:, 1);     % y1*x2
	% U(:, 6) = pts1_norm(:, 2);                      % y1
	% U(:, 7) = pts2_norm(:, 1);                      % x2
	% U(:, 8) = pts2_norm(:, 2);                      % y2
	% U(:, 9) = ones(num_pts, 1);               % 1

	% solve the homogenuous least square system Uf = 0
	[~, ~, V] = svd(U);
	% [~, ~, V] = svd(U);
	f = V(:, end);
	F = reshape(f, 3, 3);

	% ensure the singularity of F
	[W, S, V] = svd(F);
	S(3, 3) = 0;
	F = W*S*V';

	% refine the solution
	F = refineF(F, pts1_norm, pts2_norm);

	% unscaling the F
	T = [1/M, 0, 0; 0, 1/M, 0; 0, 0, 1];
	F = T'*F*T;

	F
end