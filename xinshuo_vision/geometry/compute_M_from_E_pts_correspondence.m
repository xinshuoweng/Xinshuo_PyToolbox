% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% compute the projection matrix given 
function M = compute_M_from_E_pts_correspondence(E, pts1, pts2, K1, K2, debug_mode);
	if nargin < 4
		debug_mode = true;
	end

	epsilon = 1e-5;
	if debug_mode
		assert(all(size(pts1) == size(pts2)), 'the input point correspondence is not good');
		assert(size(pts1, 2) == 2 && size(pts1, 1) > 0 && length(size(pts1)) == 2, 'the input point does not have a good shape');

        % check the essential matrix
		assert(all(size(E) == [3, 3]), 'the input intrinsic matrix does not have a good shape');
		resdual = 2 * E * E' * E - trace(E * E') * E;
		assert(norm(resdual) < epsilon, 'the essential matrix is not good');
		assert(det(E) < epsilon, 'the determinant of essential matrix is not close to 0');
	end

	%% get four possible projection matrix
	[U,S,V] = svd(E);
	W = [0,-1,0;1,0,0;0,0,1];

	% Make sure we return rotation matrices with det(R) == 1
	if (det(U*W*V')<0)
	    W = -W;
	end
	M2_set = zeros(3,4,4);
	M2_set(:,:,1) = [U*W*V', U(:,3)./max(abs(U(:,3)))];
	M2_set(:,:,2) = [U*W*V', -U(:,3)./max(abs(U(:,3)))];
	M2_set(:,:,3) = [U*W'*V', U(:,3)./max(abs(U(:,3)))];
	M2_set(:,:,4) = [U*W'*V', -U(:,3)./max(abs(U(:,3)))];

	%% find one valid projection matrix based on point correspondence
	M1 = [eye(3), zeros(3, 1)];
	index = 0;
	valid = false;
	for i = 1:size(M2_set, 3)
		M2 = squeeze(M2_set(:, :, i));

		% triangulate to get 3D points cloud
 		[P{i}, err(i)] = triangulate(pts1, pts2, K1 * M1, K2 * M2, debug_mode);
		if all(P{i}(:, 3) > 0)
			assert(~valid, 'already found another possible extrinsic matrix, bad!!!');
			index = i;
			valid = true;
		end
	end

	assert(index > 0 && index <= 4, 'no good extrinsic matrix found from possible set');
	extrinsic = squeeze(M2_set(:, :, index));
	M = K2 * extrinsic;
end