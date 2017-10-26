% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% given a cell of images with same intrinsic matrixes, compute the K
% this functions depends on the orthogonal vanishing point algorithm
% input:
%		image_cell:			N x 1 cell
%
% computation
%	[v1 v2 v3] [w1 s w2    [j1]  = 0
%				s  q w3    [j2]
%				w2 w3 w4]  [j3] 	
%
%   [j1v1, j1v3 + j3v1, j2v3 + j3v2, j3v3, j1v2 + j2v1, j2v2] * [w1; w2; w3; w4; s; q] = 0
%
%   if no skew			->  		s = 0
%	if square 			-> 			q = w1
function K = autocalibrate(image_cell, debug_mode, save_path, skew, square_pixel)
	if nargin < 2
		debug_mode = true;
	end

	if nargin < 3
		save_path = 'tmp';
	end
	mkdir_if_missing(save_path);

	if nargin < 4
		skew = false;
	end

	if nargin < 5
		square_pixel = true;
	end

	if debug_mode
		assert(iscell(image_cell), 'the input is not a cell');
		assert(size(image_cell, 2) == 1 && size(image_cell, 1) > 0 && length(size(image_cell)) == 2, 'the input cell does not have a good shape');
	end
	num_images = length(image_cell);
	if debug_mode
		for image_index = 1:num_images
			assert(ischar(image_cell{image_index, 1}), 'the input cell array does not image path inside');
		end
	end

	vp_stack = ones(3*num_images, 3);
	vp_constraint = zeros(num_images*3, 6);
	for image_index = 1:num_images
		image_file_tmp = image_cell{image_index, 1};
		image_tmp = imread(image_file_tmp);
		[~, filename, ~] = fileparts(image_file_tmp);

		save_intermediate_vp_data = fullfile(save_path, sprintf('%s_vp.mat', filename));
		
		if exist(save_intermediate_vp_data, 'file')
			fprintf('load the data for vanishing point in %d image from %s\n', image_index, save_intermediate_vp_data)
			load(save_intermediate_vp_data);
		else
			fprintf('compute and save intermediate data for vanishing point in %d image to %s\n', image_index, save_intermediate_vp_data);
			[VPs, linemem, p, lines] = getVPHedauRaw(image_tmp);
			save(save_intermediate_vp_data, 'VPs', 'linemem', 'p', 'lines');
		end

		VPs = reshape(VPs, 2, 3);
		% size(linemem)
		% linemem(1:10, 1)
		% size(p)
		% p(1:10, :)
		
		% visualize_image_with_pts(image_tmp, VPs);
		% pause;
		vp_stack((image_index-1)*3+1:image_index*3, 1:2) = VPs';		% stack vanishing points

		% convert VP to constraint for computing v^T w j = 0, where v and j are two distinct VPs, w is the conic
		% the equation is [j1v1, j1v3 + j3v1, j2v3 + j3v2, j3v3, j1v2 + j2v1, j2v2]
		vp_constraint((image_index-1)*3 + 1, :) = [VPs(1, 1) * VPs(1, 2), VPs(1, 1) + VPs(1, 2), VPs(2, 1) + VPs(2, 2), 1, VPs(1, 1) * VPs(2, 2) + VPs(2, 1) * VPs(1, 2), VPs(2, 1) * VPs(2, 2)];			% vp1 + vp2 
		vp_constraint((image_index-1)*3 + 2, :) = [VPs(1, 2) * VPs(1, 3), VPs(1, 2) + VPs(1, 3), VPs(2, 2) + VPs(2, 3), 1, VPs(1, 2) * VPs(2, 3) + VPs(2, 2) * VPs(1, 3), VPs(2, 2) * VPs(2, 3)];			% vp2 + vp3 
		vp_constraint((image_index-1)*3 + 3, :) = [VPs(1, 1) * VPs(1, 3), VPs(1, 1) + VPs(1, 3), VPs(2, 1) + VPs(2, 3), 1, VPs(1, 1) * VPs(2, 3) + VPs(2, 1) * VPs(1, 3), VPs(2, 1) * VPs(2, 3)];			% vp1 + vp3 
	end

	if ~skew && square_pixel
		vp_constraint(:, 1) = vp_constraint(:, 1) + vp_constraint(:, 6);			
		vp_constraint = vp_constraint(:, 1:4);
		skew_value = 0;
	end

	[U, S, V] = svd(vp_constraint);
	null_space = V(:, end);					% 4 x 1

	% size(U)
	% size(S)
	% size(V)
	% U
	% S
	% V
	% null_space
	% test_null = null(vp_constraint)
	% vp_constraint * null_space

	conic = [null_space(1), skew_value, null_space(2);
			skew_value,  null_space(1), null_space(3);
			null_space(2), null_space(3), null_space(4)];

	kkt = inv(conic);
	K = chol(kkt, 'upper');
	
	% [L, D] = ldl(kkt);
	% K = L * sqrt(sqrt(D^2));

	% conic
	% kkt
	% K
	% inv(K * K')
end