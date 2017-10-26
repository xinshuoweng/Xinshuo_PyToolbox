% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% given a cell of images with same intrinsic matrixes, compute the K
% this functions depends on the orthogonal vanishing point algorithm
% input:
%		image_cell:			N x 1 cell
function K = autocalibrate(image_cell, debug_mode, save_path)
	if nargin < 2
		debug_mode = true;
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

	for image_index = 1:num_images
		image_file_tmp = image_cell{image_index, 1};
		image_tmp = imread(image_file_tmp);
		[~, filename, ~] = fileparts(image_file_tmp);

		save_intermediate_vp_data = fullfile(save_path, sprintf('%s_vp.mat', filename));
		fprintf('save intermediate data for vanishing point in %d image to %s\n', image_index, save_intermediate_vp_data);
		if exist(save_intermediate_vp_data, 'file')
			load(save_intermediate_vp_data);
		else
			[VPs, linemem, p, All_lines] = getVPHedauRaw(image_tmp);
		end

		size(linemem)			% 367 x 1
		size(p)					% 367 x 4
		size(VPs)				% 1 x 6
		VPs = reshape(VPs, 2, 3);
		visualize_image_with_pts(image_tmp, VPs);
		pause;
	end


end