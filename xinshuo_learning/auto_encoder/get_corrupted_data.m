% Author: Xinshuo Weng
% Email: xinshuow@andrew.cmu.edu

% this function corrupts the input data with a level
% inputs
%		data_sample, 			num_pixel x 1
%		corrupted_level,		[0, 1], 0 is no corruption, 1 is full corruption
function data_sample = get_corrupted_data(data_sample, corrupted_level, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(corrupted_level >= 0 && corrupted_level <= 1, 'the corruption level is not correct');
		assert(size(data_sample, 2) == 1, 'the input data sample does not have a good shape');
	end

	N = ones(size(data_sample));
	corrupted_level_matrix = zeros(size(data_sample));
	corrupted_level_matrix(:) = corrupted_level;
	mask_pixel = binornd(N, corrupted_level_matrix); 		% sample the masked pixels 
	data_sample(find(mask_pixel)) = 0;
end