% Author: Xinshuo Weng
% Email: xinshuo.weng@gmail.com


% this function generate a concatenated video in grid from a list of image folders
% inputs:
%		image_folder_list:		a cell of image folders
%		im_size:				(height, width), final concatenated image size
%		grid_size:				(number of rows, number of cols)
%		save_dir:				output image folder
function concatenate_video_grid(image_folder_list, save_dir, im_size, grid_size, ext_filter, framerate, vis_resize_factor, force, edge_factor, start_frame, end_frame, debug_mode)
	if nargin < 3
		im_size = [1600, 2560];
	end

	if nargin < 4
		num_rows = floor(sqrt(num_videos));
		num_cols = ceil(sqrt(num_videos));
		while num_cols * num_rows < num_videos
			num_cols = num_cols + 1;
		end		
		grid_size = [num_rows, num_cols];
	end

	if nargin < 5
		ext_filter = {'.png', '.jpg', '.jpeg'};
	end

	if nargin < 6
		framerate = 30;
	end

	if nargin < 7
		vis_resize_factor = 1;
	end

	if nargin < 8
		force = true;
	end

	if nargin < 9
		edge_factor = 0.99;
	end

	if nargin < 10
		start_frame = 1;
	end

	if nargin < 12
		debug_mode = true;
	end

	if debug_mode
		assert(length(im_size) == 2, 'the image size is not correct');
		assert(length(grid_size) == 2, 'the grid size is not correct');
		assert(length(image_folder_list) > 0, 'the image folder list should not be empty');
	end

	depth = 1;
	mkdir_if_missing(save_dir);
	window_height = im_size(1);
	window_width = im_size(2);
	num_rows = grid_size(1);
	num_cols = grid_size(2);
	num_videos = length(image_folder_list);

	grid_height = floor(window_height / num_rows);
	grid_width  = floor(window_width  / num_cols);
	im_height   = floor(grid_height   * edge_factor);
	im_width 	= floor(grid_width 	* edge_factor);
	im_channel 	= 3;

	num_image_cell = {};
	image_list_cell = {};
	for video_index = 1:num_videos
		fprintf('loading image list for all folders %d/%d\n', video_index, num_videos);
		[imagelist, num_images] = load_list_from_folder(image_folder_list{video_index}, ext_filter, depth, debug_mode);
		num_image_cell{video_index} = num_images;
		image_list_cell{video_index} = imagelist;
	end
	num_images = num_image_cell{1};

	if debug_mode
		for video_index = 1:num_videos
			assert(num_image_cell{video_index} == num_images, 'the number of images loaded in all folders are not equal');
		end
	end
	fprintf('%d images loaded for %d folders\n', num_images, num_videos);
	if nargin < 11
		end_frame = num_images;
	end

	% concatenate
	image_merged = zeros(window_height, window_width, im_channel);
	for image_index = start_frame:end_frame
		image_list_tmp = image_list_cell{1};
		image_file = image_list_tmp{image_index};
		[~, filename, ~] = fileparts(image_file);
		save_path_tmp = fullfile(save_dir, sprintf('%s.jpg', filename));
		if exist(save_path_tmp, 'file') && ~force
			continue;
		end

		fprintf('processing %d/%d\n', image_index, num_images);
		
		% load image at current frame
		for video_index = 1:num_videos
			image_list_tmp = image_list_cell{video_index};
			image_file = image_list_tmp{image_index};
			image_tmp = imread(image_file);
			image_tmp = imresize(image_tmp, [im_height, im_width]);
			assert(length(size(image_tmp)) && size(image_tmp, 3) == im_channel, 'the image read does not have correct dimension');

			rows_index = ceil(video_index / num_cols);
			cols_index = video_index - (rows_index - 1) * num_cols;
			rows_start = 1 + (rows_index - 1) * grid_height;
			rows_end   = rows_start + im_height - 1;
			cols_start = 1 + (cols_index - 1) * grid_width;
			cols_end   = cols_start + im_width - 1;

			image_merged(rows_start : rows_end, cols_start : cols_end, :) = image_tmp;
		end
		image_merged = uint8(image_merged);
		imwrite(image_merged, save_path_tmp);
	end

	video_savepath = fullfile(save_dir, 'concatenated.avi');
	% img_src = fullfile(video_data_root_dir, 'images/cam330030');
	% img_src = video_data_root_dir;
	% img_src = video_data_root_dir;
	fprintf('generating video %d.\n', i);
	num_images = generate_video_from_folder(save_dir, video_savepath, framerate, vis_resize_factor, {'.jpg'}, debug_mode);
	fprintf('%d images loaded at %s...\n\n', num_images, save_dir);
	fprintf('\ndone!!!!!!!!!!!\n\n');

end