% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

function num_images = resize_images(src_path, resize_size, debug_mode)
	if debug_mode
		assert(ischar(src_path), 'source path is not correct');
		assert(isvector(resize_size) && numel(resize_size) == 2, 'resize size is not correct');
	end

	[image_list, num_images] = load_list_from_folder(src_path, 'jpg');
	save_dir = fullfile(src_path, '../resized');
	mkdir_if_missing(save_dir);

	for i = 1:num_images
		fprintf('processing %d/%d\n', i, num_images);
		image_path_temp = image_list{i};
		[~, filename, ~] = fileparts(image_path_temp);
		image_temp = imread(image_path_temp);
		resized = imresize(image_temp, resize_size);
		save_path = fullfile(save_dir, sprintf('%s.jpg', filename));
		imwrite(resized, save_path);
	end

end