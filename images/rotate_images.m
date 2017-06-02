% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function is to rotate a set of images given source folder and rotation degree
% parameter:
%	src_path:		source image folder, we will scan all file under given extension in this folder and subfolders
%	save_dir:		the root directory to save the result images
%	angle_degree:	how much angle in degree we should rotate
%	ext_filder:		the extension to scan
%	debug_mode:		turn on to check all stuff but slow down the running speed
%	vis:			visualization of result image on the fly
%	force:			overwrite the files if already exists
function num_images = rotate_images(src_path, save_dir, angle_degree, ext_filter, debug_mode, vis, force)
	if ~exist('debug_mode', 'var')
		debug_mode = true;
	end

	if ~exist('vis', 'var')
		vis = false;
	end

	if ~exist('force', 'var')
		force = false;
	end

	if ~exist('ext_filter', 'var')
		ext_filter = '.jpg';
	end

	if debug_mode
		assert(ischar(src_path), 'source path is not correct');
		assert(isscalar(angle_degree), 'angle of rotation is not correct');
		assert(ischar(save_dir), 'save path is not correct');
		assert(islogical(force), 'force to overwrite flag is not correct.');
		assert(islogical(vis), 'visualization flag is not correct.');
	end
	mkdir_if_missing(save_dir);
	ext_filter = check_extension(ext_filter, debug_mode);

	% load image list
	[image_list, num_images] = load_list_from_folder(src_path, ext_filter, debug_mode);
	fprintf('\nnumber of images to rotate is %d\n\n', num_images);

	% rotate all images
    time = tic;
	for i = 1:num_images
		image_path_temp = image_list{i};
		[~, filename, ~] = fileparts(image_path_temp);

		% counting time
		elapsed = toc(time);
        remaining_str = string(py.timer.format_time(elapsed / i * (num_images - i)));
        elapsed_str = string(py.timer.format_time(toc(time)));
        fprintf('rotating %d/%d, path: %s, elapsed time: %s, remaining time: %s\n', i, num_images, image_path_temp, elapsed_str, remaining_str);

        % process
        save_path = fullfile(save_dir, strcat(filename, ext_filter));
        if ~force && exist(save_path, 'file')
        	continue;
        end
		image_temp = imread(image_path_temp);
		rotated = imrotate(image_temp, angle_degree);
		if vis
			imshow(rotated);
		end
		imwrite(rotated, save_path);
	end

	% done
	fprintf('\ndone!!!!\n\n');
end