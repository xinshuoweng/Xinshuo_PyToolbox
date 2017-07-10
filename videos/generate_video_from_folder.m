% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this functions generate an .avi video from an image folder
function [num_images] = generate_video_from_folder(img_src, save_path, framerate, downsample_factor, ext_filter, debug_mode)
	if ~exist('debug_mode', 'var')
		debug_mode = true;
	end
	if ~exist('framerate', 'var')
		framerate = 30;
	elseif debug_mode
		assert(isInteger(framerate), 'framerate shoule be an integer.');
	end
	if ~exist('ext_filter', 'var')
		ext_filter = {'.jpg', '.png', '.bmp', '.jpeg'};
	end

	if debug_mode
		assert(ischar(img_src), 'input image folder path is not correct.');
		assert(ischar(save_path), 'save path is not correct.');
	end

	[imagelist, num_images] = load_list_from_folder(img_src, ext_filter, debug_mode);
	fprintf('%d images loaded from %s\n', num_images, img_src);
	[parent_dir, filename, ~] = fileparts(save_path);
	video = VideoWriter(fullfile(parent_dir, sprintf('%s.avi', filename)), 'Uncompressed AVI');
	video.FrameRate = framerate;

	open(video);
	time = tic;
	for i = 1:length(imagelist)
		elapsed = toc(time);
		remaining_str = string(py.timer.format_time(elapsed / i * (num_images - i)));
		elapsed_str = string(py.timer.format_time(toc(time)));

		fprintf('Loading images for videos: %d/%d, EP: %s, ETA: %s\n', i, num_images, elapsed_str, remaining_str);
	    img = imread(imagelist{i});
	    img = imresize(img, 1/downsample_factor);
	    writeVideo(video, img);
	end

	close(video);
end