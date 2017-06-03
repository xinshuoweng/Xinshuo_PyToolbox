% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this functions generate an .avi video from an image folder
function [num_images] = generate_video_from_folder(img_src, save_path, framerate, ext_filter, debug_mode)
	if ~exist('debug_mode', 'var')
		debug_mode = true;
	end
	if ~exist('framerate', 'var')
		framerate = 30;
	elseif debug_mode
		assert(isInteger(framerate), 'framerate shoule be an integer.');
	end
	if ~exist('ext_filter', 'var')
		ext_filter = '.jpg';
	end

	if debug_mode
		assert(ischar(img_src), 'input image folder path is not correct.');
		assert(ischar(save_path), 'save path is not correct.');
	end

	[imagelist, num_images] = load_list_from_folder(img_src, ext_filter, debug_mode);
	[parent_dir, filename, ~] = fileparts(save_path);
	video = VideoWriter(fullfile(parent_dir, sprintf('%s.avi', filename)), 'Uncompressed AVI');
	video.FrameRate = framerate;

	open(video);
	for i = 1:length(imagelist)
	    img = imread(imagelist{i});
	    writeVideo(video, img);
	end

	close(video);
end