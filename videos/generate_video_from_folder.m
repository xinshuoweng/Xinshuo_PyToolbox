% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this functions generate an .avi video from an image folder
function [num_images] = generate_video_from_folder(img_src, save_path)
	assert(ischar(img_src), 'input image folder path is not correct.');
	assert(ischar(save_path), 'save path is not correct.');

	imagelist = load_list_from_folder(img_src);
	[parent_dir, filename, ~] = fileparts(save_path);
	
	video = VideoWriter(fullfile(parent_dir, sprintf('%s.avi', filename)), 'Uncompressed AVI');
	video.FrameRate = 30;
	open(video);
	for i = 1:length(imagelist)
	    img = imread(imagelist{i});
	    writeVideo(video, img);
	end

	close(video);
end