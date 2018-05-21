% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this functions generate an .avi video from a imagelist
function dummy = generate_video_from_list(image_list, save_path)
	assert(iscell(image_list), 'image should be a list.');
	assert_cell = cellfun(@x ischar(x), image_list);
	assert(all(assert_cell), 'some elements in the image list is not a path.');
	assert(ischar(save_path), 'save path is not correct.');

	[parent_dir, filename, ~] = fileparts(save_path);
	video = VideoWriter(fullfile(parent_dir, sprintf('%s.avi', filename)), 'Uncompressed AVI');
	video.FrameRate = 30;
	open(video);
	for i = 1:length(image_list)
		img = imread(image_list{i});
	    writeVideo(video, img);
	end

	close(video);
	dummy = [];
end