
% TODO: CHECK
% script for generating video 
function [num_images] = generate_video_from_folder(img_src, save_path, videoname)

% seqname = sprintf('set%02d/V0%02d.seq', i, k);
% imagefolder = sprintf('images/SET%02d_V0%02d', i, k);
% sourcefolder = img_src;

% imagenames = dir(fullfile(sourcefolder, 'output3*.jpg'));
% imagenames = {imagenames.name}';
imagelist = load_list_from_folder(img_src);
% Is = seqIo(seqname, 'frImgs', [], 'sDir', sourcefolder);
video = VideoWriter(fullfile(save_path, sprintf('%s.avi', videoname)), 'Uncompressed AVI');
video.FrameRate = 30;
open(video);

for i = 1:length(imagelist)
    img = imread(imagelist{i});
    writeVideo(video, img);
end

close(video);
