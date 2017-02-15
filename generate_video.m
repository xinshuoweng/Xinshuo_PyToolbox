% script for generating video 
clear;
close all;
clc;

% seqname = sprintf('set%02d/V0%02d.seq', i, k);
videoname = 'testing/video/';
% imagefolder = sprintf('images/SET%02d_V0%02d', i, k);
sourcefolder = 'testing/images/';

imagenames = dir(fullfile(sourcefolder, 'output3*.jpg'));
imagenames = {imagenames.name}';
% Is = seqIo(seqname, 'frImgs', [], 'sDir', sourcefolder);
video = VideoWriter(fullfile(videoname, 'output3.avi'), 'Uncompressed AVI');
video.FrameRate = 30;
open(video);

for i = 1:length(imagenames)
    img = imread(fullfile(sourcefolder, imagenames{i}));
    writeVideo(video, img);
end

close(video);
