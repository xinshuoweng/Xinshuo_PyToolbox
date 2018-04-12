close all;
clc;
clear;

load('../data/aerialseq.mat');
frames = im2double(frames);
number_frame = size(frames, 3);

for i = 1:number_frame-1
    image1 = squeeze(frames(:,:,i));
    image2 = squeeze(frames(:,:,i+1));
    if i == 108
        keyboard;
    end
    mask = SubtractDominantMotion(image1, image2);

    output = imfuse(image1, mask, 'ColorChannels', [2 1 2]);
%     imshow(image1);
    imshow(output);
%     hold on;
    fprintf('frame %d\n', i);
        pause(0.01);
    %
    %     rects(i+1,:) = rect';
    res = getframe;
    if i == 30
        imwrite(res.cdata, '../results/aerialMotionFrame30.jpg');
    elseif i == 60
        imwrite(res.cdata, '../results/aerialMotionFrame60.jpg');
    elseif i == 90
        imwrite(res.cdata, '../results/aerialMotionFrame90.jpg');
    elseif i == 120
        imwrite(res.cdata, '../results/aerialMotionFrame120.jpg');
    end
end
%
% save('../results/carseqrects-wcrt.mat', 'rects');