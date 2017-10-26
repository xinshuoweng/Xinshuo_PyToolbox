close all;
clc;
clear;

load('../data/usseq.mat');
usseqrects = load('../results/usseqrects-wcrt.mat', 'rects');
usseqrects = usseqrects.rects;
frames = im2double(frames);

number_frame = size(frames, 3);
% rects = zeros(number_frame, 4);
% initial_rect = [60; 117; 146; 152];
% rects(1,:) = initial_rect';
% u_total = 0;
% v_total = 0;

for i = 1:number_frame-1
    image1 = squeeze(frames(:,:,i));
    image2 = squeeze(frames(:,:,i+1));
    mask = SubtractDominantMotion(image1, image2);
    
    rect_temp = usseqrects(i,:);
    mask_temp = zeros(size(image2));
    mask_temp(round(rect_temp(2):rect_temp(4)), round(rect_temp(1):rect_temp(3))) = 1;
    mask = mask.*mask_temp;
    
    output = imfuse(image2, mask, 'ColorChannels', [2 1 2]);
%     imshow(image1);
    imshow(output);
%     hold on;
    fprintf('frame %d\n', i);
        pause(0.000001);
    %
    %     rects(i+1,:) = rect';
    res = getframe;
    if i == 5 - 1
        imwrite(res.cdata, '../results/usMotionFrame5.jpg');
    elseif i == 25 - 1
        imwrite(res.cdata, '../results/usMotionFrame25.jpg');
    elseif i == 50 - 1
        imwrite(res.cdata, '../results/usMotionFrame50.jpg');
    elseif i == 75 - 1
        imwrite(res.cdata, '../results/usMotionFrame75.jpg');
    elseif i == 100 - 1
        imwrite(res.cdata, '../results/usMotionFrame100.jpg');
    end
end
%
% save('../results/carseqrects-motion.mat', 'rects');