close all;
clc;
clear;

load('carseq.mat');
addpath('../../../xinshuo_io');
addpath('../');
mkdir_if_missing('results');

frames = im2double(frames);

number_frame = size(frames, 3);
rects = zeros(number_frame, 4);
initial_rect = [60; 117; 146; 152];
rects(1,:) = initial_rect';
for i = 1:number_frame-1
    if i == 1
        rect = initial_rect;
    end
    cur = frames(:, :, i);
    next = frames(:, :, i+1);
    imshow(next);
    %     if i == 29
    %         keyboard;
    %     end
    % u refers to the rightward offset in width, v refers to the rightward offset in height
    [u, v] = LucasKanadeInverseCompositional(cur, next, rect);
    
    rect(2) = rect(2) - v;
    rect(4) = rect(4) - v;
    rect(1) = rect(1) - u;
    rect(3) = rect(3) - u;
    rects(i+1,:) = rect';
    rectangle('Position',[rect(1) rect(2) rect(3)-rect(1)+1 rect(4)-rect(2)+1], 'EdgeColor', 'yellow');
    pause(0.001);
    res = getframe;
    if i == 2 - 1
        imwrite(res.cdata, 'results/carOriginalFrame2.jpg');
    elseif i == 100 - 1
        imwrite(res.cdata, 'results/carOriginalFrame100.jpg');
    elseif i == 200 - 1
        imwrite(res.cdata, 'results/carOriginalFrame200.jpg');
    elseif i == 300 - 1
        imwrite(res.cdata, 'results/carOriginalFrame300.jpg');
    elseif i == 400 - 1
        imwrite(res.cdata, 'results/carOriginalFrame400.jpg');
    end
end

save('results/carseqrects.mat', 'rects');