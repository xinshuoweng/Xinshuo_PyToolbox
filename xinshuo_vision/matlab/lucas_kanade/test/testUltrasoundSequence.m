close all;
clc;
clear;

load('../data/usseq.mat');
frames = im2double(frames);

number_frame = size(frames, 3);
rects = zeros(number_frame, 4);
initial_rect = [255; 105; 310; 170];
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
    pause(0.05);
    res = getframe;
    if i == 5 - 1
        imwrite(res.cdata, '../results/usOriginalFrame5.jpg');
    elseif i == 25 - 1
        imwrite(res.cdata, '../results/usOriginalFrame25.jpg');
    elseif i == 50 - 1
        imwrite(res.cdata, '../results/usOriginalFrame50.jpg');
    elseif i == 75 - 1
        imwrite(res.cdata, '../results/usOriginalFrame75.jpg');
    elseif i == 100 - 1
        imwrite(res.cdata, '../results/usOriginalFrame100.jpg');
    end
end

save('../results/usseqrects.mat', 'rects');