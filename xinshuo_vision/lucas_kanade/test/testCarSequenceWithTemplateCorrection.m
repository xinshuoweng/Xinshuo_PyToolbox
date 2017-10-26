close all;
clc;
clear;

load('../data/carseq.mat');
carseqrects = load('../results/carseqrects.mat', 'rects');
carseqrects = carseqrects.rects;
frames = im2double(frames);

number_frame = size(frames, 3);
rects = zeros(number_frame, 4);
initial_rect = [60; 117; 146; 152];
rects(1,:) = initial_rect';
u_total = 0;
v_total = 0;
for i = 1:number_frame-1
    if i == 1
        rect = initial_rect;
        initial_frame = frames(:,:,1);
    end
    cur = frames(:, :, i);
    next = frames(:, :, i+1);
    imshow(next);
    %     if i == 29
    %         keyboard;
    %     end
    % u refers to the rightward offset in width, v refers to the rightward offset in height
    [u, v] = LucasKanadeInverseCompositional(cur, next, rect);

    u_total = u_total + u;
    v_total = v_total + v;
    H = [1, 0, u_total; 0, 1, v_total; 0, 0, 1];
    next_temp = warpH(next, H, size(next));
    
    [new_u, new_v] = LucasKanadeInverseCompositional(initial_frame, next_temp, initial_rect);
    v = v + new_v;
    u = u + new_u;
    u_total = u_total + new_u;
    v_total = v_total + new_v;
    
    rect(2) = rect(2) - v;
    rect(4) = rect(4) - v;
    rect(1) = rect(1) - u;
    rect(3) = rect(3) - u;
    
    % uncorrected tracker
    rect_uncorrected = [carseqrects(i+1, 1:2), carseqrects(i+1, 3)-carseqrects(i+1, 1)+1, carseqrects(i+1, 4)-carseqrects(i+1, 2)+1];
    rectangle('Position', rect_uncorrected, 'EdgeColor', 'green');
    % corrected tracked
    rectangle('Position',[rect(1) rect(2) rect(3)-rect(1)+1 rect(4)-rect(2)+1], 'EdgeColor', 'yellow');
    pause(0.0001);
    
    rects(i+1,:) = rect';
    res = getframe;
    if i == 2 - 1
        imwrite(res.cdata, '../results/carTemplateCorrectionFrame2.jpg');
    elseif i == 100 - 1
        imwrite(res.cdata, '../results/carTemplateCorrectionFrame100.jpg');
    elseif i == 200 - 1
        imwrite(res.cdata, '../results/carTemplateCorrectionFrame200.jpg');
    elseif i == 300 - 1
        imwrite(res.cdata, '../results/carTemplateCorrectionFrame300.jpg');
    elseif i == 400 - 1
        imwrite(res.cdata, '../results/carTemplateCorrectionFrame400.jpg');
    end
end

save('../results/carseqrects-wcrt.mat', 'rects');