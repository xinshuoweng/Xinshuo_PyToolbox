close all;
clc;
clear;

load('../data/usseq.mat');
usseqrects = load('../results/usseqrects.mat', 'rects');
usseqrects = usseqrects.rects;
frames = im2double(frames);

number_frame = size(frames, 3);
rects = zeros(number_frame, 4);
initial_rect = [255; 105; 310; 170];
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

    % u refers to the rightward offset in width, v refers to the rightward offset in height
    [u, v] = LucasKanadeInverseCompositional(cur, next, rect);      % optical flow initialization for (n-1)th frame

    u_total = u_total + u;      % total optical flow
    v_total = v_total + v;
    H = [1, 0, u_total; 0, 1, v_total; 0, 0, 1];    % warp next frame to the first frame reference with the total optical flow
    next_temp = warpH(next, H, size(next));
    
    % correct with the first template
    [new_u, new_v] = LucasKanadeInverseCompositional(initial_frame, next_temp, initial_rect);
    v = v + new_v;      % slightly correct the residual optical flow
    u = u + new_u;
    u_total = u_total + new_u;      % accumulate the initial and residual optical flow
    v_total = v_total + new_v;
    
    rect(2) = rect(2) - v;
    rect(4) = rect(4) - v;
    rect(1) = rect(1) - u;
    rect(3) = rect(3) - u;
    
    % uncorrected tracker
    rect_uncorrected = [usseqrects(i+1, 1:2), usseqrects(i+1, 3)-usseqrects(i+1, 1)+1, usseqrects(i+1, 4)-usseqrects(i+1, 2)+1];
    rectangle('Position', rect_uncorrected, 'EdgeColor', 'green');
    % corrected tracked
    rectangle('Position',[rect(1) rect(2) rect(3)-rect(1)+1 rect(4)-rect(2)+1], 'EdgeColor', 'yellow');
    pause(0.05);
    
    rects(i+1,:) = rect';
    res = getframe;
    if i == 5 - 1
        imwrite(res.cdata, '../results/usTemplateCorrectionFrame5.jpg');
    elseif i == 25 - 1
        imwrite(res.cdata, '../results/usTemplateCorrectionFrame25.jpg');
    elseif i == 50 - 1
        imwrite(res.cdata, '../results/usTemplateCorrectionFrame50.jpg');
    elseif i == 75 - 1
        imwrite(res.cdata, '../results/usTemplateCorrectionFrame75.jpg');
    elseif i == 100 - 1
        imwrite(res.cdata, '../results/usTemplateCorrectionFrame100.jpg');
    end
end

save('../results/usseqrects-wcrt.mat', 'rects');