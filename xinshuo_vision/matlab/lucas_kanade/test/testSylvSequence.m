close all;
clc;
clear;

load('../data/sylvseq.mat');
load('../data/sylvbases.mat');
sylvseqrects = load('../results/sylvseqrects-wcrt.mat', 'rects');
sylvseqrects = sylvseqrects.rects;
frames = im2double(frames);

number_frame = size(frames, 3);
rects = zeros(number_frame, 4);
initial_rect = [102; 62; 156; 108];
% patch = frames(62:108,102:156,1);
% imshow(patch);
rects(1,:) = initial_rect';
u_total = 0;
v_total = 0;
for i = 1:number_frame-1
    if i == 1
        rect = initial_rect;
        initial_frame = frames(:,:,1);
    end
%     if i == 207
%         keyboard;
%     end
    cur = frames(:, :, i);
    next = frames(:, :, i+1);
    imshow(next);
    % u refers to the rightward offset in width, v refers to the rightward offset in height
    [u, v] = LucasKanadeBasis(cur, next, rect, bases);      % optical flow initialization for (n-1)th frame

    u_total = u_total + u;      % total optical flow
    v_total = v_total + v;
    H = [1, 0, u_total; 0, 1, v_total; 0, 0, 1];    % warp next frame to the first frame reference with the total optical flow
    next_temp = warpH(next, H, size(next));
    
    % correct with the first template
    [new_u, new_v] = LucasKanadeBasis(initial_frame, next_temp, initial_rect, bases);
    v = v + new_v;      % slightly correct the residual optical flow
    u = u + new_u;
    u_total = u_total + new_u;      % accumulate the initial and residual optical flow
    v_total = v_total + new_v;
    
    rect(2) = rect(2) - v;
    rect(4) = rect(4) - v;
    rect(1) = rect(1) - u;
    rect(3) = rect(3) - u;
    
    % uncorrected tracker
    rect_nobasis = [sylvseqrects(i+1, 1:2), sylvseqrects(i+1, 3)-sylvseqrects(i+1, 1)+1, sylvseqrects(i+1, 4)-sylvseqrects(i+1, 2)+1];
    rectangle('Position', rect_nobasis, 'EdgeColor', 'green');
    % corrected tracked
    rectangle('Position',[rect(1) rect(2) rect(3)-rect(1)+1 rect(4)-rect(2)+1], 'EdgeColor', 'yellow');
    fprintf('frame %d\n', i);
    pause(0.001);
    
    rects(i+1,:) = rect';
    res = getframe;
    if i == 2 - 1
        imwrite(res.cdata, '../results/sylvTemplateCorrectionFrame2.jpg');
    elseif i == 200 - 1
        imwrite(res.cdata, '../results/sylvTemplateCorrectionFrame200.jpg');
    elseif i == 300 - 1
        imwrite(res.cdata, '../results/sylvTemplateCorrectionFrame300.jpg');
    elseif i == 350 - 1
        imwrite(res.cdata, '../results/sylvTemplateCorrectionFrame350.jpg');
    elseif i == 400 - 1
        imwrite(res.cdata, '../results/sylvTemplateCorrectionFrame400.jpg');
    end
end

save('../results/sylvseqrects.mat', 'rects');