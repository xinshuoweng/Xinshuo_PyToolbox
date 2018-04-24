% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% find peak locations and scores for convex blob in heatmap
% note that this function is very strict. No smooth for peak points have been applied
% if there are multiple local peak in a same blob, all of them will be returned. 
function [X, Y, score] = find_peaks(heatmap, thre, debug_mode, vis)
    %filter = fspecial('gaussian', [3 3], 2);
    %map_smooth = conv2(map, filter, 'same');
    
    % variable initialization
    if nargin < 4
        vis = false;
        if nargin < 3
            debug_mode = true;
        end
    end

    if debug_mode
        assert(isFloatImage_loose(heatmap), 'input heatmap is not a image.');
        assert(isscalar(thre) && thre <= 1 && thre >= 0, 'threshold is not correct.');
    end

    map_smooth = heatmap;
    map_smooth(map_smooth < thre) = 0;
    
    map_aug = zeros(size(map_smooth,1)+2, size(map_smooth,2)+2);
    map_aug1 = map_aug;
    map_aug2 = map_aug;
    map_aug3 = map_aug;
    map_aug4 = map_aug;
    
    % shift in different directions to find peak, only works for convex blob
    map_aug(2:end-1, 2:end-1) = map_smooth;
    map_aug1(2:end-1, 1:end-2) = map_smooth;        % top
    map_aug2(2:end-1, 3:end) = map_smooth;          % bottom
    map_aug3(1:end-2, 2:end-1) = map_smooth;        % left
    map_aug4(3:end, 2:end-1) = map_smooth;          % right
    peakMap = (map_aug > map_aug1) & (map_aug > map_aug2) & (map_aug > map_aug3) & (map_aug > map_aug4);
    peakMap = peakMap(2:end-1, 2:end-1);
    [Y, X] = find(peakMap);     % find 1
    
    if vis
        figure;
        subplot(2, 3, 1); imshow(map_aug1); title('map aug1');
        subplot(2, 3, 2); imshow(map_aug2); title('map aug2');
        subplot(2, 3, 3); imshow(map_aug3); title('map aug3');
        subplot(2, 3, 4); imshow(map_aug4); title('map aug4');
        subplot(2, 3, 5); imshow(map_aug); title('map aug');
        subplot(2, 3, 6); imshow(heatmap); title('map');
    end

    score = zeros(length(Y),1);
    for i = 1:length(Y)
        score(i) = heatmap(Y(i), X(i));
    end
end