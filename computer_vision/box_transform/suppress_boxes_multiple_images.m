% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function takes boxes and suppress the box by using nms
% it assumes that the boxes are already sorted based on scores
% input format could be a Mx1 cell, each cell saves all boxes for
% each image and have dimension Nx5
% input format: LTRB 
% output format: LTRB 

% when before_nms_topN or after_nms_topN is set to -1, we ignore this operation
% they are basically filtering boxes by the score

function boxes = suppress_boxes_multiple_images(boxes, before_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
    if ~exist('before_nms_topN', 'var')
        before_nms_topN = -1;
    end
    assert(isPositiveInteger(before_nms_topN) || before_nms_topN == -1, 'number of top boxes before nms should be positive integer or -1');
    if ~exist('after_nms_topN', 'var')
        after_nms_topN = -1;
    end
    assert(isPositiveInteger(after_nms_topN) || after_nms_topN == -1, 'number of top boxes after nms should be positive integer or -1');
    assert(nms_overlap_thres > 0 && nms_overlap_thres < 1, 'overlap threshold should be in the range of (0, 1)');
    assert(iscell(boxes) && length(boxes) > 0, 'input boxes are not correct.');
    test_boxes = boxes{1};
    boxcheck_LTRB(test_boxes);
    boxcheck_sortedscore(test_boxes);

    % to speed up nms
    if before_nms_topN > 0
        boxes = cellfun(@(x) x(1:min(size(x, 1), before_nms_topN), :), boxes, 'UniformOutput', false);
    end

    % do nms
    if use_gpu
        for i = 1:length(boxes)
            tic_toc_print('nms: %d / %d \n', i, length(boxes));
            boxes{i} = boxes{i}(nms(boxes{i}, nms_overlap_thres, use_gpu, false), :);
        end
    else
        parfor i = 1:length(boxes)
            boxes{i} = boxes{i}(nms(boxes{i}, nms_overlap_thres, false, false), :);
        end
    end

    aver_boxes_num = mean(cellfun(@(x) size(x, 1), boxes, 'UniformOutput', true));
    fprintf('aver_boxes_num = %d, select top %d\n\n', round(aver_boxes_num), after_nms_topN);
    if after_nms_topN > 0
        boxes = cellfun(@(x) x(1:min(size(x, 1), after_nms_topN), :), boxes, 'UniformOutput', false);
    end
end