% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function takes boxes in and remove very small boxes
% input format could be a 1x1 cell, which contains Nx4 
% or input boxes could be a Nx4 matrix or Nx5 matrix
% input format: LTRB 
% output format: LTRB 

function boxes = suppress_boxes_single_image(boxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
    % to speed up nms
    if per_nms_topN > 0
        boxes = cellfun(@(x) x(1:min(size(x, 1), per_nms_topN), :), boxes, 'UniformOutput', false);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1
        if use_gpu
            for i = 1:length(boxes)
                tic_toc_print('nms: %d / %d \n', i, length(boxes));
                boxes{i} = boxes{i}(nms(boxes{i}, nms_overlap_thres, use_gpu), :);
            end
        else
            parfor i = 1:length(boxes)
                boxes{i} = boxes{i}(nms(boxes{i}, nms_overlap_thres), :);
            end
        end
    end
    size(boxes(1))
    aver_boxes_num = mean(cellfun(@(x) size(x, 1), boxes, 'UniformOutput', true));
    fprintf('aver_boxes_num = %d, select top %d\n', round(aver_boxes_num), after_nms_topN);
    if after_nms_topN > 0
        boxes = cellfun(@(x) x(1:min(size(x, 1), after_nms_topN), :), boxes, 'UniformOutput', false);
    end
    size(boxes(1))
end