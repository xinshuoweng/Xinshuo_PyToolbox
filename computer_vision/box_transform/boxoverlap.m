% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function compute the symmetric intersection over union overlap between box_a set of
% bounding boxes in box_a and box_a single bounding box in box_b.

% input box could be box_a 1x1 cell, which contains Nx4 (>=4)
% or input boxes could be box_a Nx4 matrix (>=4)
% input format: LTRB (x, y)
% output format: LTRB (x, y)
% each row specifies box_a bounding box

function overlap = boxoverlap(box_a, box_b)
    % check the format of the box
    boxcheck_LTBR(box_a);
    boxcheck_LTBR(box_b);

    overlap = cell(1, size(box_b, 1));
    for i = 1:size(box_b, 1)
        x1 = max(box_a(:, 1), box_b(i, 1));
        y1 = max(box_a(:, 2), box_b(i, 2));
        x2 = min(box_a(:, 3), box_b(i, 3));
        y2 = min(box_a(:, 4), box_b(i, 4));

        w = x2 - x1 + 1;
        h = y2 - y1 + 1;
        inter = w .* h;
        aarea = (box_a(:, 3) - box_a(:, 1) + 1) .* (box_a(:, 4) - box_a(:, 2) + 1);
        barea = (box_b(i, 3) - box_b(i, 1) + 1) * (box_b(i, 4) - box_b(i, 2) + 1);
        
        % intersection over union overlap
        overlap{i} = inter ./ (aarea + barea - inter);

        % set invalid entries to 0 overlap
        overlap{i}(w <= 0) = 0;
        overlap{i}(h <= 0) = 0;
    end
    overlap = cell2mat(overlap);
end