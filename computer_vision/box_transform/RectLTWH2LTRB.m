% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function transfer the format of the input boxes
% input format could be a 1x1 cell, which contains Nx4 (>= 4)
% or input boxes could be a Nx4 (>= 4) matrix or a .mat
% input format: LTWH (x, y)
% output format: LTBR (x, y)

function rectsLTRB = RectLTWH2LTRB(rectsLTWH)
	rectsLTRB = [rectsLTWH(:, 1), rectsLTWH(:, 2), rectsLTWH(:, 1)+rectsLTWH(:,3)-1, rectsLTWH(:,2)+rectsLTWH(:,4)-1];
end

