% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function transfer the format of the input boxes
% input format could be a 1x1 cell, which contains Nx4 (>= 4)
% or input boxes could be a Nx4 (>= 4) matrix or a .mat
% input format: LTBR (x, y)
% output format: LTWH (x, y)

function rectsLTWH = RectLTRB2LTWH(rectsLTRB)
	boxcheck_LTWH(rectsLTWH);
	rectsLTWH = [rectsLTRB(:, 1), rectsLTRB(:, 2), rectsLTRB(:, 3) - rectsLTRB(:,1) + 1, rectsLTRB(:,4) - rectsLTRB(:,2) + 1];
end

