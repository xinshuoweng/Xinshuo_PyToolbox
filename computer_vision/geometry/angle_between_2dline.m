% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function computes the angle between 2 lines
% inputs
%	line:	1 x 3 vector
% 	angle:	cos theta
function cosine = angle_between_2dline(line1, line2, debug_mode)
	if nargin < 3
		debug_mode = true;
	end

	if debug_mode
		assert(all(size(line1) == [1, 3]), 'the dimension of line is not correct');
		assert(all(size(line2) == [1, 3]), 'the dimension of line is not correct');
	end


	% line1(1:3) = line1(1:3) / line1(3);
	% line2(1:3) = line2(1:3) / line2(3);

	% cosine = line1(1) * line2(1) + line1(2) * line2(2) + line1(3) * line2(3);
	cosine = line1(1) * line2(1) + line1(2) * line2(2);
	% cosine = cosine / (norm(line1) * norm(line2));
	cosine = cosine / (norm(line1(1:2)) * norm(line2(1:2)));
end