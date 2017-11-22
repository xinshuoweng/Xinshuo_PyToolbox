% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% this function takes two points in and compute the distance
% parameters
%	pts1:	1 x 2 or 1 x 3
%	pts2: 	1 x 2 or 1 x 3
%
% output
%	distance:	scalar

function distance = distance_matrix(matrix1, matrix2, direction, debug_mode)
	if nargin < 3
		direction = 'rowwise';
	else
		assert(strcmp(direction, 'rowwise') || strcmp(direction, 'colwise'), 'the input direction to compute distance is not correct');
	end

	if nargin < 4
		debug_mode = true;
	end

	if debug_mode
		assert(ismatrix(matrix1) && ismatrix(matrix2), 'the input matrices are not correct');
		assert(all(size(matrix1) == size(matrix2)), 'the shape of input matrices are not equal');
	end

	raw_distance = (matrix1 - matrix2) .^ 2;
	if strcmp(direction, 'rowwise')
		distance = sum(raw_distance, 2);			% summing over all columns 
	else
		distance = sum(raw_distance, 1);			% summing over all rows
	end

	distance = sqrt(distance);
end