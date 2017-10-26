% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function checks if the square matrix is a upper triangular matrix
function valid = isUpper(matrix) 
	if ~ismatrix(matrix)
		valid = false;
	end

	% ensure 2D matrix
	if length(size(matrix)) ~= 2
		valid = false;
	end

<<<<<<< HEAD
	epsilon = 1e-5;
=======
>>>>>>> 4ae903eec3bec55dc48915cc61cc47602218f971
	num_rows = size(matrix, 1);
	valid = true;
	for row_index = 1:num_rows
		for col_index = 1:row_index-1
<<<<<<< HEAD
			if abs(matrix(row_index, col_index)) > epsilon
=======
			if matrix(row_index, col_index) ~= 0
>>>>>>> 4ae903eec3bec55dc48915cc61cc47602218f971
				valid = false;
				break;
			end
		end
	end
end