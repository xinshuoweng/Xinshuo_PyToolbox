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

	epsilon = 1e-5;
	num_rows = size(matrix, 1);
	valid = true;
	for row_index = 1:num_rows
		for col_index = 1:row_index-1
			if abs(matrix(row_index, col_index)) > epsilon
				valid = false;
				break;
			end
		end
	end
end