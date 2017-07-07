% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function converts a 2-d matrix in matlab to a 2-d numpy array in python
function [np_array_obj, shape] = convert_to_numpy_array(matrix)
	assert(ismatrix(matrix) && length(size(matrix)) == 2 && ~iscell(matrix), 'input should be a 2-d matrix');

	% construct a cell, where each cell element is one row from original matrix
	cell_matrix = {};
	for i = 1:size(matrix, 1)
		row = matrix(i, :);
		cell_matrix{i} = row;
	end
	cell_matrix

	% cell array corresponds to tuple array in python
	np_array_obj = py.wrapper_function.get_nparray_from_array(cell_matrix);
	shape = size(matrix);
end