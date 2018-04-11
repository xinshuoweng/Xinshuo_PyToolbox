% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function checks if the input is a 2d array
function valid = is2dTensor(tensor_test)
	valid = ismatrix(tensor_test) && length(size(tensor_test)) == 2 && ~iscell(tensor_test); 
end