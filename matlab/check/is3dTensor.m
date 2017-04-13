% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function checks if the input is a 3d array
function valid = is3dTensor(tensor_test)
	valid = length(size(tensor_test)) == 3 && ~iscell(tensor_test); 
end