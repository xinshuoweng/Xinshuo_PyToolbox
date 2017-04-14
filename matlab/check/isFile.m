% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function checks if the input number is a file
function valid = isFile(folder_name_test) 
	if ischar(folder_name_test)
		[~, name, ext] = fileparts(folder_name_test)
		valid = sum(size(name)) > 0 && ~isempty(ext)
	else
		valid = false;
end