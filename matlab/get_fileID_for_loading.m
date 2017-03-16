% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

function fileID = get_fileID_for_loading(load_path)
	assert(ischar(load_path), 'The loading path should be a char while getting fileID.');
	fileID = fopen(load_path, 'r');
	assert(fileID ~= -1, 'The file is not found in the loading path while getting fileID.');
end