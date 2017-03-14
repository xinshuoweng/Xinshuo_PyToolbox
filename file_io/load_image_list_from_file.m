% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function return a cell which contains a set of image path from a txt file
function [full_image_list, num_image] = load_image_list_from_file(file_path)
	assert(ischar(file_path), 'Input should be a valid path.');
	[~, ~, extension] = fileparts(file_path);
	assert(strcmp(extension, '.txt'), 'File doesn''t have valid extension.');
    file = fopen(file_path, 'r');
    assert(file ~= -1, 'Image list not found');

    full_image_list = textscan(file, '%s', 'Delimiter', '\n');
    full_image_list = full_image_list{1};
    num_image = length(full_image_list);
    fclose(file);