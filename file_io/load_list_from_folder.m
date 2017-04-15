% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this script return a cell which contains a set of image path under folder recursively
function [full_image_list, num_image] = load_list_from_folder(folder_path, save_fullpath)
	assert(ischar(folder_path), 'Input path is not valid for obtaining list');
	fprintf('Warning! Only jpeg jpg png bmp extension are supported!\n');

	subfolder_list = get_subfolder_list(folder_path);
    imagelist_subfolder = remove_empty_cell(cellfun(@(x) load_list_from_cur_folder(x), subfolder_list, 'UniformOutput', false));	
    imagelist_curfolder = load_list_from_cur_folder(folder_path);
    full_image_list = [imagelist_curfolder, imagelist_subfolder];	
    num_image = length(full_image_list);
    
    % optional for saving
    if exist('save_fullpath', 'var')
    	mkdir_if_missing(fileparts(save_fullpath));
    	file = fopen(save_fullpath, 'w');
    	assert(file ~= -1, 'file is saved unsuccessfully while obtaining image list');
    	for i = 1:num_image
        	fprintf(file, [full_image_list{i}, '\n']);
    	end
    	fclose(file);
    end
end