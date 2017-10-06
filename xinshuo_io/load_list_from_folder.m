% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this script return a cell which contains a set of image path under folder recursively
% note that the list returned in this function is ordered
function [full_image_list, num_image] = load_list_from_folder(folder_path, ext_filter, depth, debug_mode, save_fullpath)
    if nargin < 3
        depth = 1;
    end

    if nargin < 4
        debug_mode = true;
    end

    if debug_mode
    	assert(ischar(folder_path), 'Input path is not valid for obtaining list');
        assert(depth >= 1, 'the depth is not correct');
    end

    if ~exist('ext_filter', 'var')
        ext_filter = {};
    elseif ~iscell(ext_filter)
        ext_filter = {ext_filter};
    end

    % get folder and file from current depth
    imagelist_curfolder = load_list_from_cur_folder(folder_path, ext_filter, debug_mode);

    subfolder_list_recur = get_subfolder_list(folder_path, depth-1, debug_mode);
    imagelist_subfolder = remove_empty_cell(cellfun(@(x) load_list_from_cur_folder(x, ext_filter, debug_mode), subfolder_list_recur, 'UniformOutput', false), debug_mode);

    % merge all sub-imagelist
    full_image_list = imagelist_curfolder;
    for cell_index = 1:length(imagelist_subfolder)
        full_image_list = [full_image_list, imagelist_subfolder{cell_index}];
    end
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