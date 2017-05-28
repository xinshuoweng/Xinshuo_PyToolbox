% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function return a cell array, which contains all valid subfolder under given folder path
% depth control how deep we want to fetch the subfolder list, which should be a scalar
function subfolder_list = get_subfolder_list(folder_path, depth, debug_mode)
    if nargin < 3
        debug_mode = true;
    end

    if debug_mode
    	assert(ischar(folder_path), 'The input path is not vaid while getting subfolder list');
    end

    if ~exist('depth', 'var')
        depth = intmax;
    else
        assert(isNonNegativeInteger(depth), 'The input depth should be an integer while getting subfolder list.');
    end

    if depth > 0       
    	cell_list = dir(folder_path);
        if debug_mode
            assert(~isempty(cell_list), 'The input path not found while obtaining subfolder list!');
        end
    	dir_check = {cell_list(:).isdir};
        dir_name = {cell_list(:).name};

     	keep_folder = cellfun(@(x) x == 1, dir_check);
        keep_folder = keep_folder & ~strcmp(dir_name, '.') & ~strcmp(dir_name, '..');
    	keep = find(keep_folder);		% find the index of the cell which is image
    	cur_subfolder_list = {cell_list(keep).name};
        
        % find subfolder under current dir
        cur_subfolder_path = cellfun(@(x) fullfile(folder_path, x), cur_subfolder_list, 'UniformOutput', false);	% concatenate with parent folder path
        
        % find subfolder recursively
        if ~isempty(cur_subfolder_path)
            sub_subfolder_path = cellfun(@(x) get_subfolder_list(x, depth - 1, debug_mode), cur_subfolder_path, 'UniformOutput', false);	
            if ~isempty(sub_subfolder_path)
                sub_subfolder_list = {};
                for i = 1:length(sub_subfolder_path)
                    sub_subfolder_list = [sub_subfolder_list, sub_subfolder_path{i}];
                end
            end
            subfolder_list = [cur_subfolder_path, sub_subfolder_list];    
        else
            subfolder_list = cur_subfolder_path;
        end
    else    % already get the depth expected
        subfolder_list = {};
    end
end