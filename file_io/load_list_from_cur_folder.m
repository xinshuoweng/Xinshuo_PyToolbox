% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this script return a cell which contains a set of image path under current folder
function full_image_list = load_list_from_cur_folder(folder_path, ext_filter, debug_mode)
    if nargin < 3
        debug_mode = true;
    end

    if debug_mode
		assert(ischar(folder_path), 'Input path is not valid for obtaining list');
	end
	image_list = dir(folder_path);
	if debug_mode
		assert(~isempty(folder_path), 'The input path not found while obtaining the image list!');
	end
	image_id_list = {image_list(:).name};

	keep_logical = ones(1, length(image_id_list));
	for ext_index = 1:length(ext_filter)
		ext_filter_tmp = ext_filter{ext_index};
		ext_filter_tmp = check_extension(ext_filter_tmp, debug_mode);
		% keep_logical_tmp = cellfun(@(x) ~isempty(x), strfind(image_id_list, ext_filter_tmp));
		keep_logical_tmp = cellfun(@(x) strcmp(x(max(end-length(ext_filter_tmp)+1, 1):end), ext_filter_tmp), image_id_list);
		keep_logical = keep_logical & keep_logical_tmp;
		% disp(keep_logical);
	end
	remove_logical_tmp = cellfun(@(x) strcmp(x, '.') || strcmp(x, '..'), image_id_list);
	keep_logical = keep_logical & ~remove_logical_tmp;

	keep = find(keep_logical);		% find the index of the cell which is image
	image_list = {image_list(keep).name};
	full_image_list = cellfun(@(x) fullfile(folder_path, x), image_list, 'UniformOutput', false);	% concatenate with parent folder path
end