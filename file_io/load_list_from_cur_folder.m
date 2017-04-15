% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this script return a cell which contains a set of image path under current folder
function full_image_list = load_list_from_cur_folder(folder_path)
	assert(ischar(folder_path), 'Input path is not valid for obtaining list');
	fprintf('Warning! Only jpeg jpg png bmp extension are supported!\n');
	image_list = dir(folder_path);
	assert(~isempty(folder_path), 'The input path not found while obtaining the image list!');
	image_id_list = {image_list(:).name};
	keep_logical_jpg = cellfun(@(x) ~isempty(x), strfind(image_id_list, 'jpg'));
	keep_logical_jpeg = cellfun(@(x) ~isempty(x), strfind(image_id_list, 'jpeg'));
	keep_logical_png = cellfun(@(x) ~isempty(x), strfind(image_id_list, 'png'));
	keep_logical_bmp = cellfun(@(x) ~isempty(x), strfind(image_id_list, 'bmp'));
	keep_logical = keep_logical_bmp | keep_logical_png | keep_logical_jpeg | keep_logical_jpg;
	keep = find(keep_logical);		% find the index of the cell which is image
	image_list = {image_list(keep).name};
	full_image_list = cellfun(@(x) fullfile(folder_path, x), image_list, 'UniformOutput', false);	% concatenate with parent folder path
end