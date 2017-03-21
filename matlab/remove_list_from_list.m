% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function assume the source list and target list are both cell and have 1xN or Nx1 dimension
% if there are two or more item in the source list equal to one of item in target list, all of them
% will be deleted
function [list_dst, num_delete, num_ignore] = remove_list_from_list(list_src, list_tar)
	assert(iscell(list_src) && ~isempty(list_src), 'source list is not valid.');
	assert(iscell(list_tar) && ~isempty(list_tar), 'target list to delete is not valid.');

	num_delete = 0;
	num_ignore = 0;
	for i = 1:length(list_tar)
		flag_del = false;
		target_tmp = list_tar{i};
		for j = 1:length(list_src)
			if isequal(list_src{j}, target_tmp)
				list_src{j} = {};		% delete
				num_delete = num_delete + 1;
				flag_del = true;
			end
		end

		if ~flag_del
			num_ignore = num_ignore + 1;
		end
	end

	assert(num_ignore + num_delete == length(list_tar), ...
        'The sum of item deleted and ignored is not equal to the number of item in total to delete.');
	list_dst = remove_empty_cell(list_src);
end