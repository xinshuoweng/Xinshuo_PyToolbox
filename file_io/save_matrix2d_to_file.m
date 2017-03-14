% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function save a 2d matrix to a file line by line
function [nrows, ncols] = save_matrix2d_to_file(matrix, save_path)
	assert(~isempty(matrix), 'The matrix to save is empty while saving to file.');
    fileID = get_fileID_for_saving(save_path);
    [nrows, ncols] = size(matrix);

    for i = 1:nrows
        for j = 1:ncols
        	fprintf(fileID, '%05.5f ', matrix(i, j));     
        end
        fprintf(fileID, '\n');
    end
    
    fclose(fileID);
end