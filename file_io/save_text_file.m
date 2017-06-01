% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function stores a set of stuff to a file
% this function assumes all stuff to save is in a 1d cell of 1d cell 
% each stuff is separated by whitespace
function nrows = save_text_file(content_cell, file_path)
	assert(ischar(file_path), 'The input path should be a string to a file.');
	assert(iscell(content_cell), 'The stuff to save should be a cell.');
	fid = get_fileID_for_saving(file_path);

    nrows = length(content_cell);

    for i = 1:nrows
    	row_tmp = content_cell{i};	% still a cell storing content in this row
    	row_string = cellfun(@(x) num2str(x), row_tmp, 'UniformOutput', false);	% concatenate with parent folder path
        string_tmp = strjoin(row_string, ' ');
        fprintf(fid, '%s\n', string_tmp);    
    end
    
    fclose(fid);
end
