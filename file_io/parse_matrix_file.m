% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function parse all the stuff in the file as floating number
% and save it to 2d matrix
% nrows is the number of rows parse from the file
function [data, nrows] = parse_text_file(file_path)
	assert(ischar(file_path), 'The input path should be a string to a file.');
	fid = get_fileID_for_loading(file_path);

	tline = fgetl(fid);
	data = [];
	line_count = 1;
    while ischar(tline)
        line_cell = strsplit(tline, ' ');

        line_cell = cellfun(@(x) str2num(x), line_cell, 'UniformOutput', false);
        data = [data; cell2mat(line_cell)];

        % text{line_count} = line_cell;
        line_count = line_count + 1;
        tline = fgetl(fid);
    end
    nrows = line_count - 1;
    assert(nrows > 0, 'The file is empty.');
    fclose(fid);
end
