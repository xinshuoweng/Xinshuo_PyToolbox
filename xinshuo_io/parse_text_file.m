% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function parse all the stuff in the file as string
% and save it to the cell of cell, each cell is a row of string
% this function assume all string is separate by whitespace
% nrows is the number of rows parse from the file
function [text, nrows] = parse_text_file(file_path)
	assert(ischar(file_path), 'The input path should be a string to a file.');
	fid = get_fileID_for_loading(file_path);

	tline = fgetl(fid);
	text = {};
	line_count = 1;
    while ischar(tline)
        line_cell = strsplit(tline, ' ');
        text{line_count} = line_cell;
        line_count = line_count + 1;
        tline = fgetl(fid);
    end
    nrows = line_count - 1;
    assert(nrows > 0, 'The file is empty.');
    fclose(fid);
end
