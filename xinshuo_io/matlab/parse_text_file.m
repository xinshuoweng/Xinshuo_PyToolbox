% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function parse all the stuff in the file as string
% and save it to the cell of cell, each cell is a row of string
% this function assume all string is separate by whitespace
% nrows is the number of rows parse from the file
function [text, nrows, ncols] = parse_text_file(file_path, debug_mode)
    if nargin < 2
        debug_mode = true;
    end

    if debug_mode
    	assert(ischar(file_path), 'The input path should be a string to a file.');
    end

	fid = get_fileID_for_loading(file_path);
	tline = fgetl(fid);
	text = {};
    ncols = [];
	line_count = 1;
    while ischar(tline)
        line_cell = strsplit(tline, ' ');
        line_cell = remove_empty_cell(line_cell, debug_mode);           % remove empty str at the end

        text{line_count} = line_cell;
        ncols = [ncols, length(line_cell)];
        line_count = line_count + 1;
        tline = fgetl(fid);
    end
    nrows = line_count - 1;
    assert(nrows > 0, 'The file is empty.');
    fclose(fid);
end
