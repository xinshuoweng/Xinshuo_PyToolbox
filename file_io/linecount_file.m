% Author: Xinshuo Weng
% Email: xinshuo.weng@gmail.com

% this function take a script file as input and count the number of lines within the file
function n = linecount(file_path)
	assert(ischar(file_path), 'The input path is not valid while counting the number of lines.');
	fid = fopen(file_path, 'r');
    n = 0;
    tline = fgetl(fid);
    while ischar(tline)
        tline = fgetl(fid);
        n = n+1;
    end
    fclose(f);
end