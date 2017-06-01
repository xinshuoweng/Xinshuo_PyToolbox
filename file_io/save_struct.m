% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function convert a struct to string (only applied to the field which can be converted to a string) and save it to file
function nrows = save_struct(struct_save, save_path, debug_mode)
    if ~exist('debug_mode', 'var')
        debug_mode = true;
    end

    if debug_mode
    	assert(ischar(save_path), 'The input path should be a string to a file.');
    	assert(isstruct(struct_save), 'The stuff to save should be a struct.');
    end
	file_id = get_fileID_for_saving(save_path);
    nrows = 0;

    % convert struct to string and write to file
    fields = fieldnames(struct_save);
    for field_index = 1:length(fields)
        fields_tmp = fields{field_index};
        value = getfield(struct_save, fields_tmp);

        if isstruct(value)
            continue;
        elseif isscalar(value)
            try
                str_tmp = num2str(value);
            catch
                str_tmp = 'failed to convert to string!!!';
                if debug_mode
                    fprintf('field %s cannot converted to string\n', fields_tmp);
                end
            end
        elseif ismatrix(value)
            try
                str_tmp = mat2str(value);
            catch
                str_tmp = 'failed to convert to string!!!';
                if debug_mode
                    fprintf('field %s cannot converted to string\n', fields_tmp);
                end
            end            
        elseif ischar(value)
            str_tmp = value;
        else
            continue;
        end

        % write the key and value string
        fprintf(file_id, '%-30s\t\t\t\t', fields_tmp); 
        fprintf(file_id, '%s\n', str_tmp); 
        nrows = nrows + 1;
    end
    
    fclose(file_id);
end
