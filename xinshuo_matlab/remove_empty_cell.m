% Author: Xinshuo Weng
% Email: xinshuo.weng@gmail.com

% given a cell array, remove the empty content 
% only remove the cell at depth = 1
function cell_array = remove_empty_cell(cell_array, debug_mode)
    if nargin < 2
        debug_mode = true;
    end

    if debug_mode
	    assert(iscell(cell_array), 'input is not a valid cell array while removing empty content');
	end
    cell_array = cell_array(~cellfun('isempty',cell_array));   % remove empty cell
end

