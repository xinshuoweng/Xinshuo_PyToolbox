% Author: Xinshuo Weng
% Email: xinshuo.weng@gmail.com

% given a cell array, remove the empty content 
function [ cell_array ] = remove_empty_cell( cell_array )
%REMOVE_EMPTY_CELL Summary of this function goes here
%   Detailed explanation goes here
    assert(iscell(cell_array), 'input is not a valid cell array while removing empty content');
    cell_array = cell_array(~cellfun('isempty',cell_array));   % remove empty cell
end

