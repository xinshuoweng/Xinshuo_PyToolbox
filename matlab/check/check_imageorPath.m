% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function gets a file id while saving txt given a path
function img = check_imageorPath(img, debug_mode)
    if nargin < 2
    	debug_mode = true;
    end

    if ischar(img)
        img = imread(img);
    
    if debug_mode
	    assert(isImage(img), 'The input image doesn''t have a good dimension.');
	end
end