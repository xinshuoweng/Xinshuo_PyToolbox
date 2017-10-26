% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu


% given a set of points, return a set of bbox which are centered at the points

% parameters:
%     pts_array:      2 x num_pts

% return:
%     bbox:           N x 4 (TLBR format)
function bbox = get_centered_bbox(pts_array, bbox_width, bbox_height, debug_mode)
	if nargin < 4
		debug_mode = true;
	end
    
    if debug_mode
        assert(is2dPtsArray(pts_array) || is2dPtsArray_occlusion(pts_array), sprintf('the input points should have shape: 2 or 3 x num_pts vs %d x %d', size(pts_array, 1), size(pts_array, 2)));
	end

    xmin = pts_array(1, :) - ceil(bbox_width / 2.0) + 1;    
    ymin = pts_array(2, :) - ceil(bbox_height / 2.0) + 1;
    xmax = xmin + bbox_width - 1;
    ymax = ymin + bbox_height - 1;
    
    bbox = [xmin; ymin; xmax; ymax];
    bbox = bbox';
end