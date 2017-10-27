% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% project the 3d point to 2d
% input
%		pts_3d			3 x num_pts
%		M 				3 x 4 projection matrix
%
% output	
%		pts_2d			num_pts x 2
function pts_2d = projection_from_pts(pts_3d, M, debug_mode)
    if nargin < 3
        debug_mode = true;
    end

    if debug_mode
    	assert(is3dPtsArray(pts_3d), 'the input 3D points do not have a good shape');
    	assert(all(size(M) == [3, 4]), 'the input projection matrix does not have a good shape');
    end

    num_pts = size(pts_3d, 2);
    pts_3d_homo = [pts_3d; ones(1, num_pts)];			% 4 x num_pts
    pts_2d_homo = M * pts_3d_homo;						% 3 X num_pts
    for pts_index = 1:num_pts
    	pts_2d_homo(:, pts_index) = pts_2d_homo(:, pts_index) ./ pts_2d_homo(end, pts_index);
    end

    pts_2d = pts_2d_homo(1:2, :);						% 2 x num_pts
end