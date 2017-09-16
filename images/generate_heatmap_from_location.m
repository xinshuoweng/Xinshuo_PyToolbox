% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function generate heatmap based on point locations
% the gaussian blob is centered at the points provided
function heatmap = generate_heatmap_from_location(im_size, pts, debug_mode, vis) %this function is only for center map in testing
    if nargin < 4
        vis = false;
        if nargin < 3
            debug_mode = true;
        end
    end

    if debug_mode
    	assert(numel(im_size) == 2 || numel(im_size) == 3, 'image shape is not correct.');
    	assert(is2dpts(pts), 'provided 2d points is not correct.')
    end

    sigma_ = 21;		% hyper-parameter

    [X,Y] = meshgrid(1:im_size(1), 1:im_size(2));
    x = pts(1);
    y = pts(2);
    X = X - x;
    Y = Y - y;
    D2 = X.^2 + Y.^2;
    Exponent = D2 ./ 2.0 ./ sigma_ ./ sigma_;
    heatmap{1} = exp(-Exponent);
end