% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function opens an image and let user to select several points
% the input image could be a matrix or a path to the image
% this function returns a matrix which has Nx2 dimension
% each row is [x, y]
function pts = get_pts_from_image(img, number_pts)
    if ischar(img)
        img = imread(img);
    else
        assert(isimage(img), 'The input image doesn''t ...
            have a good dimentsion while getting the point.');
    end

    if ~exist('number_pts', 'var')
        number_pts = 1;
    else
        assert(isPositiveInteger(number_pts), 'The number of points ...
            should be positive integer while getting the point from image.');
    figure;
    imshow(im);
    pts = ginput(number_pts);
end