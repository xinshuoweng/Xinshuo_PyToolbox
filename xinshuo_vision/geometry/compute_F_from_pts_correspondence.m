% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

%   pts1 - Nx2 matrix of (x,y) coordinates
%   pts2 - Nx2 matrix of (x,y) coordinates
%   normalize_factor    - max (imwidth, imheight)
%   method              - 7 point or 8 point method
%   max_iter            - maximum iteration for RANSAC
%   err_threshold       - tolerance for RANSAC
function [F, inlier_index] = compute_F_from_pts_correspondence(pts1, pts2, normalize_factor, debug_mode, method, max_iter, err_threshold)
    if nargin < 4
        debug_mode = true;
    end

    if nargin < 5
        method = '8pts';
    else
        assert(strcmp(method, '8pts') || strcmp(method, '7pts'), 'the method is 8pts or 7pts');
    end

    if nargin < 6
        max_iter = 100;
    end

    if nargin < 7
        err_threshold = 0.0002;
    end

    if debug_mode
        assert(all(size(pts1) == size(pts2)), 'the input point correspondence is not good');
        assert(size(pts1, 2) == 2 && size(pts1, 1) > 0 && length(size(pts1)) == 2, 'the input point does not have a good shape');
    end

    % initialization
    best_num_inlier = 0;
    num_pts = size(pts1, 1);
    pts1_cat = [pts1, ones(num_pts, 1)];
    pts2_cat = [pts2, ones(num_pts, 1)];

    % ransac
    for iter_index = 1:max_iter
        % randomly select 7 points pair and compute fundamental matrix
        index = randperm(num_pts, 7);
        
        % randomly select the seven correspondence points pair
        pts1_temp = pts1(index, :);
        pts2_temp = pts2(index, :);
        
        fprintf('iteration %d\n', iter_index); 
        if strcmp(method, '8pts')
            F = compute_F_from_8pts(pts1_temp, pts2_temp, normalize_factor, debug_mode);
        elseif strcmp(method, '7pts')
            F = compute_F_from_7pts(pts1_temp, pts2_temp, normalize_factor, debug_mode);
        else
            assert(false, 'error');
        end

        % calculate number of inliers
        inlier_check = abs(diag(pts1_cat * F * pts2_cat'));
        inlier_index = inlier_check < err_threshold;
        num_inlier = sum(inlier_index);
        if num_inlier > best_num_inlier
            fprintf('\bnumber of inlier is %d\n', num_inlier);
            best_num_inlier = num_inlier;
            best_inlier = inlier_index;
        end
        if num_inlier > 0.7 * num_pts
            break;
        end
    end

    inliers_pts1 = pts1(best_inlier, :);
    inliers_pts2 = pts2(best_inlier, :);
    F = compute_F_from_8pts(inliers_pts1, inliers_pts2, normalize_factor, debug_mode);

    inlier_index = best_inlier;
end