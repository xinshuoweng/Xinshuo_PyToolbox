% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% triangulation given two projection matrix and point correspondence
%       M1 - 3x4 Camera Matrix 1
%       p1 - Nx2 set of points
%       M2 - 3x4 Camera Matrix 2
%       p2 - Nx2 set of points
function [P, err] = triangulate(pts1, pts2, M1, M2, debug_mode)
    if nargin < 5
        debug_mode = true;
    end

    if debug_mode
        assert(all(size(pts1) == size(pts2)), 'the size of input point correspondence is not equal');
        assert(size(pts1, 2) == 2 && size(pts1, 1) > 0 && length(size(pts1)) == 2, 'the input point does not have a good shape');

        assert(all(size(M1) == [3, 4]), 'the input projection matrix 1 does not have a good shape');
        assert(all(size(M2) == [3, 4]), 'the input projection matrix 2 does not have a good shape');
    end

    % initialization
    num_pts = size(pts1, 1);

    % least square
    p1T1 = M1(1, :);
    p1T2 = M1(2, :);
    p1T3 = M1(3, :);
    p2T1 = M2(1, :);
    p2T2 = M2(2, :);
    p2T3 = M2(3, :);
    % A = [M1; M2];
    % H = (A'*A)\A';
    P = zeros(num_pts, 4);
    err = 0;
    for i = 1:num_pts
        U(1, :) = pts1(i, 2).* p1T3 - p1T2;
        U(2, :) = p1T1 - pts1(i, 1).*p1T3;
        U(3, :) = pts1(i, 1).*p1T2 - pts1(i, 2).*p1T1;
        U(4, :) = pts2(i, 2).* p2T3 - p2T2;
        U(5, :) = p2T1 - pts2(i, 1).*p2T3;
        U(6, :) = pts2(i, 1).*p2T2 - pts2(i, 2).*p2T1;
        
        [~, ~, V] = svd(U);
        P(i, :) = V(:, end)';
        P(i, :) = P(i, :) ./ P(i, 4);

        % compute reprojection error
        p1_proj(i, :) = (M1 * P(i, :)')';
        p2_proj(i, :) = (M2 * P(i, :)')';
        p1_proj(i, :) = p1_proj(i, :) ./ p1_proj(i, 3);
        p2_proj(i, :) = p2_proj(i, :) ./ p2_proj(i, 3);
        err = err + norm(p1_proj(i, 1:2) - pts1(i, :)) + norm(p2_proj(i, 1:2) - pts2(i, :));
    end

    P = P(:, 1:3);
end

