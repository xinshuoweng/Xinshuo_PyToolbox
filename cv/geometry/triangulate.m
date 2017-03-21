% TODO: CHECK
function [ P, error ] = triangulate( M1, p1, M2, p2 )
% triangulate:
%       M1 - 3x4 Camera Matrix 1
%       p1 - Nx2 set of points
%       M2 - 3x4 Camera Matrix 2
%       p2 - Nx2 set of points

% Q2.4 - Todo:
%       Implement a triangulation algorithm to compute the 3d locations
%       See Szeliski Chapter 7 for ideas
%

% initialization
number_points = size(p1, 1);

% least square
p1T1 = M1(1, :);
p1T2 = M1(2, :);
p1T3 = M1(3, :);
p2T1 = M2(1, :);
p2T2 = M2(2, :);
p2T3 = M2(3, :);
% A = [M1; M2];
% H = (A'*A)\A';
P = zeros(number_points, 4);
error = 0;
for i = 1:number_points
    U(1, :) = p1(i, 2).* p1T3 - p1T2;
    U(2, :) = p1T1 - p1(i, 1).*p1T3;
    U(3, :) = p1(i, 1).*p1T2 - p1(i, 2).*p1T1;
    U(4, :) = p2(i, 2).* p2T3 - p2T2;
    U(5, :) = p2T1 - p2(i, 1).*p2T3;
    U(6, :) = p2(i, 1).*p2T2 - p2(i, 2).*p2T1;
    [~, ~, V] = svd(U);
    P(i, :) = V(:, end)';
    
%     b = [p1(i, :)'; 1; p2(i, :)'; 1];    
%     P(i, :) = (H * b)';
    P(i, :) = P(i, :)./P(i, 4);

    % compute reprojection error
    p1_proj(i, :) = (M1 * P(i, :)')';
    p2_proj(i, :) = (M2 * P(i, :)')';
    p1_proj(i, :) = p1_proj(i, :) ./ p1_proj(i, 3);
    p2_proj(i, :) = p2_proj(i, :) ./ p2_proj(i, 3);
    error = error + norm(p1_proj(i, 1:2) - p1(i, :)) + norm(p2_proj(i, 1:2) - p2(i, :));
end

P = P(:, 1:3);
end

