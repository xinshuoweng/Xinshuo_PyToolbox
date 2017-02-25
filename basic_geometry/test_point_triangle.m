close all;
clc;
clear;

tri_c = [1, 1, 0];
tri_b = [1, -1, 0];
tri_a = [1, 0, 1];
% pts = tri_a;
pts = [1, 0.5, 0.5];
[res, pts_new] = point_inside_3d_triangle_test(pts, tri_a, tri_b, tri_c);
if res == 1
    disp('the point is in the triangle');
else
    disp('the point is not in the triangle');
end

pts_new