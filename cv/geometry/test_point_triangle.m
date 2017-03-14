close all;
clc;
clear;

tri_a = [-19.3038024902344, -36.8296813964844, 843.675170898438];
tri_b = [-19.3038024902344, -37.0751953125, 843.599304199219];
tri_c = [-19.5493316650391, -36.8296813964844, 843.555541992188];
% pts = tri_a;
pts = [-19.4072430557542, 35.9863537872186, 866.12578509405];
[res, pts_new] = point_inside_3d_triangle_test(pts, tri_a, tri_b, tri_c);
if res == 1
    disp('the point is in the triangle');
else
    disp('the point is not in the triangle');
end

pts_new