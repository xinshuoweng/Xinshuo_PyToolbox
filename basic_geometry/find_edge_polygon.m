function [ edge_list ] = find_edge_polygon( polygon )
% this function finds all edge from given polygon, the input polygon is a
% Nx2 matrix, which contains N points. Each of points contains x and y
% coordinate. The output edge list contains a Nx4 matrix. In each row of
% the outout, it contains [x1, y1, x2, y2]

assert(size(polygon, 1) >= 3, 'dimension of polygon is not enough');
number_point = length(polygon);
edge_list = zeros(number_point, 4);
for i = 1:size(edge_list, 1)-1
    edge_list(i, :) = [polygon(i, :), polygon(i+1, :)];
end
edge_list(end, :) = [polygon(end, :), polygon(1, :)];


end

