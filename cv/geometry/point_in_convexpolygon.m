function inter = point_in_convexpolygon(point, polygon)
% this function compute if the given point is in the polygon (not containing the boundary) or not.

points_all = [point; polygon];
convex_hull = find_convex_hull(points_all);
if isequal(convex_hull, polygon) == 1   % if convex hull is same as previous one, the point is inside or along the edge.
    inter = 1;
    
    % check if the point is along the boundary
    edge_list = find_edge_polygon(polygon);
    for i = 1:size(edge_list, 1)
        if point_in_edge(point, edge_list(i, 1:2), edge_list(i, 3:4)) == 1
            inter = 0;      % if it's along one of the edge, then it's not inside the polygon
            break;
        end
    end
else
    inter = 0;
end

end