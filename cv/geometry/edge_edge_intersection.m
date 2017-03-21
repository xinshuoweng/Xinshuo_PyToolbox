% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% TODO: CHECK
function [point, intersect] = edge_edge_intersection(line1_a, line1_b, line2_a, line2_b)
% this function test if two line segment will intersect with each other.
% The output point will return the intersection point if existing,
% otherwise empty. The intersect is a boolean variable to denote if there is an
% intersection

% in general the point should return a single point. But if two lines are
% parallel and also intersect, then the output should be two points, which
% denote an overlapping line segment.

% for line segment 1
x1 = line1_a(1);
y1 = line1_a(2);
x2 = line1_b(1);
y2 = line1_b(2);
vertical1 = 0;
if x1 ~= x2
    a1 = (y1-y2)/(x1-x2);
    b1 = y1 - a1*x1;
else
    vertical1 = 1;
end

% for line segment 2
x1 = line2_a(1);
y1 = line2_a(2);
x2 = line2_b(1);
y2 = line2_b(2);
vertical2 = 0;
if x1 ~= x2
    a2 = (y1-y2)/(x1-x2);
    b2 = y1 - a2*x1;
else
    vertical2 = 1;
end

parallel = 0;
if vertical1 == 0 && vertical2 == 0     % both line segments are not vertical
    if a1 == a2     % two line segments are parallel
        parallel = 1;
    else
        x = (b2-b1)/(a1-a2);
        y = a1*x + b1;
        point = [x, y];
    end
elseif vertical1 == 1 && vertical2 == 1 % both line segments are vertical
    parallel = 1;
elseif vertical1 == 0 && vertical2 == 1 % segment 2 is vertical
    x = line2_a(1);
    y = a1*x + b1;
    point = [x, y];
else                                    % segment 1 is vertical
    x = line1_a(1);
    y = a2*x + b2;    
    point = [x, y];        
end

if parallel == 1
    distance = distance_two_parallellines(line1_a, line1_b, line2_a, line2_b); 
    if distance ~= 0                    % if two parallel lines have a distance
        intersect = 0;
        point = [];
    else
        inter1 = point_in_edge(line2_a, line1_a, line1_b);   
        inter2 = point_in_edge(line2_b, line1_a, line1_b);   
        if inter1 == 1 || inter2 == 1   % two parallel segments overlap
            points = [line1_a; line1_b; line2_a; line2_b];
            [~, idmin] = min(points(:, 1));
            [~, idmax] = max(points(:, 1));    
            points([idmin, idmax], :) = [];
            point = unique(points, 'rows');
            intersect = 1;
        else
            intersect = 0;              % two parallel segments don't overlap
            point = [];            
        end
    end
else                                    % test if the point is in the line segment
    inter1 = point_in_edge(point, line1_a, line1_b);   
    inter2 = point_in_edge(point, line2_a, line2_b);
    if inter1 == 1 && inter2 == 1       % if the intersection point is lie in one of two segments
        intersect = 1;
    else
        intersect = 0;
        point = [];
    end
end
        
end