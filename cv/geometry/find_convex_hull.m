% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% TODO: CHECK
function convex = find_convex_hull(points)
% input are a set of points, which has a dimention Nx2, N is number of
% points, each points has coordinate (x, y)

points = unique(points, 'rows');
N = size(points, 1);

% find the point with the lowest y coordinate and place it in the first of
% the array
[~, lowest_id] = min(points(:, 2));     
points = swap(points, lowest_id, 1);

% mark the starting point
plot(points(1, 1), points(1, 2), '*', 'MarkerFaceColor', 'r');

% sort the points based on the angle referred to the first point
angle = compute_angle(repmat(points(1, :), [N-1 1]), points(2:N, :));
[~, id] = sort(angle, 'descend');
points_waitsort = points(2:end, :);
points_sorted = points_waitsort(id, :);
points(2:end, :) = points_sorted;

M = 2;
count = 0;  % record the number of points eliminated from the set
for i = 3:N
    point_temp = points(i-count, :);
    while counter_clockwise_turn(points(M-1, :), points(M, :), point_temp) <= 0
        % if it's a right turn, eliminate the middle points from the convex
        % set
        points(M, :) = [];
        count = count + 1;
        if M > 2
            M = M - 1; 
        else
            disp('error');
            keyboard;
        end
    end
    
%     points = swap(points, M, i);    
    M = M + 1;
end

convex = points;
end

function array = swap(array, i, j)
% input is a matrix, i and j are the y index, this function swap the row i
% and row j in the matrix array and return the swaped one

temp = array(i, :);
array(i, :) = array(j, :);
array(j, :) = temp;

end

function result = counter_clockwise_turn(p1, p2, p3)
% input are three points, this function compute if it's left turn or right
% turn, if result > 0, it's counter clockwise, otherwise, it's clockwise

result = (p2(1) - p1(1)) * (p3(2) - p1(2)) - (p2(2) - p1(2)) * (p3(1) - p1(1));

end

function cosine = compute_angle(p1, p2)
% input are point array, the output returns an array containing the angle
% referred to the point in the first input array
vector = p2 - p1;
vector_reference = [1, 0];
norm_vector = sqrt(sum(vector.^2, 2));
cosine = vector * vector_reference' ./ norm_vector;
end