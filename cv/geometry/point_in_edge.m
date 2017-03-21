
% TODO: CHECK
function [inter] = point_in_edge(point, line_a, line_b)
% this function return if the given point is in the line segment, including
% the endpoint

tol = 0.0000001;
vector1 = point - line_a;
vector2 = point - line_b;
norm1 = sqrt(sum(vector1.^2));
norm2 = sqrt(sum(vector2.^2));
if norm1 == 0 || norm2 == 0
    inter = 1;
else
    costheta = vector1*vector2' / (norm1 * norm2);
    if abs(costheta + 1) < tol
        inter = 1;
    else
        inter = 0;
    end
end

end