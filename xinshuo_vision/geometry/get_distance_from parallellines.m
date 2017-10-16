% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% TODO: CHECK
function distance = distance_two_parallellines(line1_a, line1_b, line2_a, line2_b)
% this function return the distance of two parallel lines

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
    
if vertical1 == 1 && vertical2 == 1     % if both lines are vertical to x axis
    distance = abs(line1_a(1) - line2_a(1));
else
    distance = abs(a2*b1 - a1*b2)/sqrt((a1*a2)^2 + a1^2);
end

end