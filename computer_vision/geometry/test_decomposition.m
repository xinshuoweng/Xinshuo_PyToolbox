disp("Picking random Euler angles (radians)");
% 
% 	x = 2*pi*rand() - pi; % -180 to 180
% 	y = pi*rand() - pi*0.5; % -90 to 90
% 	z = 2*pi*rand() - pi; % -180 to 180
% 
% 	disp("\nRotation matrix is:");
% 	R = compose_rotation(x,y,z);
R = [0.00409992,-0.0664568,-0.0739465;
     -0.0991034,0.00318163,-0.0083541;
     0.00794384,0.0739917,-0.066057];

disp("Decomposing R");
[x2,y2,z2] = decompose_rotation(R)
x2_angle = x2 / pi * 180
y2_angle = y2 / pi * 180
z2_angle = z2 / pi * 180

disp("");

R_back = compose_rotation(x2, y2, z2)
% 	err = sqrt((x2-x)*(x2-x) + (y2-y)*(y2-y) + (z2-z)*(z2-z))

% 	if err < 1e-5
% 		disp("Results are correct!");
% 	else
% 		disp("Oops wrong results :(");
% 	end

function [x,y,z] = decompose_rotation(R)
	x = atan2(R(3,2), R(3,3));
	y = atan2(-R(3,1), sqrt(R(3,2)*R(3,2) + R(3,3)*R(3,3)));
	z = atan2(R(2,1), R(1,1));
end

function R = compose_rotation(x, y, z)
	X = eye(3,3);
	Y = eye(3,3);
	Z = eye(3,3);

    X(2,2) = cos(x);
    X(2,3) = -sin(x);
    X(3,2) = sin(x);
    X(3,3) = cos(x);

    Y(1,1) = cos(y);
    Y(1,3) = sin(y);
    Y(3,1) = -sin(y);
    Y(3,3) = cos(y);

    Z(1,1) = cos(z);
    Z(1,2) = -sin(z);
    Z(2,1) = sin(z);
    Z(2,2) = cos(z);

	R = Z*Y*X;
end
