% Author: Xinshuo Weng
% Email: xinshuo.weng@gmail.com

% TODO: check

% this function takes a 4 x 4 transformation as input and decompose the rotation angle, translation and scale in x, y, z axis
% the transformation matrix is restricted to be affine transformation without shearing
function [rotation_vec, translation_vec, scale_vec] = transformation_decomposition(trans, debug_mode)
    if ~exist('debug_mode', 'var')
        debug_mode = true;
    end


    if debug_mode
        assert(all(size(trans) == [4, 4]), 'the dimension of transformation input matrix should be 4 x 4\n');
        assert(all(trans(end, 1:end) == [0, 0, 0, 1]), 'the last row of transformation matrix is not [0, 0, 0, 1]\n');
    end

    rotation_matrix = trans(1:3, 1:3);
    translation_vec = trans(1:3, end);

    % decompose rotation
    [rot_x, rot_y, rot_z] = decompose_rotation(rotation_matrix);
    rotation_matrix_back = compose_rotation(rot_x, rot_y, rot_z);
    scale_vec = [rotation_matrix_back(1, 1) / rotation_matrix(1, 1), rotation_matrix_back(2, 2) / rotation_matrix(2, 2), rotation_matrix_back(3, 3) / rotation_matrix(3, 3)];

    rotation_vec = [rot_x, rot_y, rot_z];
    rotation_vec = rotation_vec ./ pi .* 180;       % convert radian to degree
end

% decompose the rotation matrix to radian 
function [x,y,z] = decompose_rotation(R)
	x = atan2(R(3,2), R(3,3));
	y = atan2(-R(3,1), sqrt(R(3,2)*R(3,2) + R(3,3)*R(3,3)));
	z = atan2(R(2,1), R(1,1));
end

% construct rotation matrix from radian 
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
