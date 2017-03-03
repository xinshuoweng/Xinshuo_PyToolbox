% a, b, c is respectively three points representing the triangle
% pts, a, b, c are all 3d points here

% math:
%       t * b.X + u * c.X + v * n.X = P.X
%       t * b.Y + u * c.Y + v * n.Y = P.Y
%       t * b.Z + u * c.Z + v * n.Z = P.Z
% condition:
%       0 <= t <= 1
%       0 <= u <= 1
%       t + u <= 1      
function [bool, res] = point_inside_3d_triangle_test(pts, tri_A, tri_B, tri_C)
    assert(length(pts) == 3, 'The input point should be 3d point');
    assert(length(tri_A) == 3, 'The input point should be 3d point');
    assert(length(tri_B) == 3, 'The input point should be 3d point');
    assert(length(tri_C) == 3, 'The input point should be 3d point');

    vec_b = tri_B - tri_A;
    vec_c = tri_C - tri_A;
    n = cross(vec_b, vec_c);
    pts = pts - tri_A;
    if size(vec_b, 1) ~= 3
        vec_b = vec_b';
    end
    if size(vec_c, 1) ~= 3
        vec_c = vec_c';
    end
    if size(n, 1) ~= 3
        n = n';
    end
    if size(pts, 1) ~= 3
        pts = pts';
    end
    
    A = [vec_b, vec_c, n/norm(n)];
    b = pts;
    solution = (A'*A) \ (A' * b);
%     solution = inv(A) * b;
    t = solution(1);
    u = solution(2);
    v = solution(3);

    if t >=0 && t <= 1 && u >=0 && u <= 1 && (t + u <= 1)
        bool = 1;
    else
        bool = 0;
    end
    res = [t, u, v];
end