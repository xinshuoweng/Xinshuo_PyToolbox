% TODO: CHECK

function F = sevenpoint(pts1, pts2, M)
% sevenpoint:
%   pts1 - Nx2 matrix of (x,y) coordinates
%   pts2 - Nx2 matrix of (x,y) coordinates
%   M    - max (imwidth, imheight)

% Q2.2 - Todo:
%     Implement the eightpoint algorithm
%     Generate a matrix F from some '../data/some_corresp.mat'
%     Save recovered F (either 1 or 3 in cell), M, pts1, pts2 to q2_2.mat

%     Write recovered F and display the output of displayEpipolarF in your writeup

% normlize the coordinate
pts1_norm = pts1 / M;
pts2_norm = pts2 / M;

% construct the U matrix
U(:, [1, 5]) = pts1_norm.*pts2_norm;
U(:, 2) = pts1_norm(:, 2).*pts2_norm(:, 1);
U(:, 3) = pts2_norm(:, 1);
U(:, 4) = pts1_norm(:, 1).*pts2_norm(:, 2);
U(:, 6) = pts2_norm(:, 2);
U(:, 7) = pts1_norm(:, 1);
U(:, 8) = pts1_norm(:, 2);
U(:, 9) = ones(7, 1);

% solve the homogenuous least square system Uf = 0
[E, S, V] = svd(U);
f1 = V(:, end);
f2 = V(:, end-1);
F1 = reshape(f1, 3, 3);
F2 = reshape(f2, 3, 3);

syms lambda
func = symfun(det((1-lambda)*F1 + lambda*F2), lambda);
res = double(root(func, lambda));
F_check = (1-res(1))*F1 + res(1)*F2;
fprintf('check the determinant of (1 - lambda)*F1 + lambda*F2 = %f\n', abs(det(F_check)));
% for i = 1:length(res)
%     if isreal(res(i))
%         continue;
%     else
%         res(i) = [];    
%     end
% end

% ensure the singularity of F
[W, S, V] = svd(F_check);
S(3, 3) = 0;
F = W*S*V';

% refine the solution
F_refine = refineF(F, pts1_norm, pts2_norm);

% unscaling the F
T = [1/M, 0, 0; 0, 1/M, 0; 0, 0, 1];
F = T'*F_refine*T;
% F = F_refine;

end

