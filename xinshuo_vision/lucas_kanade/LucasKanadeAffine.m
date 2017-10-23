function M = LucasKanadeAffine(It, It1)
% input - image at time t, image at t+1, rectangle (top left, bot right coordinates)
% output - movement vector, [u,v] in the x- and y-directions.

% initialization
iter = 100;

% height_kernel = round(rect(4) - rect(2) + 1);
% width_kernel = round(rect(3) - rect(1) + 1);
% number_pixel = height_kernel * width_kernel;
% [Ix, Iy] = gradient(It1);

M = eye(3);
% b = zeros(2,1);
height = size(It, 1);
width = size(It, 2);
A = zeros(height*width,6);
X = 1:width;
Y = 1:height;
[X, Y] = meshgrid(X, Y);
[Ix, Iy] = gradient(It);

A(:, 1) = Ix(:).*X(:);
A(:, 2) = Ix(:).*Y(:);
A(:, 3) = Ix(:);
A(:, 4) = Iy(:).*X(:);
A(:, 5) = Iy(:).*Y(:);
A(:, 6) = Iy(:);
H = (A'*A)\A';

% loop till converge
for i = 1:iter
    %     It1 = warpH(It1, M, size(It1));
    It1 = warpH(It1, M, size(It1));
    gradient_T_temp = It1 - It;
    
    b = gradient_T_temp(:);
    
    V = H*b;
    
    M(1, 1) = V(1) + 1;
    M(1, 2) = V(2);
    M(1, 3) = V(3);
    M(2, 1) = V(4);
    M(2, 2) = V(5) + 1;
    M(2, 3) = V(6);
    
    
    % check the change is below a threshold
    distance = sum(sum(abs(M - eye(3))));
    if distance < 0.5
        break;
    end
    
end

end