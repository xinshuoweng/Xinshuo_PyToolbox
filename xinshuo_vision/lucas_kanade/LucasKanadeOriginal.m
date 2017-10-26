function [u,v] = LucasKanadeOriginal(It, It1, rect)
% input - image at time t, image at t+1, rectangle (top left, bot right coordinates)
% output - movement vector, [u,v] in the x- and y-directions.

% initialization
iter = 1000;
u = 0;
v = 0;
old_u = 0;
old_v = 0;
height_kernel = round(rect(4) - rect(2) + 1);
width_kernel = round(rect(3) - rect(1) + 1);
% number_pixel = height_kernel * width_kernel;
[Ix, Iy] = gradient(It1);

A = zeros(height_kernel*width_kernel,2);
% b = zeros(2,1);
height = size(It, 1);
width = size(It, 2);
X = 1:width;
Y = 1:height;
[X, Y] = meshgrid(X, Y);

x = (rect(1) - old_u):1:(rect(3) - old_u);
y = (rect(2) - old_v):1:(rect(4) - old_v);
if round(length(x)) < width_kernel
    x = [x, rect(3) - old_u];
elseif round(length(x)) > width_kernel
    x = x(1:end-1);
end
if round(length(y)) < height_kernel
    y = [y, rect(4) - old_v];
elseif round(length(y)) > height_kernel
    y = y(1:end-1);
end
[x_temp, y_temp] = meshgrid(x, y);
Intensity_original = interp2(X, Y, It, x_temp, y_temp, 'spline');
% loop till converge
for i = 1:iter
    % go through all pixels within the box
    x = (rect(1) - old_u):1:(rect(3) - old_u);
    y = (rect(2) - old_v):1:(rect(4) - old_v);
    if round(length(x)) < width_kernel
        x = [x, rect(3) - old_u];
    elseif round(length(x)) > width_kernel
        x = x(1:end-1);
    end
    if round(length(y)) < height_kernel
        y = [y, rect(4) - old_v];
    elseif round(length(y)) > height_kernel
        y = y(1:end-1);
    end
    
    [x_temp, y_temp] = meshgrid(x, y);
    % %     index = [X' Y'];
    %     index(:, 1) = X(:);
    %     index(:, 2) = Y(:);
    Ix_temp = interp2(X, Y, Ix, x_temp, y_temp, 'spline');
    Iy_temp = interp2(X, Y, Iy, x_temp, y_temp, 'spline');
    gradient_T_temp = interp2(X, Y, It1, x_temp, y_temp, 'spline') - Intensity_original;
    
    A(:,1) = Ix_temp(:);
    A(:,2) = Iy_temp(:);
    b = gradient_T_temp(:);
    V = (A'*A)\(A'*b);
    u = old_u + V(1);
    v = old_v + V(2);
    
    % check the change is below a threshold
    distance = abs(old_u-u) + abs(old_v-v);
    if distance < 0.5
        break;
    end
    
    old_u = u;
    old_v = v;
end
%
% u = old_u;
% v = old_v;

end