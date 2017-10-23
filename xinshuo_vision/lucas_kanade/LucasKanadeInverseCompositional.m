function [u,v] = LucasKanadeInverseCompositional(It, It1, rect)
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
[Ix, Iy] = gradient(It);            % evaluate the gradient of template

A = zeros(height_kernel*width_kernel,2);
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

% precompute the hessian matrix
[x_temp, y_temp] = meshgrid(x, y);
Ix_temp = interp2(X, Y, Ix, x_temp, y_temp, 'linear');
Iy_temp = interp2(X, Y, Iy, x_temp, y_temp, 'linear');
A(:, 1) = Ix_temp(:);
A(:, 2) = Iy_temp(:);

J = [1, 0; 0, 1]        % Jacobian is the identity matrix
H = (A'*A)\A';          % compute the hessian matrix
Intensity_original = interp2(X, Y, It, x_temp, y_temp, 'linear');

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
    % warp the rect in the incoming frame back to the original template and compute the brightness 
    gradient_T_temp = interp2(X, Y, It1, x_temp, y_temp, 'linear') - Intensity_original;            % compute the residual
    b = gradient_T_temp(:);
    V = H*b;                    % compute the delta warp function
    
    u = old_u + V(1,1);
    v = old_v + V(2,1);
    
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
