function [u,v, lambda] = LucasKanadeBasisPart2(It, It1, rect, bases)

% input - image at time t, image at t+1, rectangle (top left, bot right
% coordinates), bases
% output - movement vector, [u,v] in the x- and y-directions.
% initialization


% idea 1: inside basis, initial optimal p first and then add bases
% idea 2: add affine function first like inverse compositional
iter = 100;
u = 0;
v = 0;
old_u = 0;
old_v = 0;
number_basis = size(bases, 3);
old_lambda = zeros(number_basis, 1);
lambda = zeros(number_basis, 1);
height_kernel = round(rect(4) - rect(2) + 1);
width_kernel = round(rect(3) - rect(1) + 1);
[Ix, Iy] = gradient(It);

A = zeros(size(bases, 1)*size(bases, 2), number_basis+2);
% b = zeros(size(bases, 1)*size(bases, 2),1);
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
Ix_temp = interp2(X, Y, Ix, x_temp, y_temp, 'linear');
Iy_temp = interp2(X, Y, Iy, x_temp, y_temp, 'linear');
A(:, 1) = Ix_temp(:);
A(:, 2) = Iy_temp(:);

for j = 1:number_basis
    base_temp = squeeze(bases(:,:,j));
    if size(base_temp) ~= size(Ix_temp)
        keyboard;
    end
    A(:,j+2) = (-1).*base_temp(:);
end
% H = (A'*A)\A';
A1 = A(:,1:2);
% H1 = (A1'*A1)\A1';

Intensity_original = interp2(X, Y, It, x_temp, y_temp, 'linear');


summation = zeros(size(bases, 1)*size(bases, 2), 2);
for k = 1:number_basis
    %         gradient_T_temp = gradient_T_temp + old_lambda(j) .* squeeze(bases(:, :, j));    % minus!!!!!!!
    base_temp = squeeze(bases(:,:,k));
    base_temp = base_temp(:);
    temp = base_temp'*A1;
    summation = summation + [temp(1).*base_temp, temp(2).*base_temp];
end
SD = A1 + summation;
Hq = (SD'*SD)\SD';


% loop till converge
for i = 1:iter
    % go through all pixels within the box
    
    % stage1, update the u and v
    [x_temp, y_temp] = meshgrid(x, y);
    gradient_T_temp = interp2(X, Y, It1, x_temp, y_temp, 'linear') - Intensity_original;
    for j = 1:number_basis
            gradient_T_temp = gradient_T_temp - old_lambda(j) .* squeeze(bases(:, :, j));    % minus!!!!!!!
    end
    b = gradient_T_temp(:);
    V = Hq*b;
    
    u = old_u + V(1, 1);
    v = old_v + V(2, 1);
    
    
    % stage 2
    x = (rect(1) - u):1:(rect(3) - u);
    y = (rect(2) - v):1:(rect(4) - v);
    if round(length(x)) < width_kernel
        x = [x, rect(3) - u];
    elseif round(length(x)) > width_kernel
        x = x(1:end-1);
    end
    if round(length(y)) < height_kernel
        y = [y, rect(4) - v];
    elseif round(length(y)) > height_kernel
        y = y(1:end-1);
    end
    
    [x_temp, y_temp] = meshgrid(x, y);
    gradient_T_temp = interp2(X, Y, It1, x_temp, y_temp, 'linear') - Intensity_original;
    for j = 1:number_basis
        gradient_T_temp = gradient_T_temp - old_lambda(j) .* squeeze(bases(:, :, j));    % minus!!!!!!!
    end
    b = gradient_T_temp(:);
%         V = H*b;
    lambda = A(:,3:end)'*b;
    
    % check the change is below a threshold
    distance = abs(old_u-u) + abs(old_v-v);
    if distance < 1
        break;
    end
    
    old_lambda = lambda;
    old_u = u;
    old_v = v;
end
%
% u = old_u;
% v = old_v;

end