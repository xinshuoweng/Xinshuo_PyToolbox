% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% input - image at time t, image at t+1, rectangle (top left, bot right coordinates)
% output - movement vector, [u,v] in the x- and y-directions.
function [u, v, num_iter] = LucasKanadeInverseCompositional_batch(It, It1, rect, input_format, max_iter, weight, epsilon, debug_mode)
    if nargin < 8
        debug_mode = true;
    end

    if nargin < 7
        epsilon = 0.03;
    end

    if nargin < 6
        weight = true;
    end

    if nargin < 5
        max_iter = 1000;
    end

    if nargin < 4
        input_format = 'hw';
    else
        assert(strcmp(input_format, 'hwc') || strcmp(input_format, 'chw') || strcmp(input_format, 'hw'), 'the input format is not correct');
    end

    if debug_mode
        assert(all(size(It) == size(It1)), 'the shape of two input images should be the same');
    end

    % convert grayscale to 2d image
    It = squeeze(It);
    It1 = squeeze(It1);

    % compute the warped image
    % cropped_template = crop_interp(It, input_format, rect);                    % C x height_kernel x width_kernel
    cropped_template = crop_interp_batch(It, input_format, rect);                % num_pts x C x height_kernel x width_kernel
    
    % rect
    % rect(1, :)
    % rect(2, :)
    % squeeze(cropped_template(1, 1, :, :))
    % squeeze(cropped_template(2, 1, :, :))

    % compute the Jacobian
    

    % cropped_template = permute(cropped_template, [3, 4, 1, 2]);
    % cropped_template = squeeze(cropped_template);
    % [Iy, ~, Ix] = gradient(cropped_template);                                   % C x height_kernel x width_kernel

    [~, ~, Iy, Ix] = gradient(cropped_template);                                % num_pts x C x height_kernel x width_kernel,       note that the gradient along the edge is different in MATLAB and python, but same after adding weight

    % squeeze(Ix(1, :, :))
    % squeeze(Iy(1, :, :))

    % squeeze(Ix(2, 1, :, :))
    % squeeze(Iy(2, 1, :, :))
    % squeeze(Iy(2, 2, :, :))
    % squeeze(Ia(1, 1, :, :))
    
    % squeeze(Ib(1, 1, :, :))

    % squeeze(Iy(:, :, 1, 1))
    % squeeze(Ia(:, :, 1, 1))
    % squeeze(Ix(:, :, 1, 1))
    % squeeze(Ib(:, :, 1, 1))
    % pause;

    width_kernel = size(Ix, 4);
    height_kernel = size(Ix, 3);
    im_channel = size(Ix, 2);
    num_pts = size(Ix, 1)
    weight_map = generate_weight([height_kernel, width_kernel]);                % height_kernel x width_kernel
    % weight_map
    % pause;

    J = zeros(num_pts, 2, im_channel, height_kernel, width_kernel);                                                    % num_pts x 2 x C x height_kernel x width_kernel
    % size(J(:, 1, :, :, :))
    % size(reshape(Ix, [num_pts, 1, im_channel, height_kernel, width_kernel]))
    % J(:, 1, :, :, :) = reshape(Ix, [num_pts, 1, im_channel, height_kernel, width_kernel]);            
    J(:, 1, :, :, :) = Ix;
    J(:, 2, :, :, :) = Iy;


    % squeeze(J(1, 1, :, :))
    if weight
        % reshape_tmp_J = reshape(J, 2, im_channel, height_kernel * width_kernel);
        weightedJ = zeros(num_pts, 2, im_channel, height_kernel, width_kernel);
        for pts_index = 1:num_pts
            for channel_index = 1:im_channel
                weightedJ(pts_index, 1, channel_index, :, :) = squeeze(J(pts_index, 1, channel_index, :, :)) .* weight_map;
                weightedJ(pts_index, 2, channel_index, :, :) = squeeze(J(pts_index, 2, channel_index, :, :)) .* weight_map;
            end
        end
        weightedJ = reshape(weightedJ, num_pts, 2, []);                                      % num_pts x 2 x (C x height_kernel x width_kernel), note the the reshape is different in MATLAB and python
        J = reshape(J, num_pts, 2, []);                                                      % num_pts x 2 x (C x height_kernel x width_kernel)
    else
        weightedJ = reshape(J, num_pts, 2, []);
        J = weightedJ;
    end

    % compute the hessian matrix
    % size(weightedJ)
    % size(J)

    permuted_J = permute(J, [1, 3, 2]);             % num_pts x (C x height_kernel x width_kernel) x 2
    H = matrix_bmm(weightedJ, permuted_J);                 % num_pts x 2 x 2

    prefix_operator = matrix_bmm(H, weightedJ);            % num_pts x 2 x (C x height_kernel x width_kernel)

    % H = zeros(num_pts, 2, 2);
    % prefix_operator = zeros(num_pts, 2, im_channel*height_kernel*width_kernel);
    % for pts_index = 1:num_pts
        % H(pts_index, :, :) = squeeze(weightedJ(pts_index, :, :)) * squeeze(J(pts_index, :, :))';                               % num_pts x 2 x 2
        % prefix_operator(pts_index, :, :) = squeeze(H(pts_index, :, :)) \ squeeze(weightedJ(pts_index, :, :));                  % num_pts x 2 x (C x height_kernel x width_kernel)
    % end

    prefix_operator(1, :, 1)
    prefix_operator(1, :, end)
    % prefix_operator(2, :, 1)
    % prefix_operator(2, :, end)
    % H
    % pause;


    % loop till converge
    velocity = zeros(num_pts, 2);

    % u = 0;
    % v = 0;
    velocity_old = zeros(num_pts, 2);
    % old_u = 0;
    % old_v = 0;
    for iter_index = 1 : max_iter
        % warp the image in the next frame
        velocity_old_repeat = repmat(velocity_old, 1, 2);
        rect_new = rect - velocity_old_repeat;                  % num_pts x 4

        % rect_new = [rect(1) - old_u, rect(2) - old_v, rect(3) - old_u, rect(4) - old_v];
        % cropped_patch = crop_interp(It1, input_format, rect_new);
        cropped_patch = crop_interp_batch(It1, input_format, rect_new);         % num_pts x C x height_kernel x width_kernel

        % compute the residual
        gradient_time = cropped_patch - cropped_template;                       % num_pts x C x height_kernel x width_kernel

        squeeze(gradient_time(1, 1, :, :))
        % squeeze(gradient_time(2, 1, :, :))
        pause;

        % compute the delta warp function
        % b = gradient_T_temp(:);                                               % num_pts x (C x height_kernel x width_kernel) x 1
        batch_gradient_time = reshape(gradient_time, num_pts, []);
        % V = prefix_operator * b;      

        % size(prefix_operator)
        % size(batch_gradient_time)
        %     'ssssss'

        V = matrix_bmm(prefix_operator, batch_gradient_time);                         % num_pts x 2 x 1
        size(V)
        V

        % accumulate the transilation
        u = old_u + V(1, 1);            
        v = old_v + V(2, 1);
        
        % check the change is below a threshold
        velocity_distance = sqrt((old_u - u)^2 + (old_v - v)^2);
        if velocity_distance < epsilon
            num_iter = iter_index;
            break;
        end
        
        old_u = u;
        old_v = v;
    end
end
