% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% input
%           It:                 feature at t
%           It1:                feature at t+1
%           rect:               num_pts x 4, the window for every point
%           input_format:       denoted for the input feature:   hw-> grayscale image, hwc-> multichannel feature or rgb image, chw, transposed multichannel features
%           max_iter:           maximum number of iteration
%           weight:             weighted window for computing Jacobian
%           epsilon:            criterion for stopping the LK
% output
%           velocity:           movement vector num_pts x 2(x, y)
%           num_iter_vec:       number of iteration to stop for every point
function [velocity, num_iter_vec] = LucasKanadeInverseCompositional(It, It1, rect, input_format, max_iter, weight, epsilon, warning_mode, debug_mode)
    if nargin < 9
        debug_mode = true;
    end

    if nargin < 8
        warning_mode = true;
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

    if warning_mode
        fprintf('precomputing the Hessian\n');
    end

    cropped_template = crop_interp_batch(It, input_format, rect, debug_mode);                   % num_pts x C x height_kernel x width_kernel
    [~, ~, Iy, Ix] = gradient(cropped_template);                                                % num_pts x C x height_kernel x width_kernel,       note that the gradient along the edge is different in MATLAB and python, but same after adding weight

    width_kernel = size(Ix, 4);
    height_kernel = size(Ix, 3);
    im_channel = size(Ix, 2);
    num_pts = size(Ix, 1);
    weight_map = generate_weight([height_kernel, width_kernel], debug_mode);                    % height_kernel x width_kernel

    % compute the Jacobian
    J = zeros(num_pts, 2, im_channel, height_kernel, width_kernel);                             % num_pts x 2 x C x height_kernel x width_kernel     
    J(:, 1, :, :, :) = Ix;
    J(:, 2, :, :, :) = Iy;

    % squeeze(J(1, 1, :, :))
    if weight
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
    permuted_J = permute(J, [1, 3, 2]);             % num_pts x (C x height_kernel x width_kernel) x 2
    H = matrix_bmm(weightedJ, permuted_J, debug_mode);                 % num_pts x 2 x 2
    prefix_operator = matrix_bmm(H, weightedJ, debug_mode, 'pinv_mul');            % num_pts x 2 x (C x height_kernel x width_kernel)

    % loop till converge
    if warning_mode
        fprintf('iterating LK algorithm\n');
    end
    velocity = zeros(num_pts, 2);
    velocity_old = zeros(num_pts, 2);
    todo_mask = ones(num_pts, 1);
    num_iter_vec = zeros(num_pts, 1);
    for iter_index = 1 : max_iter
        % todo_mask
        todo_index = find(todo_mask == 1);
        num_pts_todo = sum(todo_mask);

        % update the bbox        
        velocity_old_repeat = repmat(velocity_old, 1, 2);
        rect_new = rect - velocity_old_repeat;                  % num_pts_todo x 4
        rect_new_todo = rect_new(todo_index, :);

        % warp the image in the next frame
        cropped_patch = crop_interp_batch(It1, input_format, rect_new_todo, debug_mode);         % num_pts_todo x C x height_kernel x width_kernel

        % compute the residual
        gradient_time = cropped_patch - cropped_template(todo_index, :, :, :);                       % num_pts_todo x C x height_kernel x width_kernel
        batch_gradient_time = reshape(gradient_time, num_pts_todo, []);

        % compute the update
        velocity_update = matrix_bmm(prefix_operator(todo_index, :, :), batch_gradient_time, debug_mode);                         % num_pts_todo x 2
        velocity_update_all = zeros(num_pts, 2);
        velocity_update_all(todo_index, :) = velocity_update;

        % accumulate the transilation
        velocity = velocity_old + velocity_update_all;
        
        % check the change is below a threshold
        velocity_distance = matrix_norm_rowwise(velocity_update_all);
        converged_index = find(velocity_distance < epsilon);
        num_iter_vec(intersect(converged_index, todo_index)) = iter_index;
        todo_mask(converged_index) = 0;                                    % mask out the converged points
        if sum(todo_mask) == 0
            break;
        end

        if warning_mode
            fprintf('iter: %d, %.2f%% points have converged\n', iter_index, length(converged_index)*100.0/num_pts);
        end
        
        velocity_old = velocity;
    end
end