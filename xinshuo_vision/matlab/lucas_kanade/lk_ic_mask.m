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
function [velocity, num_iter_vec] = lk_ic_mask(It, It1, rect, input_format, max_iter, weight, epsilon, warning_mode, debug_mode)
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

    save_hist = false;
    % mask_template = true;

    % convert grayscale to 2d image
    It = squeeze(It);
    It1 = squeeze(It1);

    if warning_mode
        fprintf('precomputing the Hessian\n');
    end

    cropped_template = crop_interp_batch(It, input_format, rect, debug_mode);                   % num_pts x C x height_kernel x width_kernel
    
    zero_mask = find(cropped_template == 0);
    % zero_mask
    [sparsity_template, ~] = compute_sparsity(cropped_template);                  
    fprintf('sparsity of cropped template is %f\n', sparsity_template);

    % compute_sparsity(cropped_template(zero_mask))
    % size(zero_mask)
    % pause

    [~, ~, Iy, Ix] = gradient(cropped_template);                                                % num_pts x C x height_kernel x width_kernel,       note that the gradient along the edge is different in MATLAB and python, but same after adding weight

    Ix(zero_mask) = 0;
    Iy(zero_mask) = 0;

    % compute_sparsity(Ix)
    % compute_sparsity(Iy)
    % masked_grad_x = Ix(zero_mask);
    % masked_grad_y = Iy(zero_mask);
    % compute_sparsity(masked_grad_x)
    % compute_sparsity(masked_grad_y)
    % pause;
    % squeeze(cropped_template(1, 1, :, :))
    % squeeze(cropped_template(1, 2, :, :))
    % squeeze(cropped_template(1, 3, :, :))
    % squeeze(cropped_template(1, 4, :, :))
    % squeeze(cropped_template(1, 5, :, :))
    % squeeze(cropped_template(1, 6, :, :))
    % squeeze(cropped_template(1, 7, :, :))
    % squeeze(cropped_template(1, 8, :, :))
    % squeeze(cropped_template(1, 9, :, :))
    % squeeze(cropped_template(1, 10, :, :))

    width_kernel = size(Ix, 4);
    height_kernel = size(Ix, 3);
    im_channel = size(Ix, 2);
    num_pts = size(Ix, 1);
    weight_map = generate_weight([height_kernel, width_kernel], debug_mode);                    % height_kernel x width_kernel

    % compute the Jacobian
    J = zeros(num_pts, 2, im_channel, height_kernel, width_kernel);                             % num_pts x 2 x C x height_kernel x width_kernel     
    J(:, 1, :, :, :) = Ix;
    J(:, 2, :, :, :) = Iy;                                                                      % sparsity of J is larger or equal than the input template as flat area exists

    [sparsity_jacobian, ~] = compute_sparsity(J)             ;
    fprintf('sparsity of Jacobian is %f\n', sparsity_jacobian);
    assert(sparsity_jacobian >= sparsity_template);

    % squeeze(J(1, 1, :, :))
    % weightedJ_mask = zeros(num_pts, 2, )
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

        % weightedJ_mask = weightedJ
    else
        weightedJ = reshape(J, num_pts, 2, []);                                              % sparsity of weighted J is larger or equal than the J after weighting
        J = weightedJ;                                                                              
    end


    % arr_test = squeeze(weightedJ(:, 1, :));
    [sparsity_weighted_jacobian, ~] = compute_sparsity(weightedJ);                                    
    fprintf('sparsity of weighted Jacobian is %f\n', sparsity_weighted_jacobian);
    assert(sparsity_weighted_jacobian >= sparsity_jacobian);
    % index1
    % compute_sparsity(arr_test(zero_mask))
    % pause

    % compute the hessian matrix
    permuted_J = permute(J, [1, 3, 2]);                                                     % num_pts x (C x height_kernel x width_kernel) x 2
    H = matrix_bmm(weightedJ, permuted_J, debug_mode);                                      % num_pts x 2 x 2
    prefix_operator = matrix_bmm(H, weightedJ, debug_mode, 'pinv_mul');                     % num_pts x 2 x (C x height_kernel x width_kernel)

    % H
    [sparsity_prefix, ~] = compute_sparsity(prefix_operator);                                    
    fprintf('sparsity of prefix operator is %f\n', sparsity_prefix);
    assert(sparsity_prefix <= sparsity_weighted_jacobian);                                  % sparsity of prefix operator is less or equal than the weighted Jacobian as there are some partial 0 in x or y direction in weighed Jacobian
    
    % index_bad = find(ismember(index1, index2) ~= 0);
    % length(index_bad)
    % [pts_index, ppp_index, n_index] = ind2sub([num_pts, 1, im_channel*height_kernel*width_kernel], index1(index_bad));
    % size(pts_index)
    % pts_index
    % ppp_index

    % for i = 1:length(index_bad)
        % test_index = index_bad(i);
        % test_index = i;
        % test_index = index1(index_bad(i));
        % weightedJ_tmp = weightedJ(pts_index(test_index), :, n_index(test_index));
        % tmp = prefix_operator(pts_index(test_index), :, n_index(test_index));
        % weightedJ_tmp
        % tmp
        % pause
    % end
    % prefix_operator(pts_index())
    % intersect(index1, index2)
    % ratio
    % all(index1 == index2)
    % index_bad = find(prefix_operator(:, 1, :) ~= 0);
    % length(index_bad)
    % [pts_index, dummy, n_index] = ind2sub([num_pts, 1, im_channel*height_kernel*width_kernel], index_bad);
    % dummy
    % prefix_operator(pts_index(1), :, n_index(1));
    % pause;

    % save histogram of template and gradient
    if save_hist
        vectorized = cropped_template(:);
        index_0 = find(vectorized == 0);
        ratio = length(index_0) / length(vectorized);
        figure; histogram(vectorized, 100);
        title(sprintf('ratio: %f', ratio));
        print('hist_cropped_template.png', '-dpng');

        vectorized = J(:);
        histogram(vectorized, 100)
        index_0 = find(vectorized == 0);
        ratio = length(index_0) / length(vectorized);
        figure; histogram(vectorized, 100);
        title(sprintf('ratio: %f', ratio));
        print('hist_jacobian.png', '-dpng');
    end

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
            fprintf('iter: %d, %.2f%% points have converged.\n', iter_index, length(converged_index)*100.0/num_pts);
        end
        
        velocity_old = velocity;


        % converged_index
        velocity_update_all
        % pause;
    end
end