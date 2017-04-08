close all;
clear;
clc;

num_epoch = 30;
classes = 26;
layers = [32*32, 400, classes];
learning_rate = 0.01;

load('../data/nist26_train.mat', 'train_data', 'train_labels')
load('../data/nist26_test.mat', 'test_data', 'test_labels')
load('../data/nist26_valid.mat', 'valid_data', 'valid_labels')

[W, b] = InitializeNetwork(layers);
train_acc = zeros(num_epoch, 1);
train_loss = zeros(num_epoch, 1);
valid_loss = zeros(num_epoch, 1);
valid_acc = zeros(num_epoch, 1);
number_layer = length(layers);
number_check = 5;        % number of weight for checking in each layer
theta = 1e-7;

check_sample_id = randperm(size(train_data, 1), 1);
data_temp = train_data(check_sample_id, :)';  % N x 1
label_temp = train_labels(check_sample_id, :)';      % Cx1 wait for checking
[~, act_h_temp, act_a_temp] = Forward(W, b, data_temp);
[grad_W, grad_b] = Backward(W, b, data_temp, label_temp, act_h_temp, act_a_temp);

% check the gradient of weight W for each layer
fprintf('start checking %d randomly chosen weight W at each layer\n', number_check);
for m = 1:number_layer - 1
    fprintf('checking layer %d\n', m);
    dim1 = randperm(size(W{m}, 1), number_check);   % randomly choose some weight to check
    dim2 = randperm(size(W{m}, 2), number_check);
    for n = 1:number_check           % check each weight individually
        new_W_plus = W;
        new_W_minus = W;
        new_W_plus{m}(dim1(n), dim2(n)) = new_W_plus{m}(dim1(n), dim2(n)) + theta;
        new_W_minus{m}(dim1(n), dim2(n)) = new_W_minus{m}(dim1(n), dim2(n)) - theta;
        
        [output_W_plus, ~, ~] = Forward(new_W_plus, b, data_temp);
        [output_W_minus, ~, ~] = Forward(new_W_minus, b, data_temp);
        
        loss_plus_W = -log(output_W_plus' * label_temp);        % new computed loss
        loss_minus_W = -log(output_W_minus' * label_temp);
        
        grad_check = (loss_plus_W - loss_minus_W)/(2*theta);    % gradient with respect to W
        grad_W_temp = grad_W{m}(dim1(n), dim2(n));
        
        % compute the relative error 
        error = abs(grad_check - grad_W_temp)/max(abs(grad_check), abs(grad_W_temp));
        
        if error > 1e-1
            disp('gradient error!!!');
            keyboard;
        end
    end
end
disp('no gradient error found in weight W');

% check the gradient of bias b for each layer
fprintf('start checking %d randomly chosen bias b at each layer\n', number_check);
for m = 1:number_layer - 1
    fprintf('checking layer %d\n', m);
    dim1 = randperm(size(b{m}, 1), number_check);   % randomly choose some weight to check
    for n = 1:number_check           % check each weight individually
        new_b_plus = b;
        new_b_minus = b;
        new_b_plus{m}(dim1(n)) = new_b_plus{m}(dim1(n)) + theta;
        new_b_minus{m}(dim1(n)) = new_b_minus{m}(dim1(n)) - theta;
        
        [output_b_plus, ~, ~] = Forward(W, new_b_plus, data_temp);
        [output_b_minus, ~, ~] = Forward(W, new_b_minus, data_temp);
        
        loss_plus_b = -log(output_b_plus' * label_temp);        % new computed loss
        loss_minus_b = -log(output_b_minus' * label_temp);
        
        grad_check = (loss_plus_b - loss_minus_b)/(2*theta);    % gradient with respect to b
        grad_b_temp = grad_b{m}(dim1(n));
        
        % compute the relative error
        error = abs(grad_check - grad_b_temp)/max(abs(grad_check), abs(grad_W_temp));
        if error > 1e-1
            disp('gradient error!!!');
            keyboard;
        end
    end
end
disp('no gradient error found in bias b');