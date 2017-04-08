function [W, b] = Train(W, b, train_data, train_label, learning_rate)
% [W, b] = Train(W, b, train_data, train_label, learning_rate) trains the network
% for one epoch on the input training data 'train_data' and 'train_label'. This
% function should returned the updated network parameters 'W' and 'b' after
% performing backprop on every data sample.


% This loop template simply prints the loop status in a non-verbose way.
% Feel free to use it or discard it

number_data = size(train_data, 1);
% id = 1:number_data;
shuffle_id = randperm(number_data);
train_data = train_data(shuffle_id, :);
train_label = train_label(shuffle_id, :);
for i = 1:number_data
    data_temp = train_data(i, :)';  % N x 1
    label_temp = train_label(i, :)';      % Cx1 
    [~, act_h_temp, act_a_temp] = Forward(W, b, data_temp);
    [grad_W, grad_b] = Backward(W, b, data_temp, label_temp, act_h_temp, act_a_temp);
    [W, b] = UpdateParameters(W, b, grad_W, grad_b, learning_rate);

    if mod(i, 100) == 0
        fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
        fprintf('Done %.2f %%', i/size(train_data,1)*100)
    end
end
fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')


end
