close all;
clear;
clc;

%% configuration
num_epoch = 30;
classes = 26;
number_units = 400;
input_size = 32;
layers = [input_size*input_size, number_units, classes];
learning_rate = 0.01;

load('../data/nist26_train.mat', 'train_data', 'train_labels')
load('../data/nist26_test.mat', 'test_data', 'test_labels')
load('../data/nist26_valid.mat', 'valid_data', 'valid_labels')

[W, b] = InitializeNetwork(layers);
train_acc = zeros(num_epoch, 1);
train_loss = zeros(num_epoch, 1);
valid_loss = zeros(num_epoch, 1);
valid_acc = zeros(num_epoch, 1);

%% visualize the feature map after initialization
number_visualize = 64;
visualize_id = randperm(number_units, number_visualize);
featureMap = W{1}(visualize_id, :);
map = zeros(input_size, input_size, 1, number_visualize);
for i = 1:number_visualize
    feature_temp = featureMap(i, :);
    min_temp = min(feature_temp);
    feature_temp = feature_temp - min_temp;
    max_temp = max(feature_temp);
    feature_temp = feature_temp ./ max_temp;
    map_temp = reshape(feature_temp, input_size, input_size);
%     map_temp = (map_temp + min_temp)./max_temp;
    map(:, :, 1, i) = map_temp;
end
figure;
montage(map);
% print('../writeup/feature_map_before.eps', '-depsc');

%% train the network
for j = 1:num_epoch
    [W, b] = Train(W, b, train_data, train_labels, learning_rate);
    
    [train_acc(j), train_loss(j)] = ComputeAccuracyAndLoss(W, b, train_data, train_labels);
    [valid_acc(j), valid_loss(j)] = ComputeAccuracyAndLoss(W, b, valid_data, valid_labels);
    
    fprintf('Epoch %d - accuracy: %.5f, %.5f \t loss: %.5f, %.5f \n', j, train_acc(j), valid_acc(j), train_loss(j), valid_loss(j));
end

%% visualize the feature map after training
featureMap = W{1}(visualize_id, :);
map = zeros(input_size, input_size, 1, number_visualize);
for i = 1:number_visualize
    feature_temp = featureMap(i, :);
    min_temp = min(feature_temp);
    feature_temp = feature_temp - min_temp;
    max_temp = max(feature_temp);
    feature_temp = feature_temp ./ max_temp;
    map_temp = reshape(feature_temp, input_size, input_size);
%     map_temp = (map_temp + min_temp)./max_temp;
    map(:, :, 1, i) = map_temp;
end
figure;
montage(map);
% print('../writeup/feature_map_after.eps', '-depsc');

%% test the network with the test dataset
[test_acc, test_loss] = ComputeAccuracyAndLoss(W, b, test_data, test_labels);

%% visualize the confusion matrix
outputs = Classify(W, b, test_data);		
[~, cm] = confusion(test_labels', outputs');
cm = cm./max(max(cm));
cm = imresize(cm, [classes*10, classes*10], 'nearest');
figure;
imshow(cm);
% print('../writeup/confusion_matrix.eps', '-depsc2');

%% plot the performance
figure;
plot(1:num_epoch, train_acc);
hold on;
plot(1:num_epoch, valid_acc);
hold off;
lg = legend('training dataset', 'validation dataset', 'Location', 'northwest');
lg.FontSize = 26;
title('Classification Rate over Epoch', 'FontSize', 26);
xlabel('Epoch', 'FontSize', 26);
ylabel('Accuracy', 'FontSize', 26);
set(gca, 'fontsize', 16);
% print('../writeup/accuracy_train26.eps', '-depsc');

figure;
plot(1:num_epoch, train_loss);
hold on;
plot(1:num_epoch, valid_loss);
hold off;
lg = legend('training dataset', 'validation dataset');
lg.FontSize = 26;
title('Cross-Entropy Loss over Epoch', 'FontSize', 26);
xlabel('Epoch', 'FontSize', 26);
ylabel('Loss', 'FontSize', 26);
set(gca, 'fontsize', 16);
% print('../writeup/loss_train26.eps', '-depsc');

%% save the model and parameters
% save_name = sprintf('nist26_model_train26_%s_%s.mat', num2str(num_epoch), num2str(learning_rate));
% save(save_name, 'W', 'b', 'num_epoch', 'learning_rate', 'train_acc', 'valid_acc', 'train_loss', 'valid_loss', 'test_acc', 'test_loss');
