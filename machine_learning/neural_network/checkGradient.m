close all;
clear;
clc;

num_epoch = 30;
classes = 10;
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


