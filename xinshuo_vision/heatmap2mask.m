% this script is to generate 0-1 mask from the heatmap by cutting a
% specific threshold
close all;
clear;
clc;

% source folder path
heatmap_path_base = '/home/xinshuo/workspace/MATLAB/pedestrian-detection/testing/images';
imagelist = dir(fullfile(heatmap_path_base, 'output2*.png'));

% save folder path
mask_path_base = '/home/xinshuo/workspace/MATLAB/pedestrian-detection/testing/masks';
if ~exist(mask_path_base, 'dir')
    mkdir(mask_path_base);
end

threshold = 0.1;

for i = 1:size(imagelist, 1)
    name = imagelist(i).name;
    image_path = fullfile(heatmap_path_base, name);
    image = im2double(imread(image_path));
    %     imshow(image);
    image(image(:) > threshold) = 1;
    image(image(:) <= threshold) = 0;
    
    save_path = fullfile(mask_path_base, name);
    imwrite(image, save_path);
    %     imshow(image);
end
