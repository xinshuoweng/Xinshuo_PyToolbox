% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

clc;
close;
clear;


generate_video_from_folder('output/non-overlap/0/images', 'output/non-overlap/0', '0_image');
generate_video_from_folder('output/non-overlap/0/features', 'output/non-overlap/0', '0_features');

generate_video_from_folder('output/non-overlap/1/images', 'output/non-overlap/1', '1_image');
generate_video_from_folder('output/non-overlap/1/features', 'output/non-overlap/1', '1_features');