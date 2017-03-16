% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

close all;
clc;
clear;

full_list = get_subfolder_list('../../xinshuo_toolbox');
test_depth1 = get_subfolder_list('../../xinshuo_toolbox', 1); % test depth equals 1
test_depth2 = get_subfolder_list('../../xinshuo_toolbox', 2); % test depth equals 1