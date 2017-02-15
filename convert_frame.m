% this script is for resizing the original raw frame from towncenter to different size for evaluation

close all; clear; clc;

%% inputs to function
data_base = '../4/';
frame_list = dir(fullfile([data_base, '*.png']));
fnum      = length(frame_list);        % 54, 105, 111, 214, 256, 300
save_dir = '../4_resize_640_480';
resize_size = [480, 640];

if ~exist(save_dir, 'dir')
	mkdir(save_dir);
end

number_resize = size(resize_size, 1);
for j = 1:number_resize
	resize_temp = resize_size(j, :);

	for i = 1:fnum
		image_path_temp = fullfile([data_base, frame_list(i).name]);
		image_temp = imread(image_path_temp);
		image_temp = imresize(image_temp, resize_temp);

		name = frame_list(i).name;
		index2 = strfind(name, '.png');
		id = str2double(name(index2-5:index2-1));
		% fprintf('%d', id);
		% pause(1);
		save_dir_temp = fullfile([save_dir, sprintf('/frame%05d.png', id)]);
		imwrite(image_temp, save_dir_temp);
		fprintf('loading image %d/%d, resize [%4d, %4d]\n', i, fnum, resize_temp(2), resize_temp(1));
	end

end
% fname_in  = fullfile(fpath_in,  sprintf('frame%06d.jpg', fnum));
% fname_out = fullfile(fpath_out, sprintf('frame%06d.jpg', fnum));

% crop.w = 47;
% crop.h = 87;


% %% main function: crop_location
% %- draw rectangle
% pos.x = 224; pos.y = 74;
% pos.rec = [pos.x, pos.y, crop.w, crop.h];
% img = imread(fname_in);
% figure; imshow(img); hold on;
% rectangle('Position', pos.rec); 

% %- crop
% img_crop = img(pos.y:pos.y+crop.h, pos.x:pos.x+crop.w, :);
% %figure; imshow(img_crop);

% %- save
% imwrite(img_crop, fname_out);
