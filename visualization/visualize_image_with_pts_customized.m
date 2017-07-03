% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function takes an image in and plot points on top of it
% Note that this function is different from visualize_image_with_pts because we change the pixel value directly on the image without using MATLAB plot function
% parameter:
%	img:		an image read from imread()
%	pts_array:	2 x num_pts matrix to represent (x, y) locations
%	label:		a logical value to judge if display a text for every points
%	label_str:	a cell array where every cell has a string inside
function img_with_pts = visualize_image_with_pts_customized(img, pts_array, vis, debug_mode, save_path)
	if ~exist('debug_mode', 'var')
		debug_mode = true;
	end

	if ~exist('vis', 'var')
		vis = false;
	end

	if debug_mode
		assert(isImage(img), 'the input is not an image format.');
		assert(size(pts_array, 1) == 2 && size(pts_array, 2) >= 0, 'shape of points to draw is not correct.');
		assert(isIntegerImage(img), 'the input image should be an integer image.\n');
	end

	% draw image and points
	time = tic;
	elapsed = toc(time);
	radius = 1;
	num_pts = size(pts_array, 2);
	color_pixel = [255, 0, 0];			% red
	
	x = pts_array(1, :);
	y = pts_array(2, :);

	% color_pixel = reshape(color_pixel, [1, 1, 3]);		
	% for pts_index = 1:num_pts
	% 	img(y(pts_index)-radius:y(pts_index)+radius, x(pts_index), :) = repmat(color_pixel, [3, 1, 1]);
	% 	img(y(pts_index), x(pts_index)-radius:x(pts_index)+radius, :) = repmat(color_pixel, [1, 3, 1]);
	% end

	x_index_list = [];
	y_index_list = [];
	z_index_list = [];
	for pts_index = 1:num_pts 
		for x_index = x(pts_index)-radius:1:x(pts_index)+radius
			for z_index = 1:3
				y_index_list = [y_index_list, y(pts_index)];
				x_index_list = [x_index_list, x_index];
				z_index_list = [z_index_list, z_index];
			end
		end

		for y_index = y(pts_index)-radius:1:y(pts_index)+radius
			for z_index = 1:3
				y_index_list = [y_index_list, y_index];
				x_index_list = [x_index_list, x(pts_index)];
				z_index_list = [z_index_list, z_index];
			end
		end
	end
	img(sub2ind(size(img), y_index_list, x_index_list, z_index_list)) = repmat(color_pixel, [1, length(y_index_list)/3]);
	assign_time = toc(time);
	fprintf('time spent on assigning value is %f seconds\n', assign_time - elapsed);

	if vis
		title('points prediction.'); 
		imshow(img);
	end
	
	% % add labels
	% if label
	% 	if exist('label_str', 'var')
	% 		if debug_mode
	% 			assert(iscell(label_str), 'the label string is not a cell.');
	% 			assert(size(label_str, 1) == 1 && size(label_str, 2) == num_pts, 'shape of label string cell is not correct');
	% 			assert(all(cellfun(@(x) ischar(x), label_str)), 'the elements in the label string cell are not all string.');
	% 		end
	% 	else
	% 		label_str = cell(1, num_pts);
	% 		for i = 1:num_pts
	% 		    label_str{i} = sprintf('%d', i);
	% 		end
	% 	end

	% 	text(x, y, label_str, 'Color', 'y');
	% end

	% save
	if exist('save_path', 'var')
		assert(ischar(save_path), 'save path is not correct.');
		imwrite(img, save_path);
		fprintf('save image to %s\n', save_path);
	end
	save_time = toc(time);
	fprintf('time spent on saving is %f seconds\n', save_time - assign_time);

	img_with_pts = img;
end
