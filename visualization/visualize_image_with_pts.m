% Author: Xinshuo Weng
% email: xinshuo.weng@gmail.com

% this function takes an image in and plot points on top of it
% parameter:
%	img:		an image read from imread()
%	pts_array:	2 x num_pts matrix to represent (x, y) locations
%	label:		a logical value to judge if display a text for every points
%	label_str:	a cell array where every cell has a string inside
function img_with_pts = visualize_image_with_pts(img, pts_array, vis, debug_mode, save_path, label, label_str)
	if ~exist('debug_mode', 'var')
		debug_mode = true;
	end

	if ~exist('vis', 'var')
		vis = false;
	end

	if ~exist('label', 'var')
		label = true;
	end

	if debug_mode
		assert(isImage(img), 'the input is not an image format.');
		assert(size(pts_array, 1) == 2 && size(pts_array, 2) >= 0, 'shape of points to draw is not correct.');
	end

	% draw image and points
	title('points prediction.'); imshow(img); hold on;
	x = pts_array(1, :);
	y = pts_array(2, :);
	plot(x, y, 'rx', 'MarkerSize', 10);
	
	% add labels
	if label
		num_pts = size(pts_array, 2);
		if exist('label_str', 'var') && debug_mode
			assert(iscell(label_str), 'the label string is not a cell.');
			assert(size(label_str, 1) == 1 && size(label_str, 2) == num_pts, 'shape of label string cell is not correct');
			assert(all(cellfun(@(x) ischar(x), label_str)), 'the elements in the label string cell are not all string.');
		else
			label_str = cell(num_pts, 1);
			for i = 1:num_pts
			    label_str{i} = sprintf('%d', i);
			end
		end
		text(x, y, label_str, 'Color', 'y');
	end

	% get the current frame to return
	img_with_pts = getframe;
	img_with_pts = img_with_pts.cdata;

	% save
	if exist('save_path', 'var')
		assert(ischar(save_path), 'save path is not correct.');
		im_size = size(img);
		save_figure(save_path, im_size);
		fprintf('save image to %s\n', save_path);
	end
end
