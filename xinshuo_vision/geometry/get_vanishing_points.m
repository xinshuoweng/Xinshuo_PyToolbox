% Author: Xinshuo
% Email: xinshuow@andrew.cmu.edu

% this function takes an image in and output the dominate vanishing points
function VPs = get_vanishing_points(img, num_vp, num_iter, debug_mode, vis_mode, error_threshold)
	if nargin < 6
		error_threshold = 10;
	end

	if nargin < 5
		vis_mode = false;
	end

	if nargin < 4 
		debug_mode = true;
	end

	if nargin < 3
		num_iter = 1000;
	end

	if nargin < 2
		num_vp = 3;
	end

	% extract the line segments from the image
	im_gray = rgb2gray(img);

	im_height = size(im_gray, 1);
	im_width = size(im_gray, 2);
	length_diagonal = sqrt(im_height ^ 2 + im_width ^ 2);
	lines = APPgetLargeConnectedEdges(im_gray, 0.025 * length_diagonal);
	% lines

	% [index_line, centroid] = kmeans(lines(:, 5), num_vp);

	% compute the vanishing point inside each cluster
	% VPs = zeros(num_vp, 2);
	% corresponding_lines = cell(num_vp, 1);
	% for cluster_index = 1:num_vp
	% 	lines_tmp = lines(index_line == cluster_index, :);			% num_lines x 6

	% 	% RANSAC to compute the vanishing point
	% 	num_lines = size(lines_tmp, 1);
	% 	biggest_num_valid = 0;
	% 	for iter_index = 1:num_iter
	% 		line1_index = randi(num_lines);
	% 		line2_index = randi(num_lines-1);
	% 		if line2_index >= line1_index
	% 			line2_index = line2_index + 1;
	% 		end

	% 		pts1 = lines_tmp(line1_index, 1:2);
	% 		pts2 = lines_tmp(line1_index, 3:4);
	% 		pts3 = lines_tmp(line2_index, 1:2);
	% 		pts4 = lines_tmp(line2_index, 3:4);

	% 		line1_2d = get_2dline_from_pts(pts1, pts2, debug_mode);
	% 		line2_2d = get_2dline_from_pts(pts3, pts4, debug_mode);

	% 		pts_infinity = get_pts_from_2dlines(line1_2d, line2_2d, debug_mode);

	% 		% compute the distance for all line within the cluster
	% 		num_valid = 0;
	% 		lines_valid = zeros(num_lines, 6);
	% 		for line_index = 1:num_lines
	% 			pts_test1 = lines_tmp(line_index, 1:2);
	% 			pts_test2 = lines_tmp(line_index, 3:4);
	% 			line_test = get_2dline_from_pts(pts_test1, pts_test2, debug_mode);

	% 			distance = get_distance_from_pts_2dline(pts_infinity, line_test, debug_mode);
	% 			if distance < error_threshold
	% 				num_valid = num_valid + 1;
	% 				lines_valid(num_valid, :) = lines_tmp(line_index, :);
	% 			end
	% 		end

	% 		if num_valid > biggest_num_valid
	% 			biggest_num_valid = num_valid;
	% 			best_vp = pts_infinity;
	% 			best_lines = lines_valid(1:num_valid, :);
	% 		end
	% 	end

	% 	best_vp = best_vp / best_vp(3);
	% 	VPs(cluster_index, :) = best_vp(1:2);
	% 	size(best_lines)
	% 	corresponding_lines{cluster_index} = best_lines;
	% end

	% VPs

	if vis_mode
		% save_path = '';
		% label = false;
		% label_str = '';
		% vis_radius = 10;
		% vis_resize_factor = 1;
		% closefig = false;
		% img_with_pts = visualize_image_with_pts(img, VPs', vis_mode, debug_mode, save_path, label, label_str, vis_radius, vis_resize_factor, closefig);
		% size(corresponding_lines{1})
		% size(corresponding_lines{2})
		% size(corresponding_lines{3})
		% for cluster_index = 1:num_vp
		% 	hold on; plot(corresponding_lines{cluster_index}(:, [1 2])', corresponding_lines{cluster_index}(:, [3 4])', 'r');
		% end
		lines
		index = lines(:, 7) > 50000

		figure(1); hold off; imshow(im_gray);
		figure(1); hold on; plot(lines(index, [1 2])', lines(index, [3 4])');
		% figure(1); hold on; plot(VPs(:, 1), VPs(:, 2), 'r', 'MarkerSize', 10);
	end
end